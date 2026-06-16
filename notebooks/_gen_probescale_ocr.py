"""Generator for the ProbeScale-for-OCR Colab notebook.

Run: python notebooks/_gen_probescale_ocr.py
Produces: notebooks/ProbeScale_OCR_TrOCR.ipynb
"""
import json
import os

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ---------------------------------------------------------------------------
md(r"""# ProbeScale for AI-OCR — TrOCR エンコーダのプロービング圧縮

> 論文: **ProbeScale: Probing Analysis to Optimize Neural Scaling Laws for Efficient Small Language Model Inference** (Sourav Das, arXiv:2606.01806)

このノートブックは ProbeScale のアルゴリズムを **AI OCR（画像から文字列を抽出するタスク）** に適用して再現実験するものです。Google Colab（GPU ランタイム推奨: T4 以上）での実行を前提としています。

## 何をするのか

ProbeScale は「事前学習済みモデルの各層が、対象タスクにどれだけ寄与しているか」を**軽量なプローブ**で定量化し、**最も寄与の大きい連続した層ブロックだけ**を残してモデルを圧縮するフレームワークです（追加学習はサブネットワークの短い fine-tune のみ）。

| 論文(NLP)での要素 | 本ノートブック(OCR)での対応 |
|---|---|
| ベースモデル RoBERTa-Large / T5-Base | **TrOCR (`microsoft/trocr-base-handwritten`)** の **ビジョンエンコーダ (DeiT, 12層)** |
| 対象タスク SST-2 / QNLI | **手書き行 OCR**（画像→テキスト） |
| データセット GLUE | **`Teklia/IAM-line`**（IAM 手書き行データセット） |
| 層プローブ（線形分類器で性質を予測）| **各エンコーダ層の特徴に載せた線形 CTC プローブ**で文字列を認識 |
| プローブ性能 $V_{p,l}$ = accuracy/F1 | プローブの **文字正解率 (1 − CER)** |
| 層関連度 $R_{l,T}=\sum_p w(p,T)V_{p,l}$ | 単一プローブ・$w=1$（論文の SST-2 設定に対応） |
| 連続ブロック選択 (Algorithm 1) | **そのまま実装**（連続 $k$ 層の総関連度を最大化） |
| サブネット抽出 + fine-tune | エンコーダ層を間引いてデコーダはそのまま、短く fine-tune |

## 最終的に比較するもの

1. **どれくらい圧縮できたか** — 元モデル vs 圧縮モデルのパラメータ数・圧縮率（エンコーダ単体／モデル全体）
2. **精度はどう変化したか** — テストセットの **CER / WER** を、元モデル・ProbeScale・ベースライン (Top-k / Uniform-k) で比較

> **プロービングの注記**: OCR は系列出力タスクなので、論文の「単一ベクトルに対する線形分類プローブ」をそのまま使えません。本ノートブックは各層のパッチトークン系列に**線形 CTC ヘッド**を載せて文字列を読み取らせ、その認識性能を層の関連度 $R_l$ とします。これは ProbeScale の中核思想（プローブ性能を層選択の*相対的*スコアに転用する）に忠実な拡張です。
""")

# ---------------------------------------------------------------------------
md(r"""## 0. セットアップ

GPU ランタイムを選択してください（メニュー: ランタイム → ランタイムのタイプを変更 → ハードウェアアクセラレータ → GPU）。
""")

code(r"""
!pip -q install -U "transformers>=4.40" "datasets>=2.18" jiwer accelerate matplotlib pandas pillow
""")

code(r"""
import os, math, random, copy, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import jiwer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)
""")

# ---------------------------------------------------------------------------
md(r"""### 実験設定（ここを変えれば規模を調整できます）

Colab の無料 GPU(T4) でも数十分〜1時間程度で一周できるよう、各ステップは小さなサブセットを使います。`runtime` に余裕がある場合は件数やエポックを増やすと結果が安定します。
""")

code(r"""
# ===== Experiment configuration =====
MODEL_NAME = "microsoft/trocr-base-handwritten"   # DeiT encoder (12 layers) + text decoder
DATASET    = "Teklia/IAM-line"                     # IAM handwritten lines (image -> text)

# --- probing (ProbeScale Step 1) ---
N_PROBE_TRAIN = 300    # images used to TRAIN the per-layer linear CTC probes
N_PROBE_VAL   = 120    # images used to MEASURE probe performance (= layer relevance R_l)
PROBE_EPOCHS  = 12     # epochs for each linear probe (cheap: only a linear layer is trained)
PROBE_LR      = 1e-3

# --- subnetwork fine-tuning + evaluation ---
N_FT_TRAIN = 1200      # images to fine-tune each extracted subnetwork
N_EVAL     = 400       # test images for final CER / WER
FT_EPOCHS  = 1
FT_LR      = 5e-6
TRAIN_BATCH = 4
EVAL_BATCH  = 16
MAX_LEN     = 64       # max target token length

# --- budgets to evaluate (number of encoder layers to keep) ---
K_BUDGETS = [6, 4]     # full encoder has 12 layers -> 6 = 50%, 4 = 33% of encoder depth
METHODS   = ["ProbeScale", "Top-k", "Uniform-k"]
""")

# ---------------------------------------------------------------------------
md(r"""## 1. データセットの読み込み

`Teklia/IAM-line` は `image`（PIL 画像, 高さ128pxに正規化済み）と `text`（正解文字列）の2列を持ちます。""")

code(r"""
raw = load_dataset(DATASET)
print(raw)

train_split = raw["train"].shuffle(seed=SEED)
val_split   = raw["validation"].shuffle(seed=SEED)
test_split  = raw["test"].shuffle(seed=SEED)

probe_train = train_split.select(range(N_PROBE_TRAIN))
probe_val   = val_split.select(range(min(N_PROBE_VAL, len(val_split))))
ft_train    = train_split.select(range(N_FT_TRAIN))
eval_test   = test_split.select(range(min(N_EVAL, len(test_split))))

# quick peek
ex = probe_train[0]
print("sample text:", repr(ex["text"]))
plt.figure(figsize=(8, 1.5)); plt.imshow(ex["image"], cmap="gray"); plt.axis("off")
plt.title(ex["text"][:80]); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""## 2. ベースモデルの読み込みとベースライン評価

TrOCR は Vision-Encoder-Decoder 構造です。本実験では **エンコーダ（DeiT, 12層）を圧縮対象**にします（デコーダは保持）。これは論文が T5 の *encoder* を圧縮対象にしたのと同じ立場です。
""")

code(r"""
processor  = TrOCRProcessor.from_pretrained(MODEL_NAME)
full_model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
full_model.eval()

# make sure generation ids are set (they are in the pretrained config, but be explicit)
full_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
full_model.config.pad_token_id = processor.tokenizer.pad_token_id
full_model.config.eos_token_id = processor.tokenizer.sep_token_id

L = full_model.config.encoder.num_hidden_layers
HIDDEN = full_model.config.encoder.hidden_size
print(f"Encoder layers L={L}, hidden={HIDDEN}")


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def model_param_report(model):
    total = count_params(model)
    enc   = count_params(model.encoder)
    dec   = count_params(model.decoder)
    return {"total": total, "encoder": enc, "decoder": dec}


full_report = model_param_report(full_model)
print("Full model params (M):",
      {k: round(v/1e6, 1) for k, v in full_report.items()})
""")

code(r"""
@torch.no_grad()
def evaluate_model(model, dataset, n=None, batch=EVAL_BATCH, max_new_tokens=MAX_LEN):
    '''Run the OCR model and return CER / WER plus predictions.'''
    model.eval()
    n = n if n is not None else len(dataset)
    preds, refs = [], []
    for i in range(0, n, batch):
        sub = dataset.select(range(i, min(i + batch, n)))
        images = [im.convert("RGB") for im in sub["image"]]
        pv = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            gen = model.generate(pv, max_new_tokens=max_new_tokens)
        preds += processor.batch_decode(gen, skip_special_tokens=True)
        refs  += sub["text"]
    # jiwer expects non-empty references
    refs_safe  = [r if len(r) > 0 else " " for r in refs]
    preds_safe = [p if len(p) > 0 else " " for p in preds]
    cer = jiwer.cer(refs_safe, preds_safe)
    wer = jiwer.wer(refs_safe, preds_safe)
    return {"cer": cer, "wer": wer}, preds, refs
""")

code(r"""
print("Evaluating the ORIGINAL TrOCR model (baseline)...")
t0 = time.time()
baseline_metrics, base_preds, base_refs = evaluate_model(full_model, eval_test, n=N_EVAL)
print(f"  done in {time.time()-t0:.0f}s")
print(f"  Original TrOCR  CER={baseline_metrics['cer']*100:.2f}%  WER={baseline_metrics['wer']*100:.2f}%")
for p, r in zip(base_preds[:3], base_refs[:3]):
    print("   GT  :", r)
    print("   pred:", p, "\n")
""")

# ---------------------------------------------------------------------------
md(r"""## 3. ProbeScale Step 1 — 層ごとのプロービング解析

論文の Algorithm 1 の `CalculateRelevance` に対応します。

1. 各エンコーダ層 $l$ の隠れ表現 $h_l$ を抽出（`output_hidden_states=True` で1回の forward から全層取得）
2. その層の表現の上に **線形 CTC プローブ** $g_l$ を学習（エンコーダは凍結＝特徴は固定）
3. 検証セットでプローブの **文字正解率 $V_l = 1 - \mathrm{CER}$** を測定
4. 層関連度 $R_{l} = w \cdot V_l$（単一プローブ・$w=1$）

特徴は一度だけ計算してキャッシュします（全層を再計算しないため高速）。
""")

code(r"""
# ----- character vocabulary for the CTC probe (built from probe-train texts) -----
charset = set()
for t in probe_train["text"]:
    charset.update(t)
itos = ["<blank>"] + sorted(charset)     # index 0 = CTC blank
stoi = {c: i for i, c in enumerate(itos)}
V = len(itos)
print("probe vocab size:", V)


def encode_text(t):
    return [stoi[c] for c in t if c in stoi]
""")

code(r"""
@torch.no_grad()
def extract_layer_features(dataset, n):
    '''Forward the (frozen) encoder once per image and cache every layer's token
    sequence. Returns feats[layer] = fp16 CPU tensor of shape (n, T, HIDDEN).'''
    full_model.eval()
    per_layer = [[] for _ in range(L + 1)]   # hidden_states has L+1 entries (emb + L layers)
    targets, target_lens, texts = [], [], []
    for i in range(0, n, EVAL_BATCH):
        sub = dataset.select(range(i, min(i + EVAL_BATCH, n)))
        images = [im.convert("RGB") for im in sub["image"]]
        pv = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            enc_out = full_model.encoder(pv, output_hidden_states=True)
        for li, hs in enumerate(enc_out.hidden_states):
            per_layer[li].append(hs.to("cpu", torch.float16))
        for t in sub["text"]:
            ids = encode_text(t)
            targets.append(torch.tensor(ids, dtype=torch.long))
            target_lens.append(len(ids))
            texts.append(t)
    feats = [torch.cat(chunks, dim=0) for chunks in per_layer]   # list len L+1
    return feats, targets, texts


print("Extracting cached encoder features for probing...")
t0 = time.time()
train_feats, train_targets, _ = extract_layer_features(probe_train, N_PROBE_TRAIN)
val_feats,   _,             val_texts = extract_layer_features(probe_val, len(probe_val))
T_tokens = train_feats[0].shape[1]
print(f"  cached in {time.time()-t0:.0f}s | tokens/image T={T_tokens} | "
      f"approx RAM={sum(f.numel()*2 for f in train_feats)/1e9:.1f}GB (train)")
""")

code(r"""
def ctc_greedy_decode(logits):
    '''logits: (T, V) -> decoded string (collapse repeats, drop blank).'''
    idx = logits.argmax(-1).tolist()
    out, prev = [], None
    for j in idx:
        if j != prev and j != 0:   # 0 = blank
            out.append(itos[j])
        prev = j
    return "".join(out)


def train_probe_for_layer(layer_index):
    '''Train a linear CTC probe on cached features of one layer; return V_l = 1 - CER.'''
    feats = train_feats[layer_index]           # (N, T, HIDDEN) fp16 CPU
    probe = nn.Linear(HIDDEN, V).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    N = feats.shape[0]
    order = np.arange(N)
    for ep in range(PROBE_EPOCHS):
        np.random.shuffle(order)
        for i in range(0, N, TRAIN_BATCH * 4):
            bidx = order[i:i + TRAIN_BATCH * 4]
            x = feats[bidx].to(device, torch.float32)             # (b, T, H)
            logp = probe(x).log_softmax(-1).transpose(0, 1)       # (T, b, V)
            tgt = torch.cat([train_targets[j] for j in bidx]).to(device)
            tgt_len = torch.tensor([len(train_targets[j]) for j in bidx], device=device)
            in_len = torch.full((len(bidx),), T_tokens, dtype=torch.long, device=device)
            # skip samples whose target is longer than T (CTC requirement)
            loss = ctc(logp, tgt, in_len, tgt_len)
            opt.zero_grad(); loss.backward(); opt.step()
    # ---- measure probe CER on validation features ----
    probe.eval()
    vfeats = val_feats[layer_index]
    preds = []
    with torch.no_grad():
        for i in range(0, vfeats.shape[0], EVAL_BATCH):
            x = vfeats[i:i + EVAL_BATCH].to(device, torch.float32)
            lg = probe(x)
            for b in range(x.shape[0]):
                preds.append(ctc_greedy_decode(lg[b]))
    refs = [t if len(t) > 0 else " " for t in val_texts]
    prd  = [p if len(p) > 0 else " " for p in preds]
    cer = jiwer.cer(refs, prd)
    del probe; torch.cuda.empty_cache() if device == "cuda" else None
    return max(0.0, 1.0 - cer)
""")

code(r"""
print("Training one linear CTC probe per encoder layer...")
relevance = np.zeros(L)   # R_l for l = 1..L  (stored at index l-1)
for l in range(1, L + 1):
    t0 = time.time()
    V_l = train_probe_for_layer(l)             # uses hidden_states[l]
    relevance[l - 1] = V_l
    print(f"  layer {l:2d}: V_l (1-CER) = {V_l:.3f}   [{time.time()-t0:.0f}s]")

print("\nLayer relevance R_l:", np.round(relevance, 3))
""")

code(r"""
# ----- Figure 1 style: layer relevance profile -----
plt.figure(figsize=(9, 4))
plt.bar(np.arange(1, L + 1), relevance, color="#4C72B0")
plt.xlabel("Encoder layer $l$")
plt.ylabel(r"Relevance $R_l = V_l\ (1-\mathrm{CER}_{\mathrm{probe}})$")
plt.title("ProbeScale layer-relevance profile (TrOCR encoder, IAM OCR)")
plt.xticks(np.arange(1, L + 1))
plt.grid(axis="y", alpha=0.3)
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""## 4. ProbeScale Step 2 — 予算制約付きサブネットワーク選択

論文 Algorithm 1 の `SelectSubnetwork`。サイズ $k$ の **連続ブロック** $S=\{l_{start},...,l_{end}\}$ の中で $\sum_{l\in S} R_l$ を最大化します。比較用に Top-k（最終 $k$ 層）と Uniform-k（等間隔 $k$ 層）も用意します。
""")

code(r"""
def select_probescale(R, k):
    '''Algorithm 1: contiguous block of size k maximizing sum of relevance.'''
    best_start, best_score = 0, -1e9
    for start in range(0, L - k + 1):
        s = R[start:start + k].sum()
        if s > best_score:
            best_score, best_start = s, start
    return list(range(best_start, best_start + k))   # 0-indexed layer ids


def select_topk(k):
    return list(range(L - k, L))                      # final k layers


def select_uniform(k):
    idx = sorted(set(np.linspace(0, L - 1, k).round().astype(int).tolist()))
    j = 0
    while len(idx) < k:                                # fill if rounding collided
        if j not in idx:
            idx = sorted(idx + [j])
        j += 1
    return idx[:k]


for k in K_BUDGETS:
    ps = select_probescale(relevance, k)
    print(f"k={k}: ProbeScale -> layers {[i+1 for i in ps]} "
          f"(score={relevance[ps].sum():.3f}) | "
          f"Top-k -> {[i+1 for i in select_topk(k)]} | "
          f"Uniform-k -> {[i+1 for i in select_uniform(k)]}")
""")

# ---------------------------------------------------------------------------
md(r"""## 5. サブネットワークの抽出と fine-tune

選んだ層インデックスでエンコーダの層 `ModuleList` を作り直します。入力（パッチ）埋め込みと最終 LayerNorm は保持、デコーダはそのまま（次元 768 は不変なのでクロスアテンションは問題なし）。連続ブロックでも先頭層をスキップすると分布がずれるため、論文どおり短く fine-tune して回復させます。
""")

code(r"""
def build_subnetwork(layer_indices):
    '''Return a TrOCR model whose encoder keeps only `layer_indices` (deep-copied).'''
    m = copy.deepcopy(full_model)
    enc_layers = m.encoder.encoder.layer            # DeiT: VisionEncoderDecoder.encoder.encoder.layer
    new_layers = nn.ModuleList([enc_layers[i] for i in layer_indices])
    m.encoder.encoder.layer = new_layers
    m.encoder.config.num_hidden_layers = len(layer_indices)
    m.config.encoder.num_hidden_layers = len(layer_indices)
    return m.to(device)


def finetune(model, dataset, epochs=FT_EPOCHS, lr=FT_LR, batch=TRAIN_BATCH):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    n = len(dataset)
    for ep in range(epochs):
        order = np.random.permutation(n)
        running = 0.0; steps = 0
        for i in range(0, n, batch):
            bidx = order[i:i + batch].tolist()
            sub = dataset.select(bidx)
            images = [im.convert("RGB") for im in sub["image"]]
            pv = processor(images=images, return_tensors="pt").pixel_values.to(device)
            labels = processor.tokenizer(
                sub["text"], padding="max_length", max_length=MAX_LEN,
                truncation=True, return_tensors="pt").input_ids
            labels[labels == processor.tokenizer.pad_token_id] = -100
            labels = labels.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                out = model(pixel_values=pv, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item(); steps += 1
            if steps % 50 == 0:
                print(f"    ep{ep} step{steps}/{math.ceil(n/batch)} loss={running/steps:.3f}")
    return model
""")

code(r"""
results = []   # one row per (k, method)

# reference row: original model (k = L)
results.append({
    "method": "Original", "k": L, "layers": list(range(1, L + 1)),
    "total_M": full_report["total"]/1e6, "encoder_M": full_report["encoder"]/1e6,
    "total_ratio": 1.0, "encoder_ratio": 1.0,
    "cer": baseline_metrics["cer"], "wer": baseline_metrics["wer"],
})

selectors = {
    "ProbeScale": lambda k: select_probescale(relevance, k),
    "Top-k":      lambda k: select_topk(k),
    "Uniform-k":  lambda k: select_uniform(k),
}

for k in K_BUDGETS:
    for method in METHODS:
        layers = selectors[method](k)
        print(f"\n=== {method} | k={k} | layers={[i+1 for i in layers]} ===")
        sub = build_subnetwork(layers)
        rep = model_param_report(sub)
        print("  params (M):", {kk: round(v/1e6, 1) for kk, v in rep.items()})
        print("  fine-tuning...")
        sub = finetune(sub, ft_train)
        m, _, _ = evaluate_model(sub, eval_test, n=N_EVAL)
        print(f"  -> CER={m['cer']*100:.2f}%  WER={m['wer']*100:.2f}%")
        results.append({
            "method": method, "k": k, "layers": [i + 1 for i in layers],
            "total_M": rep["total"]/1e6, "encoder_M": rep["encoder"]/1e6,
            "total_ratio": rep["total"]/full_report["total"],
            "encoder_ratio": rep["encoder"]/full_report["encoder"],
            "cer": m["cer"], "wer": m["wer"],
        })
        del sub
        gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None
""")

# ---------------------------------------------------------------------------
md(r"""## 6. 結果の比較 — 圧縮率と精度

- **圧縮**: エンコーダ単体の比率 `encoder_ratio` とモデル全体の比率 `total_ratio`
- **精度**: CER / WER と、元モデルに対する **精度維持率**（論文の "95–98% retained" に対応; ここでは CER ベースの相対精度 `(1−CER_sub)/(1−CER_orig)`）
""")

code(r"""
df = pd.DataFrame(results)
orig_acc = 1 - baseline_metrics["cer"]
df["char_acc"] = 1 - df["cer"]
df["acc_retained_%"] = (df["char_acc"] / orig_acc) * 100
df["CER_%"] = df["cer"] * 100
df["WER_%"] = df["wer"] * 100
df["encoder_ratio_%"] = df["encoder_ratio"] * 100
df["total_ratio_%"] = df["total_ratio"] * 100

show = df[["method", "k", "layers", "encoder_M", "encoder_ratio_%",
           "total_M", "total_ratio_%", "CER_%", "WER_%", "acc_retained_%"]].copy()
for c in ["encoder_M", "encoder_ratio_%", "total_M", "total_ratio_%",
          "CER_%", "WER_%", "acc_retained_%"]:
    show[c] = show[c].round(2)
show
""")

code(r"""
# ----- side-by-side comparison per budget -----
print("="*78)
print("COMPRESSION & ACCURACY SUMMARY")
print("="*78)
print(f"Original TrOCR: encoder={full_report['encoder']/1e6:.1f}M, "
      f"total={full_report['total']/1e6:.1f}M, "
      f"CER={baseline_metrics['cer']*100:.2f}%, WER={baseline_metrics['wer']*100:.2f}%\n")
for k in K_BUDGETS:
    print(f"--- Budget k={k} layers (encoder {k}/{L} = {k/L*100:.0f}% depth) ---")
    sub = df[(df.k == k)]
    for _, r in sub.iterrows():
        print(f"  {r['method']:11s} | enc {r['encoder_M']:5.1f}M "
              f"({r['encoder_ratio_%']:4.1f}%) | CER {r['CER_%']:5.2f}% | "
              f"WER {r['WER_%']:5.2f}% | acc retained {r['acc_retained_%']:5.1f}%")
    print()
""")

code(r"""
# ----- Figure 2 style: accuracy vs parameter ratio trade-off -----
plt.figure(figsize=(8, 5))
markers = {"ProbeScale": "o", "Top-k": "s", "Uniform-k": "^"}
colors  = {"ProbeScale": "#C44E52", "Top-k": "#4C72B0", "Uniform-k": "#55A868"}
for method in METHODS:
    sub = df[df.method == method].sort_values("encoder_ratio")
    plt.plot(sub["encoder_ratio_%"], (1 - sub["cer"]) * 100,
             marker=markers[method], color=colors[method], label=method, linewidth=2, markersize=9)
plt.scatter([100], [(1 - baseline_metrics["cer"]) * 100], color="black",
            marker="*", s=250, zorder=5, label="Original")
plt.xlabel("Encoder parameter ratio (%)")
plt.ylabel("Character accuracy (1 - CER) %")
plt.title("Performance vs. efficiency trade-off (IAM OCR)")
plt.legend(); plt.grid(alpha=0.3)
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""## 7. 定性的な確認（任意）

ProbeScale で選んだ最小予算のサブネットを再構築し、実際の予測を元モデルと並べて見てみます。
""")

code(r"""
k_show = min(K_BUDGETS)
ps_layers = select_probescale(relevance, k_show)
ps_model = finetune(build_subnetwork(ps_layers), ft_train)

n_show = 4
sub = eval_test.select(range(n_show))
images = [im.convert("RGB") for im in sub["image"]]
pv = processor(images=images, return_tensors="pt").pixel_values.to(device)
with torch.no_grad():
    g_full = full_model.generate(pv, max_new_tokens=MAX_LEN)
    g_ps   = ps_model.generate(pv, max_new_tokens=MAX_LEN)
pred_full = processor.batch_decode(g_full, skip_special_tokens=True)
pred_ps   = processor.batch_decode(g_ps, skip_special_tokens=True)

fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.0 * n_show))
for ax, img, gt, pf, pp in zip(axes, sub["image"], sub["text"], pred_full, pred_ps):
    ax.imshow(img, cmap="gray"); ax.axis("off")
    ax.set_title(f"GT: {gt}\nOriginal: {pf}\nProbeScale(k={k_show}): {pp}",
                 loc="left", fontsize=9)
plt.tight_layout(); plt.show()

del ps_model; gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None
""")

# ---------------------------------------------------------------------------
md(r"""## 8. まとめ

- **ProbeScale を OCR タスクへ忠実に移植**しました: TrOCR のビジョンエンコーダ各層に線形 CTC プローブを当てて層関連度 $R_l$ を算出し（Step 1）、連続ブロック選択 (Algorithm 1) で最も寄与の大きい $k$ 層を残し（Step 2）、短い fine-tune で回復（Step 3）。
- **圧縮**: エンコーダを 12→{6,4} 層に削減し、エンコーダのパラメータを約 50% / 33% に圧縮。デコーダ込みの全体比率も表に出力されます。
- **精度**: 同じ層数のベースライン（Top-k / Uniform-k）に対して、ProbeScale が CER/WER で優位になることを確認するための表・グラフを出力します（論文 Table 1 / Fig. 2 に対応）。

### 結果の読み方
- 「どれくらい圧縮できたか」→ 結果表の `encoder_ratio_%` / `total_ratio_%`
- 「精度はどう変化したか」→ `CER_% / WER_%` と元モデルに対する `acc_retained_%`

### さらに実験を深めるには
- `K_BUDGETS` に他の値を追加して圧縮率-精度カーブを密にする
- `N_FT_TRAIN` / `FT_EPOCHS` を増やして fine-tune を強化（論文は3エポック）
- プローブ重み付け（複数プロパティ $P_T$）を導入: 例えば「文字認識」+「単語境界」など複数プローブを線形結合
- `microsoft/trocr-base-printed` + 印刷文字データセットなど、別ドメインで再現
""")

# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(__file__), "ProbeScale_OCR_TrOCR.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "with", len(cells), "cells")
