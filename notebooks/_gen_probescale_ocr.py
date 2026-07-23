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
md(r"""## 論文の数式とアルゴリズム（実装の対応表）

実装コードには、論文のどの式・どのアルゴリズム行に対応するかを `# [Paper ...]` の形でコメントしてあります。先に対応先となる原典をここにまとめておきます。

### 式 (1): 層関連度 (Layer Relevance)
$$R_{l,T} = \sum_{p_j \in P_T} w(p_j, T)\cdot V_{p_j,l} \qquad (1)$$
- $V_{p_j,l}$: 層 $l$ の表現上で訓練したプローブ $p_j$ の検証性能（本実装では **OCR プローブの文字正解率 $1-\mathrm{CER}$**）
- $P_T$: タスク $T$ に関連するプローブ集合、$w$: その重み
- 本実装は単一プローブ（OCR 文字認識）・$w=1$ なので $R_l = V_l$（論文の SST-2 設定と同じ）

### 式 (2): 予算制約付きサブネットワーク選択
$$S^{*} = \underset{S \in \mathcal{S}_k}{\arg\max} \sum_{l \in S} R_{l,T}
\quad \text{s.t.}\quad |\Theta_S| \le B \qquad (2)$$
- $\mathcal{S}_k$: サイズ $k$ の許容される層部分集合（本実装では**連続ブロック**）
- $|\Theta_S|$: サブネットのパラメータ数、$B$: パラメータ予算

### Algorithm 1: ProbeScale (Contiguous Block Selection)
```
function CALCULATERELEVANCE(M, P_T, w):                # -> Step 1 / 第3節
  for each layer l in {1..L}:
    Extract representations h_l on probe datasets       # (line 6)
    for each probe p in P_T:
      Train probe g_{p,l}(h_l; phi_{p,l})               # (line 8)
      Evaluate performance V_{p,l}                      # (line 9)
      R_{l,T} += w(p,T) * V_{p,l}                        # (line 10) = 式(1)
  return {R_1..R_L}

function SELECTSUBNETWORK({R_l}, B, L):                 # -> Step 2 / 式(2)
  for block size k in {1..k_max}:                       # (line 18)
    for start l_start in {1..L-k+1}:                    # (line 19)
      S = {l_start .. l_start+k-1}                      # (line 21)
      score(S) = sum_{l in S} R_{l,T}                   # (line 22)
      if score(S) > max and |Theta_S| <= B: S* = S      # (line 24)
  return S*

R_T  <- CalculateRelevance(M, P_T, w)                   # (line 32)
S*   <- SelectSubnetwork(R_T, B, L)                     # (line 33)
M_S  <- Extract subnetwork using S*, embeddings, head   # (line 34)
Fine-tune M_S on task T                                 # (line 35)
```

ノートブックのセクション構成: **Step 1** = `CalculateRelevance`（第3節 *Layer-wise Probing Analysis* + 式(1)）、**Step 2** = `SelectSubnetwork`（式(2)）、**Step 3** = サブネット抽出 + fine-tune（第3節 *Subnetwork Extraction and Fine-tuning*）。
""")

# ---------------------------------------------------------------------------
md(r"""## 0. セットアップ

GPU ランタイムを選択してください（メニュー: ランタイム → ランタイムのタイプを変更 → ハードウェアアクセラレータ → GPU）。

> **重要 (Colab)**: 下のセルは依存関係をインストールします。Colab に同梱の Pillow を壊さないよう、**Pillow はアップグレードせず固定**しています。初回インストール後に Pillow などが入れ替わった場合は、**一度ランタイムを再起動**（ランタイム → セッションを再起動）してから、このインストールセルは飛ばして次の import セルから実行してください。
""")

code(r"""
# transformers / datasets などをインストール。
# 注意: Colab 同梱の Pillow を `-U ... pillow` で 12.x に上げると PIL のファイルが
# 混在し `cannot import name '_Ink' from 'PIL._typing'` で import に失敗します。
# そのため Pillow はアップグレード対象から外し、整合した版に固定します。
!pip -q install "transformers>=4.40" "datasets>=2.18" jiwer accelerate
!pip -q install --force-reinstall --no-deps "Pillow==11.3.0"
print("\n=== インストール完了 ===")
print("PIL 周りが入れ替わった初回は、ランタイムを再起動してから import セル以降を実行してください。")
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
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_cosine_schedule_with_warmup
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
TRAIN_BATCH   = 4      # base batch size (probe trainer uses TRAIN_BATCH*4 over cached features)

# --- subnetwork fine-tuning + evaluation (Option B: CTC head) ---
# The compressed model is [embeddings -> selected encoder layers -> linear CTC head],
# faithful to the paper's "encoder layers + small task head". No generative decoder,
# so recovery is mostly head-training: stable and light. CTC-specific knobs
# (CTC_BATCH / CTC_EPOCHS / ENC_LR / HEAD_LR) are defined in the training cell below.
N_FT_TRAIN = 1500      # images to fine-tune each subnetwork (encoder block + CTC head)
N_EVAL     = 300       # test images for final CER / WER
FT_EPOCHS  = 3         # base epoch count (CTC head gets max(FT_EPOCHS, 4))
FT_LR      = 5e-5      # encoder LR while adapting the kept layers
EVAL_BATCH  = 16
MAX_LEN     = 64       # max target token length (used by the original-TrOCR baseline only)

# --- budgets to evaluate (number of encoder layers to keep) ---
# 6 = 50%, 4 = 33% of the 12-layer encoder. These are aggressive; recovery is hard.
# Tip: add 8 (=67%) for a gentler point that recovers closer to the original.
K_BUDGETS = [6, 4]
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
        refs  += [t if isinstance(t, str) else "" for t in sub["text"]]   # guard None / non-str
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

**対応**: 論文 Algorithm 1 `CalculateRelevance`（line 3–14）／第3節 *Layer-wise Probing Analysis* (a)(b)(c)／**式 (1)**。

各ステップと論文の対応:

| 本実装 | 論文 |
|---|---|
| 1. 各層 $l$ の隠れ表現 $h_l$ を抽出（1回の forward で全層取得）| Alg.1 line 6 / 第3節(a) "Extract representations $h_l$" |
| 2. その層に **線形 CTC プローブ** $g_l$ を学習（エンコーダ凍結）| Alg.1 line 8 / 第3節(b) "Train probe $g_{p,l}(h_l;\phi_{p,l})$" |
| 3. 検証セットで **文字正解率 $V_l = 1-\mathrm{CER}$** を測定 | Alg.1 line 9 / 第3節(c) "Evaluate $V_{p,l}$" |
| 4. 層関連度 $R_l = \sum_p w\,V_{p,l}$（単一プローブ・$w=1$）| Alg.1 line 10 / **式 (1)** |

特徴は一度だけ計算してキャッシュします（全層を再計算しないため高速）。
""")

code(r"""
# ----- character vocabulary for the CTC probe (built from probe-train texts) -----
charset = set()
for t in probe_train["text"]:
    if isinstance(t, str):                # guard None / non-str
        charset.update(t)
itos = ["<blank>"] + sorted(charset)     # index 0 = CTC blank
stoi = {c: i for i, c in enumerate(itos)}
V = len(itos)
print("probe vocab size:", V)


def encode_text(t):
    return [stoi[c] for c in t if c in stoi]
""")

code(r"""
# [Paper Alg.1 line 6 / Sec.3 "Layer-wise Probing Analysis (a)"]
#   "Extract representations h_l from M for a dataset D_{p_j}".
# h_l = f_l(h_{l-1}; theta_l) for l=1..L (Sec.3). One forward with
# output_hidden_states=True returns h_0..h_L, so we cache every layer at once.
@torch.no_grad()
def extract_layer_features(dataset, n):
    '''Forward the (frozen) encoder once per image and cache every layer's token
    sequence. Returns feats[layer] = fp16 CPU tensor of shape (n, T, HIDDEN).'''
    full_model.eval()
    per_layer = [[] for _ in range(L + 1)]   # hidden_states has L+1 entries: h_0 (emb) + h_1..h_L
    targets, target_lens, texts = [], [], []
    for i in range(0, n, EVAL_BATCH):
        sub = dataset.select(range(i, min(i + EVAL_BATCH, n)))
        images = [im.convert("RGB") for im in sub["image"]]
        pv = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            enc_out = full_model.encoder(pv, output_hidden_states=True)   # frozen M
        for li, hs in enumerate(enc_out.hidden_states):                  # h_li for li=0..L
            per_layer[li].append(hs.to("cpu", torch.float16))
        for t in sub["text"]:
            t = t if isinstance(t, str) else ""        # guard None / non-str
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
    '''Train a linear CTC probe on cached features of one layer; return V_l = 1 - CER.

    [Paper Alg.1 line 8 / Sec.3(b)]  Train probe g_{p,l}(h_l; phi_{p,l}).
        Here g_{p,l} is a single Linear(HIDDEN -> vocab) read out over the patch
        sequence and decoded with CTC -> the "simple linear probe" of the paper.
        The encoder M is frozen (we only fit phi_{p,l}), exactly as in probing.
    [Paper Alg.1 line 9 / Sec.3(c)]  Evaluate V_{p,l} on a held-out set.
    '''
    feats = train_feats[layer_index]           # cached h_l, shape (N, T, HIDDEN), fp16 CPU
    probe = nn.Linear(HIDDEN, V).to(device)    # g_{p,l}: the probe classifier phi_{p,l}
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)   # L_probe: probe training objective (Sec.3b)
    N = feats.shape[0]
    order = np.arange(N)
    for ep in range(PROBE_EPOCHS):
        np.random.shuffle(order)
        for i in range(0, N, TRAIN_BATCH * 4):
            bidx = order[i:i + TRAIN_BATCH * 4]
            x = feats[bidx].to(device, torch.float32)             # h_l for this batch (b, T, H)
            logp = probe(x).log_softmax(-1).transpose(0, 1)       # g_{p,l}(h_l) -> (T, b, V)
            tgt = torch.cat([train_targets[j] for j in bidx]).to(device)   # y_{p_j}: target chars
            tgt_len = torch.tensor([len(train_targets[j]) for j in bidx], device=device)
            in_len = torch.full((len(bidx),), T_tokens, dtype=torch.long, device=device)
            loss = ctc(logp, tgt, in_len, tgt_len)                # minimize L_probe(g_{p,l}(h_l), y)
            opt.zero_grad(); loss.backward(); opt.step()
    # ---- [Alg.1 line 9 / Sec.3(c)] measure V_{p,l} = 1 - CER on validation features ----
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
# [Paper Alg.1 lines 4-13 = CalculateRelevance] loop over every layer and
# accumulate the task relevance.  With a single probe (OCR) and w=1 this is
#     R_l = sum_{p in P_T} w(p,T) * V_{p,l}   == V_l        # <- Eq. (1)
print("Training one linear CTC probe per encoder layer...")
relevance = np.zeros(L)   # R_{l,T} for l = 1..L  (stored at index l-1)
W_OCR = 1.0               # w(p, T): single-probe uniform weight (Sec.4 "we use w = 1")
for l in range(1, L + 1):                       # Alg.1 line 5: for each layer l
    t0 = time.time()
    V_l = train_probe_for_layer(l)              # Alg.1 lines 8-9: V_{p,l} from h_l = hidden_states[l]
    relevance[l - 1] = W_OCR * V_l              # Alg.1 line 10 / Eq.(1): R_l += w * V_{p,l}
    print(f"  layer {l:2d}: V_l (1-CER) = {V_l:.3f}   [{time.time()-t0:.0f}s]")

print("\nLayer relevance R_l (Eq.1):", np.round(relevance, 3))
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

**対応**: 論文 Algorithm 1 `SelectSubnetwork`（line 15–31）／**式 (2)**。

サイズ $k$ の **連続ブロック** $S=\{l_{start},...,l_{end}\}$ の中で $\sum_{l\in S} R_l$ を最大化します（式(2)）。パラメータ予算 $B$ は「残す層数 $k$」で表現します（各 Transformer 層のパラメータ数はほぼ一定なので $|\Theta_S|\le B \Leftrightarrow k\le k_{max}$）。比較用に Top-k（最終 $k$ 層）と Uniform-k（等間隔 $k$ 層 — 論文 *Experimental Setup* のベースライン）も用意します。
""")

code(r"""
# [Paper Alg.1 lines 15-31 = SelectSubnetwork] + Eq.(2):
#   S* = argmax_{S in S_k} sum_{l in S} R_{l,T}   s.t. |Theta_S| <= B
# S_k = all CONTIGUOUS blocks of size k (Sec.3 "Budget-Constrained Subnetwork Selection").
def select_probescale(R, k):
    '''Eq.(2): contiguous block of size k maximizing sum of layer relevance R_l.'''
    best_start, best_score = 0, -1e9
    for start in range(0, L - k + 1):              # Alg.1 line 19: each start l_start
        s = R[start:start + k].sum()               # Alg.1 line 22: score(S) = sum_{l in S} R_l
        if s > best_score:                         # Alg.1 line 24: keep arg-max under budget
            best_score, best_start = s, start
    return list(range(best_start, best_start + k))   # S* (0-indexed layer ids)


# --- baselines from the paper's "Subnetwork Selection & Baselines" ---
def select_topk(k):
    return list(range(L - k, L))                      # "Top-k Layers": the final k layers


def select_uniform(k):                                # "Uniform-k Layers": k evenly spaced layers
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
md(r"""## 5. サブネットワークの構築・学習・評価（案B: 論文忠実な CTC ヘッド）

**対応**: 論文 Algorithm 1 line 34–35 ／ 第3節 *Subnetwork Extraction and Fine-tuning*。

論文は「エンコーダ層 + 小さなタスクヘッド $\theta'_{head}$」を抽出し、ヘッドを再学習します（必要なら数エポックの fine-tune）。OCR の忠実な対応として、圧縮モデルを

$$\text{パッチ埋め込み }\theta_{emb}\;\rightarrow\;\text{選択エンコーダ層 }S^{*}\;\rightarrow\;\text{線形 CTC ヘッド}$$

とします（**巨大な生成デコーダは使いません**）。これにより、

- 論文の機構（層＋軽量ヘッド）と一致する
- 回復は主にヘッド学習なので**安定・軽量**（生成デコーダが壊れて暴走する問題が消える）
- ProbeScale の層選択効果が CER に素直に反映される

比較の基準 *Original* は **12 層フルエンコーダ + 同じ CTC ヘッド**（論文の「元の SLM」に相当）。圧縮率はエンコーダのパラメータ比で測ります。

> 注: 元の TrOCR（seq2seq, 333.9M, CER≈5%）は強力ですが、それは別アーキテクチャ（生成デコーダ）の値です。ここでは「**同じ CTC ヘッドの下でエンコーダ層数・選択を変えたときの挙動**」を見るため、フルエンコーダ CTC を基準に比較します。生成デコーダ版は論文の将来課題（generative SLM）で、Colab 予算では回復しきれないことを前段で確認済みです。
""")

code(r"""
# ---- ProbeScale subnetwork as a CTC recognizer: encoder block + linear CTC head ----
# This mirrors the paper's "encoder layers + small task head" exactly (head = the probe).
def _locate_encoder_layers(vision_model):
    '''Return (parent_module, attr_name, ModuleList) holding the transformer layers.'''
    cands = [(n, mod) for n, mod in vision_model.named_modules()
             if isinstance(mod, nn.ModuleList) and len(mod) == L]
    if not cands:
        lists = [(n, mod) for n, mod in vision_model.named_modules()
                 if isinstance(mod, nn.ModuleList)]
        if not lists:
            raise AttributeError("No ModuleList found inside the vision encoder")
        cands = [max(lists, key=lambda nm: len(nm[1]))]
    name, module_list = cands[0]
    parent = vision_model
    for p in name.split(".")[:-1]:
        parent = getattr(parent, p)
    return parent, name.split(".")[-1], module_list


class CTCRecognizer(nn.Module):
    '''theta_emb -> selected encoder layers -> linear CTC head (= the paper's task head).'''
    def __init__(self, encoder, hidden, vocab):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden, vocab)
    def forward(self, pixel_values):
        h = self.encoder(pixel_values).last_hidden_state   # (B, T, H)
        return self.head(h)                                # (B, T, V)


# [Paper Alg.1 line 34] Construct M_S = embeddings + layers in S* + a fresh CTC head.
def build_ctc_subnetwork(layer_indices):
    enc = copy.deepcopy(full_model.encoder)                          # theta_emb kept
    parent, attr, enc_layers = _locate_encoder_layers(enc)
    setattr(parent, attr, nn.ModuleList([enc_layers[i] for i in layer_indices]))  # keep theta_l, l in S*
    cfg = getattr(enc, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        cfg.num_hidden_layers = len(layer_indices)
    return CTCRecognizer(enc, HIDDEN, V).to(device)


_p, _a, _ml = _locate_encoder_layers(full_model.encoder)
print(f"encoder layer stack: {type(_p).__name__}.{_a}  (len={len(_ml)})")
""")

code(r"""
# CTC-track hyper-parameters (no generative decoder -> can use a bigger batch)
CTC_BATCH  = 8
CTC_EPOCHS = max(FT_EPOCHS, 4)   # the head is trained from scratch, give it a few epochs
ENC_LR     = FT_LR               # gently adapt the kept encoder layers
HEAD_LR    = 1e-3                # the random CTC head needs a larger LR to converge


# [Paper Alg.1 line 35] Fine-tune the subnetwork (encoder block + CTC head) for a few epochs.
def train_ctc(model, dataset, epochs=CTC_EPOCHS, batch=CTC_BATCH):
    model.train()
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": ENC_LR},
        {"params": model.head.parameters(),    "lr": HEAD_LR},
    ], weight_decay=0.01)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    n = len(dataset)
    steps_per_epoch = math.ceil(n / batch)
    total = max(1, steps_per_epoch * epochs)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    for ep in range(epochs):
        order = np.random.permutation(n)
        running = 0.0; seen = 0
        for i in range(0, n, batch):
            bidx = order[i:i + batch].tolist()
            sub = dataset.select(bidx)
            images = [im.convert("RGB") for im in sub["image"]]
            texts  = [t if isinstance(t, str) else "" for t in sub["text"]]
            pv = processor(images=images, return_tensors="pt").pixel_values.to(device)
            tgts, tlens = [], []
            for t in texts:
                ids = encode_text(t)[:T_tokens - 1]      # CTC needs target_len <= input_len
                tgts.append(torch.tensor(ids, dtype=torch.long)); tlens.append(len(ids))
            targets = torch.cat(tgts).to(device) if tgts else torch.zeros(0, dtype=torch.long, device=device)
            tlens = torch.tensor(tlens, device=device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(pv)                       # (B, T, V)
            logp = logits.float().log_softmax(-1).transpose(0, 1)   # (T, B, V), fp32 for CTC stability
            in_len = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=device)
            loss = ctc(logp, targets, in_len, tlens)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item(); seen += 1
            if seen % 100 == 0:
                print(f"    ep{ep} {seen}/{steps_per_epoch} loss={running/seen:.3f}")
    return model


@torch.no_grad()
def eval_ctc(model, dataset, n=N_EVAL, batch=EVAL_BATCH):
    model.eval()
    preds, refs = [], []
    for i in range(0, n, batch):
        sub = dataset.select(range(i, min(i + batch, n)))
        images = [im.convert("RGB") for im in sub["image"]]
        pv = processor(images=images, return_tensors="pt").pixel_values.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(pv)
        for b in range(logits.size(0)):
            preds.append(ctc_greedy_decode(logits[b].float()))
        refs += [t if isinstance(t, str) else "" for t in sub["text"]]
    refs_safe  = [r if len(r) > 0 else " " for r in refs]
    preds_safe = [p if len(p) > 0 else " " for p in preds]
    return {"cer": jiwer.cer(refs_safe, preds_safe), "wer": jiwer.wer(refs_safe, preds_safe)}
""")

code(r"""
results = []
selectors = {
    "ProbeScale": lambda k: select_probescale(relevance, k),   # Eq.(2)
    "Top-k":      lambda k: select_topk(k),
    "Uniform-k":  lambda k: select_uniform(k),
}

# ---- Reference: full 12-layer encoder + CTC head (the paper's "Original SLM") ----
print("=== Reference: full encoder (12L) + CTC head ===")
ref_model = train_ctc(build_ctc_subnetwork(list(range(L))), ft_train)
enc_full = count_params(ref_model.encoder)
ref = eval_ctc(ref_model, eval_test, n=N_EVAL)
print(f"  -> CER={ref['cer']*100:.2f}%  WER={ref['wer']*100:.2f}%  (encoder {enc_full/1e6:.1f}M)")
results.append({"method": "Original(12L)", "k": L, "layers": list(range(1, L + 1)),
                "encoder_M": enc_full/1e6, "encoder_ratio": 1.0,
                "cer": ref["cer"], "wer": ref["wer"]})
del ref_model; gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None

# ---- ProbeScale vs baselines at each budget (Algorithm 1 lines 32-35) ----
for k in K_BUDGETS:
    for method in METHODS:
        layers = selectors[method](k)                         # Alg.1 line 33: S*
        print(f"\n=== {method} | k={k} | layers={[i+1 for i in layers]} ===")
        m = build_ctc_subnetwork(layers)                      # Alg.1 line 34: M_S
        enc_p = count_params(m.encoder)
        print(f"  encoder params: {enc_p/1e6:.1f}M ({enc_p/enc_full*100:.1f}% of full encoder)")
        m = train_ctc(m, ft_train)                            # Alg.1 line 35: fine-tune
        met = eval_ctc(m, eval_test, n=N_EVAL)
        print(f"  -> CER={met['cer']*100:.2f}%  WER={met['wer']*100:.2f}%")
        results.append({"method": method, "k": k, "layers": [i + 1 for i in layers],
                        "encoder_M": enc_p/1e6, "encoder_ratio": enc_p/enc_full,
                        "cer": met["cer"], "wer": met["wer"]})
        del m; gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None
""")

# ---------------------------------------------------------------------------
md(r"""## 6. 結果の比較 — 圧縮率と精度

- **圧縮**: エンコーダのパラメータ比 `encoder_ratio_%`（フル12層エンコーダ基準）
- **精度**: CER / WER と、基準（フルエンコーダ CTC）に対する **精度維持率** `acc_retained_% = (1-CER_sub)/(1-CER_orig)`（論文の "95-98% retained" に対応）
""")

code(r"""
# Reproduces the paper's Table 1 (Params, ratio, Acc.) for the CTC track.
df = pd.DataFrame(results)
orig_acc = 1 - df.loc[0, "cer"]                         # Original(12L) char accuracy
df["char_acc"] = 1 - df["cer"]
df["acc_retained_%"] = (df["char_acc"] / orig_acc) * 100
df["CER_%"] = df["cer"] * 100
df["WER_%"] = df["wer"] * 100
df["encoder_ratio_%"] = df["encoder_ratio"] * 100

show = df[["method", "k", "layers", "encoder_M", "encoder_ratio_%",
           "CER_%", "WER_%", "acc_retained_%"]].copy()
for c in ["encoder_M", "encoder_ratio_%", "CER_%", "WER_%", "acc_retained_%"]:
    show[c] = show[c].round(2)
show
""")

code(r"""
print("="*78)
print("COMPRESSION & ACCURACY SUMMARY (CTC track)")
print("="*78)
orig = df.iloc[0]
print(f"Original(12L) + CTC head: encoder={orig['encoder_M']:.1f}M, "
      f"CER={orig['CER_%']:.2f}%, WER={orig['WER_%']:.2f}%\n")
for k in K_BUDGETS:
    print(f"--- Budget k={k} layers (encoder {k}/{L} = {k/L*100:.0f}% depth) ---")
    for _, r in df[df.k == k].iterrows():
        print(f"  {r['method']:13s} | enc {r['encoder_M']:5.1f}M "
              f"({r['encoder_ratio_%']:4.1f}%) | CER {r['CER_%']:6.2f}% | "
              f"WER {r['WER_%']:6.2f}% | acc retained {r['acc_retained_%']:5.1f}%")
    print()
""")

code(r"""
# Figure 2 style: accuracy vs parameter ratio trade-off
plt.figure(figsize=(8, 5))
markers = {"ProbeScale": "o", "Top-k": "s", "Uniform-k": "^"}
colors  = {"ProbeScale": "#C44E52", "Top-k": "#4C72B0", "Uniform-k": "#55A868"}
for method in METHODS:
    sub = df[df.method == method].sort_values("encoder_ratio")
    plt.plot(sub["encoder_ratio_%"], (1 - sub["cer"]) * 100,
             marker=markers[method], color=colors[method], label=method, linewidth=2, markersize=9)
plt.scatter([100], [(1 - df.loc[0, "cer"]) * 100], color="black",
            marker="*", s=250, zorder=5, label="Original(12L)")
plt.xlabel("Encoder parameter ratio (%)")
plt.ylabel("Character accuracy (1 - CER) %")
plt.title("Performance vs. efficiency trade-off (IAM OCR, CTC head)")
plt.legend(); plt.grid(alpha=0.3)
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""## 7. 定性的な確認（任意）

フルエンコーダ CTC と、ProbeScale で選んだ最小予算のサブネットの予測を並べて見ます。
""")

code(r"""
k_show = min(K_BUDGETS)
full_ctc = train_ctc(build_ctc_subnetwork(list(range(L))), ft_train)
ps_ctc   = train_ctc(build_ctc_subnetwork(select_probescale(relevance, k_show)), ft_train)

n_show = 4
sub = eval_test.select(range(n_show))
images = [im.convert("RGB") for im in sub["image"]]
pv = processor(images=images, return_tensors="pt").pixel_values.to(device)
with torch.no_grad():
    lf = full_ctc(pv); lp = ps_ctc(pv)
pred_full = [ctc_greedy_decode(lf[b].float()) for b in range(n_show)]
pred_ps   = [ctc_greedy_decode(lp[b].float()) for b in range(n_show)]

fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.0 * n_show))
for ax, img, gt, pf, pp in zip(axes, sub["image"], sub["text"], pred_full, pred_ps):
    ax.imshow(img, cmap="gray"); ax.axis("off")
    ax.set_title(f"GT: {gt}\nFull-12L CTC: {pf}\nProbeScale(k={k_show}) CTC: {pp}",
                 loc="left", fontsize=9)
plt.tight_layout(); plt.show()

del full_ctc, ps_ctc; gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None
""")

# ---------------------------------------------------------------------------
md(r"""## 8. まとめ

- **ProbeScale を OCR に論文忠実な形で移植**しました: 各エンコーダ層に線形 CTC プローブを当てて層関連度 $R_l$ を算出（Step 1, 式(1)）→ 連続ブロック選択（Step 2, 式(2)/Algorithm 1）→ **選択層 + 線形 CTC ヘッド**としてサブネットを構築・短く学習（Step 3, line 34–35）。
- これは論文の機構「**エンコーダ層 + 軽量タスクヘッド（=プローブ）**」と一致し、生成デコーダの回復問題を回避します（生成版＝論文の将来課題で、Colab では回復困難なことを確認済み）。

### ご依頼の2点
- **どれくらい圧縮できたか** → 結果表 `encoder_ratio_%`（12→{6,4} 層で約 50% / 33%）。さらに生成デコーダ(247M)を使わない CTC 認識器なので、元 TrOCR(333.9M) 比ではモデル全体も大幅に小型化。
- **精度はどう変化したか** → `CER_% / WER_%` と、フルエンコーダ CTC 基準の `acc_retained_%`。同一予算で **ProbeScale vs Top-k / Uniform-k** を比較（論文 Table 1 / Fig. 2 に対応）。

### さらに深めるには
- `K_BUDGETS` に 8（=67%）など追加して圧縮率-精度カーブを密にする
- `CTC_EPOCHS` / `N_FT_TRAIN` を増やしてヘッド・エンコーダの学習を強化
- 複数プローブ（文字認識＋単語境界など）の重み付き結合で $R_l$ を多面的にする
- `microsoft/trocr-base-printed` + 印刷文字データセットなど別ドメインで再現
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
