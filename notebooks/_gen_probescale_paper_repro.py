"""Generator for the ProbeScale PAPER-REPRODUCTION notebook.

Reproduces the paper's actual experiment (RoBERTa-Large on GLUE SST-2,
classification) to validate that the ProbeScale algorithm implementation is
correct. Shares the same selection logic (contiguous-block, Eq.2) as the OCR
notebook, so a faithful reproduction here validates the OCR extension too.

Run: python notebooks/_gen_probescale_paper_repro.py
Produces: notebooks/ProbeScale_Paper_Repro.ipynb
"""
import json
import os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------------------
md(r"""# ProbeScale 追試（論文どおりの分類タスクで実装を検証）

> 論文: **ProbeScale: Probing Analysis to Optimize Neural Scaling Laws for Efficient Small Language Model Inference** (Sourav Das, arXiv:2606.01806)

このノートブックは **論文が実際に検証した設定**——**RoBERTa-Large × GLUE SST-2（感情分類）**——で ProbeScale を再現し、**アルゴリズム実装が正しいか**を確かめるためのものです。OCR ノートブック（`ProbeScale_OCR_TrOCR.ipynb`）と**同じ選択ロジック**（連続ブロック選択, 式(2)）を使うので、ここで論文の傾向（**ProbeScale > Top-k / Uniform-k**、維持率 **95–98%**）が再現できれば、実装の妥当性が裏付けられます。

## 論文の設定（第4節）に忠実に

| 要素 | 本ノートブック |
|---|---|
| ベースモデル | **RoBERTa-Large**（24層, hidden 1024, 355M）|
| 対象タスク | **SST-2**（GLUE, 文の感情2値分類, 指標=accuracy）|
| プローブ | 各層の **[CLS] 表現**に**線形プローブ**を学習し **SST-2 ラベルを直接予測**（$P_T=\{$SST-2$\}$, $w=1$）|
| 関連度 | $R_{l,T}=\sum_p w(p,T)V_{p,l}$ — 単一プローブ・$w=1$（式(1)）|
| 選択 | 連続ブロック $S^*=\arg\max\sum_{l\in S}R_l$（式(2), Algorithm 1）|
| 抽出＋学習 | 選択層 + **新しい分類ヘッド** $\theta'_{head}$、**3エポック** fine-tune（第4節）|
| ベースライン | Top-k（最終k層）, Uniform-k（等間隔k層）, Original（フルfine-tune）|

> 分類タスクなので、論文どおり「**プローブ＝最終ヘッド**」が成立し、軽量ヘッドの再学習で高い維持率が出ます。これが OCR（生成）との本質的な違いでした。
""")

# ---------------------------------------------------------------------------
md(r"""## 論文の式とアルゴリズム

- **式(1)** 層関連度: $R_{l,T}=\sum_{p\in P_T} w(p,T)\,V_{p,l}$（本実装は単一プローブ・$w=1$ なので $R_l=V_l$）
- **式(2)** 予算制約付き選択: $S^*=\arg\max_{S\in\mathcal{S}_k}\sum_{l\in S}R_{l,T}\ \text{s.t.}\ |\Theta_S|\le B$（$\mathcal{S}_k$=連続ブロック）
- **Algorithm 1**: `CalculateRelevance`（層ごとにプローブ学習→$V_{p,l}$→$R_l$）→ `SelectSubnetwork`（連続ブロック最大化）→ 抽出→ fine-tune

コード中に `# [Paper Eq.(1)]` / `# [Paper Alg.1 line N]` のコメントで対応を示します。
""")

# ---------------------------------------------------------------------------
md(r"""## 0. セットアップ（GPU ランタイム推奨）""")

code(r"""
!pip -q install "transformers>=4.40" "datasets>=2.18" scikit-learn
""")

code(r"""
import math, random, copy, time, gc
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          get_cosine_schedule_with_warmup)
from sklearn.linear_model import LogisticRegression

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
md(r"""### 実験設定""")

code(r"""
# ===== Experiment configuration =====
MODEL_NAME = "roberta-large"   # paper's base SLM (24 layers, hidden 1024). Use "roberta-base" for speed.
TASK       = "sst2"            # GLUE task (single-sentence classification)
NUM_LABELS = 2

# --- probing (Step 1) ---
N_PROBE_TRAIN = 4000   # train examples for the per-layer linear probes (cheap: sklearn LogisticRegression)
N_PROBE_VAL   = 872    # SST-2 has 872 validation examples (use all)

# --- subnetwork fine-tuning (Step 3) ---
N_FT_TRAIN = 8000      # train subset for fine-tuning each subnetwork (raise toward 67k for exact paper repro)
FT_EPOCHS  = 3         # paper fine-tunes for 3 epochs
FT_LR      = 1e-5      # typical RoBERTa-Large fine-tuning LR
BATCH      = 16
EVAL_BATCH = 64
MAX_LEN    = 128

# --- budgets (number of layers to keep), matching the paper's Table 1 (k=6, k=4) ---
K_BUDGETS = [6, 4]
METHODS   = ["ProbeScale", "Top-k", "Uniform-k"]
""")

# ---------------------------------------------------------------------------
md(r"""## 1. データセット（GLUE SST-2）の読み込み""")

code(r"""
raw = load_dataset("glue", TASK)
print(raw)
TEXT_KEYS = {"sst2": ("sentence", None), "qnli": ("question", "sentence")}[TASK]

train_split = raw["train"].shuffle(seed=SEED)
val_split   = raw["validation"]
probe_train = train_split.select(range(N_PROBE_TRAIN))
ft_train    = train_split.select(range(N_FT_TRAIN))
print("example:", {k: train_split[0][k] for k in (*[t for t in TEXT_KEYS if t], "label")})
""")

# ---------------------------------------------------------------------------
md(r"""## 2. モデル・トークナイザの読み込み""")

code(r"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()   # for feature extraction (probing)
L = encoder.config.num_hidden_layers
HIDDEN = encoder.config.hidden_size
print(f"Encoder layers L={L}, hidden={HIDDEN}")


def tokenize(batch_texts_a, batch_texts_b=None):
    return tokenizer(batch_texts_a, batch_texts_b, padding=True, truncation=True,
                     max_length=MAX_LEN, return_tensors="pt")


def count_params(m):
    return sum(p.numel() for p in m.parameters())
""")

# ---------------------------------------------------------------------------
md(r"""## 3. Step 1 — 層ごとのプロービング解析（式(1) / Algorithm 1 CalculateRelevance）

各層 $l$ の **[CLS] 表現** $h_l$ を抽出し、**線形プローブ**で SST-2 ラベルを直接予測。検証精度 $V_l$ が層関連度 $R_l$（単一プローブ・$w=1$）。論文 Table 2 の RoBERTa-Large の傾向（上位〜上位中層でピーク）を再現できれば実装は妥当です。
""")

code(r"""
# [Paper Alg.1 line 6 / Sec.3(a)] Extract [CLS] representation h_l for every layer (one forward).
@torch.no_grad()
def extract_cls_features(dataset, n):
    a_key, b_key = TEXT_KEYS
    per_layer = [[] for _ in range(L + 1)]   # hidden_states: h_0(emb) + h_1..h_L
    labels = []
    for i in range(0, n, EVAL_BATCH):
        sub = dataset.select(range(i, min(i + EVAL_BATCH, n)))
        enc = tokenize(sub[a_key], sub[b_key] if b_key else None).to(device)
        out = encoder(**enc, output_hidden_states=True)
        for li, hs in enumerate(out.hidden_states):
            per_layer[li].append(hs[:, 0, :].float().cpu().numpy())   # [CLS] = position 0
        labels.extend(sub["label"])
    feats = [np.concatenate(c, axis=0) for c in per_layer]
    return feats, np.array(labels)


print("Extracting [CLS] features for probing...")
t0 = time.time()
train_feats, train_y = extract_cls_features(probe_train, N_PROBE_TRAIN)
val_feats,   val_y   = extract_cls_features(val_split, min(N_PROBE_VAL, len(val_split)))
print(f"  done in {time.time()-t0:.0f}s | layers cached: {len(train_feats)}")
""")

code(r"""
# [Paper Alg.1 lines 8-10 / Eq.(1)] Train a linear probe per layer, V_l = val accuracy, R_l = w*V_l.
relevance = np.zeros(L)   # R_l for l=1..L (index l-1)
W = 1.0                   # w(p,T): single-probe uniform weight (Sec.4: "we use w = 1")
for l in range(1, L + 1):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(train_feats[l], train_y)               # probe g_{p,l} on h_l (Alg.1 line 8)
    V_l = clf.score(val_feats[l], val_y)           # V_{p,l} (Alg.1 line 9)
    relevance[l - 1] = W * V_l                      # R_l (Alg.1 line 10 / Eq.1)
    print(f"  layer {l:2d}: probe acc V_l = {V_l:.3f}")
print("\nLayer relevance R_l:", np.round(relevance, 3))
""")

code(r"""
plt.figure(figsize=(10, 4))
plt.bar(np.arange(1, L + 1), relevance, color="#4C72B0")
plt.xlabel("Layer $l$"); plt.ylabel(r"$R_l = V_l$ (probe accuracy)")
plt.title(f"ProbeScale layer relevance ({MODEL_NAME}, {TASK.upper()}) — cf. paper Table 2 / Fig. 1")
plt.xticks(np.arange(1, L + 1)); plt.grid(axis="y", alpha=0.3); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""## 4. Step 2 — 連続ブロック選択（式(2) / Algorithm 1 SelectSubnetwork）

OCR ノートブックと**同一の選択関数**です（実装が共通であることが検証のポイント）。
""")

code(r"""
def select_probescale(R, k):   # [Paper Eq.(2) / Alg.1 lines 15-31] contiguous block maximizing sum R_l
    best_start, best = 0, -1e9
    for start in range(0, L - k + 1):
        s = R[start:start + k].sum()
        if s > best:
            best, best_start = s, start
    return list(range(best_start, best_start + k))


def select_topk(k):            # baseline: final k layers
    return list(range(L - k, L))


def select_uniform(k):         # baseline: k evenly spaced layers
    idx = sorted(set(np.linspace(0, L - 1, k).round().astype(int).tolist()))
    j = 0
    while len(idx) < k:
        if j not in idx:
            idx = sorted(idx + [j])
        j += 1
    return idx[:k]


for k in K_BUDGETS:
    print(f"k={k}: ProbeScale {[i+1 for i in select_probescale(relevance,k)]} "
          f"(score={relevance[select_probescale(relevance,k)].sum():.3f}) | "
          f"Top-k {[i+1 for i in select_topk(k)]} | Uniform {[i+1 for i in select_uniform(k)]}")
""")

# ---------------------------------------------------------------------------
md(r"""## 5. Step 3 — サブネット抽出 + fine-tune（Algorithm 1 line 34–35）

選択層でエンコーダの層 `ModuleList` を作り直し、**新しい分類ヘッド** $\theta'_{head}$ を付けて 3 エポック fine-tune します（論文どおり）。
""")

code(r"""
def get_base(m):                       # base encoder (e.g. m.roberta) across model types
    return getattr(m, m.base_model_prefix)


# [Paper Alg.1 line 34] Construct M_S: embeddings + selected layers + fresh classification head.
def build_subnetwork(layer_indices):
    m = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    base = get_base(m)
    base.encoder.layer = nn.ModuleList([base.encoder.layer[i] for i in layer_indices])  # keep theta_l
    m.config.num_hidden_layers = len(layer_indices)
    return m.to(device)


# [Paper Alg.1 line 35] Fine-tune M_S on the task for FT_EPOCHS epochs.
def finetune(m, dataset, epochs=FT_EPOCHS, lr=FT_LR, batch=BATCH):
    a_key, b_key = TEXT_KEYS
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    n = len(dataset)
    total = math.ceil(n / batch) * epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    for ep in range(epochs):
        order = np.random.permutation(n)
        run = 0.0; seen = 0
        for i in range(0, n, batch):
            bidx = order[i:i + batch].tolist()
            sub = dataset.select(bidx)
            enc = tokenize(sub[a_key], sub[b_key] if b_key else None).to(device)
            labels = torch.tensor(sub["label"], device=device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss = m(**enc, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item(); seen += 1
            if seen % 100 == 0:
                print(f"    ep{ep} {seen}/{math.ceil(n/batch)} loss={run/seen:.3f}")
    return m


@torch.no_grad()
def evaluate(m, dataset):
    a_key, b_key = TEXT_KEYS
    m.eval()
    correct = total = 0
    for i in range(0, len(dataset), EVAL_BATCH):
        sub = dataset.select(range(i, min(i + EVAL_BATCH, len(dataset))))
        enc = tokenize(sub[a_key], sub[b_key] if b_key else None).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = m(**enc).logits
        pred = logits.argmax(-1).cpu().numpy()
        y = np.array(sub["label"])
        correct += (pred == y).sum(); total += len(y)
    return correct / total
""")

code(r"""
# free the probing encoder to save memory before fine-tuning big models
del encoder; gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None

results = []
selectors = {"ProbeScale": lambda k: select_probescale(relevance, k),
             "Top-k": lambda k: select_topk(k), "Uniform-k": lambda k: select_uniform(k)}

# ---- Original: full model (24 layers) fine-tuned, like the paper's first row ----
print("=== Original (full, 24L) ===")
full = finetune(build_subnetwork(list(range(L))), ft_train)
full_total = count_params(full)
acc = evaluate(full, val_split)
print(f"  -> acc={acc*100:.2f}%  params={full_total/1e6:.1f}M")
results.append({"method": "Original", "k": L, "layers": list(range(1, L + 1)),
                "params_M": full_total/1e6, "ratio": 1.0, "acc": acc})
del full; gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None

# ---- ProbeScale vs baselines (Algorithm 1 lines 32-35) ----
for k in K_BUDGETS:
    for method in METHODS:
        layers = selectors[method](k)
        print(f"\n=== {method} | k={k} | layers={[i+1 for i in layers]} ===")
        m = finetune(build_subnetwork(layers), ft_train)
        p = count_params(m)
        acc = evaluate(m, val_split)
        print(f"  -> acc={acc*100:.2f}%  params={p/1e6:.1f}M ({p/full_total*100:.1f}%)")
        results.append({"method": method, "k": k, "layers": [i + 1 for i in layers],
                        "params_M": p/1e6, "ratio": p/full_total, "acc": acc})
        del m; gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None
""")

# ---------------------------------------------------------------------------
md(r"""## 6. 結果 — 論文 Table 1 の再現

`acc_retained_%`（元モデルに対する維持率）が **95–98%** 付近で、かつ各予算で **ProbeScale ≥ Top-k / Uniform-k** なら、論文の主張＝実装の妥当性が再現できています。
""")

code(r"""
df = pd.DataFrame(results)
orig_acc = df.loc[0, "acc"]
df["acc_%"] = df["acc"] * 100
df["ratio_%"] = df["ratio"] * 100
df["acc_retained_%"] = df["acc"] / orig_acc * 100
show = df[["method", "k", "layers", "params_M", "ratio_%", "acc_%", "acc_retained_%"]].round(2)
show
""")

code(r"""
print("="*72)
print(f"PAPER REPRODUCTION — {MODEL_NAME} / {TASK.upper()}")
print("="*72)
print(f"Original (full {L}L): acc={df.loc[0,'acc_%']:.2f}%  params={df.loc[0,'params_M']:.1f}M\n")
for k in K_BUDGETS:
    print(f"--- k={k} layers ---")
    for _, r in df[df.k == k].iterrows():
        print(f"  {r['method']:11s} | {r['params_M']:6.1f}M ({r['ratio_%']:4.1f}%) | "
              f"acc {r['acc_%']:5.2f}% | retained {r['acc_retained_%']:5.1f}%")
    print()
""")

code(r"""
plt.figure(figsize=(8, 5))
markers = {"ProbeScale": "o", "Top-k": "s", "Uniform-k": "^"}
colors = {"ProbeScale": "#C44E52", "Top-k": "#4C72B0", "Uniform-k": "#55A868"}
for method in METHODS:
    sub = df[df.method == method].sort_values("ratio")
    plt.plot(sub["ratio_%"], sub["acc_%"], marker=markers[method], color=colors[method],
             label=method, linewidth=2, markersize=9)
plt.scatter([100], [df.loc[0, "acc_%"]], color="black", marker="*", s=250, zorder=5, label="Original")
plt.xlabel("Parameter ratio (%)"); plt.ylabel(f"{TASK.upper()} accuracy (%)")
plt.title(f"ProbeScale paper reproduction ({MODEL_NAME} / {TASK.upper()})")
plt.legend(); plt.grid(alpha=0.3); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""## 7. まとめ

- 論文の実設定（**RoBERTa-Large × SST-2**）で ProbeScale を再現しました。**層プローブ→関連度(式1)→連続ブロック選択(式2)→抽出＋3エポック fine-tune** という Algorithm 1 をそのまま実装。
- 選択関数（`select_probescale/topk/uniform`）は **OCR ノートブックと同一**。ここで論文の傾向（ProbeScale ≥ ベースライン、維持率 95–98%）が出れば、**実装が正しい**ことの裏付けになります。
- 分類タスクは「プローブ＝最終ヘッド」が成立するため軽量ヘッドの再学習で高維持率になり、生成（OCR）が難しかった理由が対比で理解できます。

### 調整ノブ
- `N_FT_TRAIN` を 67349（SST-2 全量）に上げると論文値（Original≈96.5, ProbeScale-6≈94.9）に近づきます
- `MODEL_NAME="roberta-base"` で高速化（12層）/ `TASK="qnli"` で別タスク
- `K_BUDGETS` を増やして圧縮率-精度カーブを密に
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
out = os.path.join(os.path.dirname(__file__), "ProbeScale_Paper_Repro.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "with", len(cells), "cells")
