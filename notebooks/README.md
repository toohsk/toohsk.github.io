# Notebooks

論文の再現・実験用ノートブック置き場です（ブログ本文 `content/` とは別管理）。

## ProbeScale_OCR_TrOCR.ipynb

論文 **ProbeScale: Probing Analysis to Optimize Neural Scaling Laws for Efficient Small Language Model Inference**（Sourav Das, arXiv:2606.01806）のアルゴリズムを、**AI OCR（画像→文字列抽出）タスク**へ適用して再現する Google Colab ノートブックです。

- **ベースモデル**: [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten)（Vision-Encoder-Decoder。DeiT エンコーダ12層を圧縮対象にする）
- **データセット**: [`Teklia/IAM-line`](https://huggingface.co/datasets/Teklia/IAM-line)（IAM 手書き行、image→text）
- **実装するもの**:
  1. **Step 1 – 層プロービング**: 各エンコーダ層の特徴に線形 CTC プローブを載せ、文字認識性能 `V_l = 1 − CER` を層関連度 `R_l` として算出
  2. **Step 2 – 連続ブロック選択 (Algorithm 1)**: 関連度の総和を最大化する連続 `k` 層を選択
  3. **Step 3 – サブネット抽出 + fine-tune**: 選んだ層だけ残してデコーダはそのまま、短く fine-tune
- **比較する指標**:
  - **圧縮率**: 元モデル vs 圧縮モデルのパラメータ数・比率（エンコーダ単体／全体）
  - **精度**: テストの **CER / WER**、および元モデルに対する精度維持率。ベースライン Top-k / Uniform-k と並べて比較（論文 Table 1 / Fig. 2 に対応）

### 使い方（Colab）

1. [Google Colab](https://colab.research.google.com/) で本ノートブックを開く（`ファイル → ノートブックをアップロード`、または GitHub URL から開く）
2. `ランタイム → ランタイムのタイプを変更 → GPU`（T4 以上推奨）
3. 上から順に実行。規模は冒頭の「実験設定」セル（`N_PROBE_TRAIN`, `N_FT_TRAIN`, `K_BUDGETS`, `FT_EPOCHS` など）で調整できます

### 論文との対応（実行しながら学べる設計）

ノートブックには「論文の数式とアルゴリズム（実装の対応表）」セクションがあり、**式(1)（層関連度）・式(2)（予算制約付き選択）・Algorithm 1 の擬似コード**を行番号つきで再掲しています。各コードセルには `# [Paper Alg.1 line N ...]` や `# ... Eq.(1)` といったコメントを付け、実装の各行がどの式・どのアルゴリズム行・どの節に対応するかを実行時に追えるようにしています。

- **Step 1**（層プロービング）= Algorithm 1 `CalculateRelevance` / 式(1)
- **Step 2**（連続ブロック選択）= Algorithm 1 `SelectSubnetwork` / 式(2)
- **Step 3**（抽出 + fine-tune）= Algorithm 1 line 34–35 / 第3節 *Subnetwork Extraction and Fine-tuning*

### 補足

OCR は系列出力タスクのため、論文の「単一ベクトルに対する線形分類プローブ」をそのままは使えません。本ノートブックは各層のパッチトークン系列に線形 CTC ヘッドを載せて文字列を読み取らせ、その性能を層関連度とします。これは「プローブ性能を層選択の*相対的*スコアへ転用する」という ProbeScale の中核思想に忠実な拡張です。

`_gen_probescale_ocr.py` はこの `.ipynb` を生成するスクリプトです（差分レビューを容易にするため同梱）。
