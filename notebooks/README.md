# Notebooks

論文の再現・実験用ノートブック置き場です（ブログ本文 `content/` とは別管理）。

## ProbeScale_OCR_TrOCR.ipynb

論文 **ProbeScale: Probing Analysis to Optimize Neural Scaling Laws for Efficient Small Language Model Inference**（Sourav Das, arXiv:2606.01806）のアルゴリズムを、**AI OCR（画像→文字列抽出）タスク**へ適用して再現する Google Colab ノートブックです。

- **ベースモデル**: [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten) の **ビジョンエンコーダ（DeiT 12層）** を圧縮対象にする
- **データセット**: [`Teklia/IAM-line`](https://huggingface.co/datasets/Teklia/IAM-line)（IAM 手書き行、image→text）
- **方式（案B: 論文忠実な CTC ヘッド）**: 論文は「エンコーダ層 + 軽量タスクヘッド」を抽出してヘッドを再学習します。OCR の忠実な対応として、圧縮モデルを **`[パッチ埋め込み → 選択エンコーダ層 → 線形 CTC ヘッド]`** とし、巨大な生成デコーダは使いません（生成デコーダ版は論文の将来課題で、Colab 予算では回復困難なことを実験で確認済み）。
  1. **Step 1 – 層プロービング**: 各エンコーダ層の特徴に線形 CTC プローブを載せ、文字認識性能 `V_l = 1 − CER` を層関連度 `R_l` として算出（式(1)）
  2. **Step 2 – 連続ブロック選択 (Algorithm 1)**: 関連度の総和を最大化する連続 `k` 層を選択（式(2)）
  3. **Step 3 – サブネット構築 + 学習**: 選択層 + 線形 CTC ヘッドを構築し、短く学習（line 34–35）
- **比較する指標**:
  - **圧縮率**: フル12層エンコーダ基準のエンコーダ・パラメータ比（`encoder_ratio_%`）。さらに生成デコーダ(247M)を持たないため元 TrOCR(333.9M) 比でも大幅に小型
  - **精度**: テストの **CER / WER**、およびフルエンコーダ CTC に対する精度維持率。ベースライン Top-k / Uniform-k と並べて比較（論文 Table 1 / Fig. 2 に対応）

### 使い方（Colab）

1. [Google Colab](https://colab.research.google.com/) で本ノートブックを開く（`ファイル → ノートブックをアップロード`、または GitHub URL から開く）
2. `ランタイム → ランタイムのタイプを変更 → GPU`（T4 以上推奨）
3. 上から順に実行。規模は冒頭の「実験設定」セル（`N_PROBE_TRAIN`, `N_FT_TRAIN`, `K_BUDGETS`, `FT_EPOCHS` など）で調整できます

### 論文との対応（実行しながら学べる設計）

ノートブックには「論文の数式とアルゴリズム（実装の対応表）」セクションがあり、**式(1)（層関連度）・式(2)（予算制約付き選択）・Algorithm 1 の擬似コード**を行番号つきで再掲しています。各コードセルには `# [Paper Alg.1 line N ...]` や `# ... Eq.(1)` といったコメントを付け、実装の各行がどの式・どのアルゴリズム行・どの節に対応するかを実行時に追えるようにしています。

- **Step 1**（層プロービング）= Algorithm 1 `CalculateRelevance` / 式(1)
- **Step 2**（連続ブロック選択）= Algorithm 1 `SelectSubnetwork` / 式(2)
- **Step 3**（選択層 + CTC ヘッドの構築・学習）= Algorithm 1 line 34–35 / 第3節 *Subnetwork Extraction and Fine-tuning*

### 補足

OCR は系列出力タスクのため、論文の「単一ベクトルに対する線形分類プローブ」をそのままは使えません。本ノートブックは各層のパッチトークン系列に線形 CTC ヘッドを載せて文字列を読み取らせ、その性能を層関連度とします。これは「プローブ性能を層選択の*相対的*スコアへ転用する」という ProbeScale の中核思想に忠実な拡張です。

### トラブルシューティング

- **`ImportError: cannot import name '_Ink' from 'PIL._typing'`**（import セルで発生）: Colab 同梱の Pillow がアップグレードで混在状態になったのが原因です。次を実行して Pillow を整合した版に入れ直し、**ランタイムを再起動**（ランタイム → セッションを再起動）してから import セル以降を実行してください。

  ```python
  !pip -q install --force-reinstall --no-deps "Pillow==11.3.0"
  ```

  なお最新版のセットアップセルは Pillow を固定インストールするため、この問題は起きません。

`_gen_probescale_ocr.py` はこの `.ipynb` を生成するスクリプトです（差分レビューを容易にするため同梱）。
