# toohsk.github.io

My GitHub page. [Pelican](https://getpelican.com/) 製の静的サイトです。

## ブランチ構成

- `dev-blog` — ソースブランチ（記事・設定・`Makefile` などを管理）。通常の執筆はここで行います。
- `main` — GitHub Pages の配信用ブランチ。`ghp-import` により `output/` の内容で更新されます。直接コミットしません。

## セットアップ

依存関係は `Pipfile` で管理しています。初回のみ以下を実行します。

```bash
pipenv install
```

## 依存関係の更新

Pelican などのパッケージを更新するときは以下を実行します。

```bash
# 全パッケージを Pipfile の制約内で最新化し、Pipfile.lock を更新
pipenv update

# 特定のパッケージだけ更新する場合
pipenv update pelican

# Pipfile を編集せず lock ファイルだけ作り直す場合
pipenv lock
```

更新後は `pipenv run make devserver` で表示崩れが無いか確認し、`Pipfile` と `Pipfile.lock` の両方をコミットしてください。

## ローカルで確認する

`dev-blog` ブランチで記事を書き、以下のコマンドでプレビューできます。

```bash
# http://localhost:8000 でライブリロード付きサーバを起動
pipenv run make devserver

# 生成物を削除
pipenv run make clean
```

## デプロイ

`dev-blog` ブランチで記事の変更をコミットしたあと、以下を実行するとビルドと公開が一度に行われます。

```bash
pipenv run make github
```

内部では次の処理が走ります（`Makefile` の `github` ターゲット）。

1. `publishconf.py` を使って `output/` に本番用の静的ファイルを生成。
2. `ghp-import -m "Generate Pelican site" -b main output` で `main` ブランチを生成物で更新。
3. `git push origin main` で GitHub Pages に反映。

公開後、反映までは GitHub Pages のビルドに数十秒〜数分かかります。
