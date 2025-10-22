# SQL類似検索ツールキット

このリポジトリは、指定したフォルダ配下のSQLファイルをベクトル化し、類似したSQLを検索するためのユーティリティを提供します。

## 特長

1. 指定フォルダ内のすべての`.sql`ファイルをTF-IDFベースのベクトル空間モデルでインデックス化し、結果をディスクに保存できます。
2. 入力した任意のSQL文とインデックス済みコレクションを比較し、類似度の高いSQLファイルを取得できます。
3. テーブル・カラムの日本語論理名メタデータを読み込み、SQL本文に自動で付加して日本語クエリとのマッチング精度を高められます。

## 使い方（CLI）

本ツールキットはシンプルなコマンドラインインターフェースを提供しています。

```bash
python -m sql_vectorizer.indexer index path/to/sql/folder --output sql_index.pkl
```

上記コマンドは次の2つのファイルを生成します。

- `sql_index.pkl` – TF-IDFモデルとベクトル行列をpickle形式で保存したファイル。
- `sql_index.json` – インデックス対象となったSQLファイルのパスをまとめたメタデータ。

類似SQLを検索する場合は以下のように実行します。

```bash
python -m sql_vectorizer.indexer search sql_index.pkl "SELECT * FROM users"
```

結果件数は`--top-k`で制御できます。

### 日本語メタデータを併用する

テーブル・カラム名と日本語論理名の対応表（CSV/JSON）を用意しておくと、インデックス化の際にSQL本文へ自動的に日本語ラベルが追記されます。これにより「受注テーブルの注文数を知りたい」のような日本語クエリを投げても、該当テーブルを参照するSQLがヒットしやすくなります。

```bash
python -m sql_vectorizer.indexer index samples/sql \
    --output enriched_index.pkl \
    --metadata samples/metadata/table_columns.csv
```

`--metadata`には複数ファイルを指定でき、CSVの場合は`table,column,japanese_name`などの列を、JSONの場合は同等のキーを持つオブジェクトの配列（または`{"entries": [...]}`）を想定しています。複数の日本語ラベルを付与したい場合はスペース/カンマ/セミコロン区切りで列挙してください。

検索時は日本語の論理名がクエリに含まれていれば、対応する英語のテーブル/カラム名を自動で追記したうえでベクトル化されます。

## ライブラリAPI

Pythonコードから直接利用することもできます。

```python
from sql_vectorizer import vectorize_folder, load_index

# インデックスを構築または読み込む
index = vectorize_folder("path/to/sql", output="sql_index.pkl")
# 既存のインデックスを読み込む場合
# index = load_index("sql_index.pkl")

# 類似SQLを検索
for path, score in index.search("SELECT * FROM users", top_k=3):
    print(path, score)
```

## チュートリアル

以下では、付属のサンプルSQLファイルを利用して基本的なワークフローを体験します。

1. サンプルフォルダを指定してインデックス化します。
   ```bash
   python -m sql_vectorizer.indexer index samples/sql --output tutorial_index.pkl
   ```
2. 作成された`tutorial_index.pkl`と`tutorial_index.json`が確認できるはずです。
3. 任意のSQLで検索します。以下は`orders_by_status.sql`を想定した問い合わせ例です。
   ```bash
   python -m sql_vectorizer.indexer search tutorial_index.pkl "SELECT status, COUNT(*) FROM orders GROUP BY status"
   ```
4. 類似度スコアとともに、最も近いSQLファイルのパスが表示されます。

これで、フォルダ内のSQLスニペットを素早く比較・検索する準備が整いました。必要に応じて`--top-k`や`--encoding`などのオプションを調整してください。

### 付属サンプル

`samples/sql`ディレクトリには以下のようなサンプルクエリが用意されています。

- `orders_by_status.sql`
- `customers_active.sql`
- `top_products_by_revenue.sql`
- `monthly_sales_summary.sql`
- `overdue_invoices.sql`
- `user_login_activity.sql`
- `inventory_low_stock.sql`
- `employee_performance_scores.sql`
- `revenue_by_region.sql`
- `cancellations_by_reason.sql`
- `product_returns.sql`

これらのSQLファイルを利用してすぐに検索を試したい場合は、前述のチュートリアルと同様に`samples/sql`をインデックス化してください。例えば次のようにコマンドを実行すると、`samples/sql_index.pkl`と`sql_index.json`が生成されます。

```bash
python -m sql_vectorizer.indexer index samples/sql --output samples/sql_index.pkl
python -m sql_vectorizer.indexer search samples/sql_index.pkl "SELECT status, COUNT(*) FROM orders GROUP BY status"
```

生成されたインデックスを使って、お好きなSQLで検索してみてください。

## 使い方（Web API）

CLIだけでなく、FastAPIベースのWeb APIを用意しており、HTTP経由で検索を実行できます。

1. 事前にインデックスを作成しておきます（例: `samples/sql_index.pkl`）。
2. 以下のようにアプリケーションを起動します。

   ```bash
   uvicorn sql_vectorizer.webapi:app --reload
   ```

   既定ではリクエストボディに`index`フィールドを指定することで、任意のインデックスファイルを利用できます。特定のインデックスを常に使用したい場合は、`sql_vectorizer.webapi:create_app("path/to/index.pkl")`のようにアプリを生成し、Uvicornなどに渡してください。

3. `POST /search` エンドポイントに対して次のようなJSONを送信すると、CLIと同じ検索結果を取得できます。

   ```json
   {
     "query": "SELECT status, COUNT(*) FROM orders GROUP BY status",
     "top_k": 5,
     "index": "samples/sql_index.pkl"
   }
   ```

   レスポンスは、検索結果のパスとスコアの配列を含むJSONになります。

## Windows環境でのセットアップ手順

WindowsでもPythonさえインストールされていればCLIおよびWeb APIのどちらも動作します。以下はPowerShellを想定した一連の手順です。

1. [python.org](https://www.python.org/downloads/windows/)からPython 3.10以降をインストールし、セットアップ時に「Add python.exe to PATH」にチェックを入れます。
2. 作業したいフォルダを開き、PowerShellで次のように仮想環境を作成してアクティベートします。
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   py -m pip install --upgrade pip
   ```
   *もし「このシステムではスクリプトの実行が無効になっています」と表示された場合は、`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`を実行してから再度アクティベートしてください。*
3. CLIのみを利用する場合は追加パッケージは不要です。Web APIを使う場合は次の依存関係をインストールします。
   ```powershell
   py -m pip install "fastapi>=0.100,<1" "uvicorn[standard]>=0.20,<1"
   ```
4. インデックス作成や検索は通常どおり `py -m sql_vectorizer.indexer ...` の形式で実行できます。インデックス作成の例:
   ```powershell
   py -m sql_vectorizer.indexer index samples/sql --output samples/sql_index.pkl
   ```
5. Web APIを使う場合は、仮想環境を有効にした状態で以下のコマンドを実行し、ブラウザやHTTPクライアントからアクセスしてください。
   ```powershell
   uvicorn sql_vectorizer.webapi:app --reload --host 127.0.0.1 --port 8000
   ```

これでWindows上でも同様の手順でCLIとWeb APIの両方を利用できます。必要に応じてPowerShellの代わりにコマンドプロンプト（`cmd.exe`）を用いる場合は、アクティベートコマンドが `\.venv\Scripts\activate.bat` になる点だけ読み替えてください。
