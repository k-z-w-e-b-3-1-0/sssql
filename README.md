# SQL類似検索ツールキット

このリポジトリは、指定したフォルダ配下のSQLファイルをベクトル化し、類似したSQLを検索するためのユーティリティを提供します。

## 特長

1. 指定フォルダ内のすべての`.sql`ファイルをTF-IDFベースのベクトル空間モデルでインデックス化し、結果をディスクに保存できます。
2. 入力した任意のSQL文とインデックス済みコレクションを比較し、類似度の高いSQLファイルを取得できます。

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
