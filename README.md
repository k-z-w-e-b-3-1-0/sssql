# SQL Similarity Toolkit

This repository provides utilities for vectorising SQL files and searching for
similar SQL statements.

## Features

1. Index every `.sql` file under a target folder using a TF-IDF vector space
   model and store the result on disk.
2. Compare an input SQL statement with the indexed collection to retrieve the
   most similar SQL files.

## Usage

The toolkit exposes a small command line interface:

```bash
python -m sql_vectorizer.indexer index path/to/sql/folder --output sql_index.pkl
```

The command above creates two files:

* `sql_index.pkl` – a pickled representation of the TF-IDF model and the vector
  matrix.
* `sql_index.json` – metadata listing the indexed file paths.

To search for similar SQL files with a query:

```bash
python -m sql_vectorizer.indexer search sql_index.pkl "SELECT * FROM users"
```

Use `--top-k` to control how many results are shown.

## Library API

You can also integrate the functionality directly into Python code:

```python
from sql_vectorizer import vectorize_folder, load_index

# Build or load an index
index = vectorize_folder("path/to/sql", output="sql_index.pkl")
# or index = load_index("sql_index.pkl")

# Search for similar SQL strings
for path, score in index.search("SELECT * FROM users", top_k=3):
    print(path, score)
```
