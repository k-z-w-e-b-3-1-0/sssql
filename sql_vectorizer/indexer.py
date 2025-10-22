"""Tools for building and searching a TF-IDF index of SQL files."""
from __future__ import annotations

import csv
import json
import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple


class SimpleTfidfVectorizer:
    """A lightweight TF-IDF vectorizer tailored for SQL token patterns."""

    _default_token_pattern = re.compile(r"[\w一-龯ぁ-んァ-ヶー]+", re.UNICODE)

    def __init__(
        self,
        *,
        stop_words: Sequence[str] | None = None,
        token_pattern: re.Pattern[str] | None = None,
    ) -> None:
        self.stop_words = {word.lower() for word in stop_words} if stop_words else None
        self.vocabulary_: dict[str, int] = {}
        self.idf_: List[float] = []
        self.token_pattern: re.Pattern[str] = token_pattern or self._default_token_pattern

    # pickle support -----------------------------------------------------
    def __getstate__(self) -> dict:
        return {
            "stop_words": self.stop_words,
            "vocabulary_": self.vocabulary_,
            "idf_": self.idf_,
            "token_pattern": self.token_pattern,
        }

    def __setstate__(self, state: dict) -> None:
        self.stop_words = state["stop_words"]
        self.vocabulary_ = state["vocabulary_"]
        self.idf_ = state["idf_"]
        self.token_pattern = state.get("token_pattern", self._default_token_pattern)

    # public API ---------------------------------------------------------
    def fit_transform(self, documents: Sequence[str]) -> List[List[float]]:
        tokenized = [self._tokenize(doc) for doc in documents]
        self._build_vocabulary(tokenized)
        self._compute_idf(tokenized)
        return [self._tfidf_vector(tokens) for tokens in tokenized]

    def transform(self, documents: Sequence[str]) -> List[List[float]]:
        return [self._tfidf_vector(self._tokenize(doc)) for doc in documents]

    # helpers -------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        tokens = [match.group(0).lower() for match in self.token_pattern.finditer(text)]
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [token for token in tokens if not token.isdigit()]
        return tokens

    def _build_vocabulary(self, tokenized_docs: Sequence[Sequence[str]]) -> None:
        vocabulary: dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
        self.vocabulary_ = vocabulary

    def _compute_idf(self, tokenized_docs: Sequence[Sequence[str]]) -> None:
        doc_count = len(tokenized_docs)
        df = [0] * len(self.vocabulary_)
        for tokens in tokenized_docs:
            seen: set[str] = set()
            for token in tokens:
                if token in self.vocabulary_ and token not in seen:
                    df[self.vocabulary_[token]] += 1
                    seen.add(token)
        self.idf_ = [
            math.log((1 + doc_count) / (1 + freq)) + 1.0 if freq else 0.0
            for freq in df
        ]

    def _tfidf_vector(self, tokens: Sequence[str]) -> List[float]:
        if not self.vocabulary_:
            return []

        counts = Counter(token for token in tokens if token in self.vocabulary_)
        total = sum(counts.values())
        vector = [0.0] * len(self.vocabulary_)
        if total == 0:
            return vector
        for token, term_freq in counts.items():
            idx = self.vocabulary_[token]
            idf = self.idf_[idx] if idx < len(self.idf_) else 0.0
            vector[idx] = (term_freq / total) * idf
        return vector


@dataclass
class MetadataEntry:
    """Mapping between SQL identifiers and Japanese logical names."""

    table: str | None
    column: str | None
    aliases: Tuple[str, ...] = ()
    table_lower: str | None = field(default=None, init=False)
    column_lower: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.table_lower = self.table.lower() if self.table else None
        self.column_lower = self.column.lower() if self.column else None

    def matched_by(self, identifiers: set[str]) -> bool:
        table_ok = self.table_lower is None or self.table_lower in identifiers
        column_ok = self.column_lower is None or self.column_lower in identifiers
        return table_ok and column_ok


@dataclass
class SqlVectorIndex:
    """Represents a TF-IDF index of SQL files."""

    vectorizer: SimpleTfidfVectorizer
    matrix: List[List[float]]
    file_paths: List[Path]
    metadata: Tuple["MetadataEntry", ...] = ()

    @classmethod
    def from_folder(
        cls,
        folder: Path,
        *,
        encoding: str = "utf-8",
        stop_words: Sequence[str] | None = None,
        metadata: Sequence["MetadataEntry"] | None = None,
    ) -> "SqlVectorIndex":
        """Create an index from all ``.sql`` files in ``folder``."""
        folder = folder.expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder}")

        sql_files = sorted(folder.rglob("*.sql"))
        if not sql_files:
            raise ValueError(f"No .sql files found under {folder}")

        documents = [
            _augment_with_metadata(
                _read_sql_file(path, encoding=encoding), metadata or ()
            )
            for path in sql_files
        ]
        vectorizer = SimpleTfidfVectorizer(stop_words=stop_words)
        matrix = vectorizer.fit_transform(documents)

        return cls(
            vectorizer=vectorizer,
            matrix=matrix,
            file_paths=sql_files,
            metadata=tuple(metadata or ()),
        )

    def to_dict(self) -> dict:
        """Serialize the index metadata to a JSON-compatible dict."""
        return {
            "file_paths": [str(path) for path in self.file_paths],
        }

    def save(self, destination: Path) -> None:
        """Persist the index to ``destination`` using pickle for the model."""
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "file_paths": self.file_paths,
            "metadata": self.metadata,
        }
        with destination.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, source: Path) -> "SqlVectorIndex":
        """Load a previously saved index."""
        source = source.expanduser().resolve()
        with source.open("rb") as f:
            payload = pickle.load(f)
        return cls(
            vectorizer=payload["vectorizer"],
            matrix=payload["matrix"],
            file_paths=[Path(p) for p in payload["file_paths"]],
            metadata=tuple(payload.get("metadata", ())),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_scores: bool = True,
    ) -> List[Tuple[Path, float] | Path]:
        """Return the ``top_k`` most similar SQL files to ``query``."""
        if not query.strip():
            raise ValueError("Query must not be empty")
        query_with_aliases = _augment_query_with_metadata(query, self.metadata, self.vectorizer)
        query_vec = self.vectorizer.transform([query_with_aliases])[0]
        similarities = _cosine_similarity(query_vec, self.matrix)
        ranked = sorted(
            enumerate(similarities), key=lambda item: item[1], reverse=True
        )[:top_k]
        results: List[Tuple[Path, float] | Path] = []
        for idx, score in ranked:
            path = self.file_paths[idx]
            if include_scores:
                results.append((path, score))
            else:
                results.append(path)
        return results


def _read_sql_file(path: Path, *, encoding: str) -> str:
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding,
            exc.object,
            exc.start,
            exc.end,
            f"Failed to decode {path}: {exc.reason}",
        ) from exc


def vectorize_folder(
    folder: str | Path,
    *,
    output: str | Path,
    encoding: str = "utf-8",
    stop_words: Sequence[str] | None = None,
    metadata: Sequence[str | Path] | None = None,
    metadata_encoding: str | None = None,
) -> SqlVectorIndex:
    """Build an index from ``folder`` and save it to ``output``."""
    metadata_entries = _load_metadata(metadata or (), encoding=metadata_encoding or encoding)
    index = SqlVectorIndex.from_folder(
        Path(folder),
        encoding=encoding,
        stop_words=stop_words,
        metadata=metadata_entries,
    )
    index.save(Path(output))
    return index


def load_index(path: str | Path) -> SqlVectorIndex:
    """Convenience wrapper around :meth:`SqlVectorIndex.load`."""
    return SqlVectorIndex.load(Path(path))


_SQL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _augment_with_metadata(
    sql_text: str, metadata: Sequence[MetadataEntry]
) -> str:
    if not metadata:
        return sql_text

    identifiers = {match.group(0).lower() for match in _SQL_IDENTIFIER.finditer(sql_text)}
    appended: List[str] = []
    for entry in metadata:
        if not entry.matched_by(identifiers):
            continue
        alias_text = " ".join(alias for alias in entry.aliases if alias)
        if entry.column:
            descriptor = f"column {entry.table or ''} {entry.column}".strip()
        elif entry.table:
            descriptor = f"table {entry.table}"
        else:
            descriptor = "identifier"
        if alias_text:
            appended.append(f"{descriptor} {alias_text}")
        else:
            appended.append(descriptor)

    if not appended:
        return sql_text
    return f"{sql_text}\n\n-- METADATA\n" + "\n".join(appended)


def _augment_query_with_metadata(
    query: str, metadata: Sequence[MetadataEntry], vectorizer: SimpleTfidfVectorizer
) -> str:
    if not metadata:
        return query

    expansions: List[str] = []
    query_lower = query.lower()
    for entry in metadata:
        matched = False
        for alias in entry.aliases:
            alias_lower = alias.lower()
            if alias_lower and alias_lower in query_lower:
                matched = True
                break
        if not matched:
            continue

        english_parts: List[str] = []
        if entry.table:
            english_parts.append(entry.table)
        if entry.column:
            english_parts.append(entry.column)
        if english_parts:
            expansions.append(" ".join(english_parts))

    if not expansions:
        return query

    return query + "\n" + "\n".join(expansions)


def _cosine_similarity(query: Sequence[float], matrix: Sequence[Sequence[float]]) -> List[float]:
    query_norm = math.sqrt(sum(value * value for value in query))
    if query_norm == 0:
        return [0.0 for _ in matrix]

    similarities: List[float] = []
    for row in matrix:
        dot = sum(a * b for a, b in zip(query, row))
        row_norm = math.sqrt(sum(value * value for value in row))
        if row_norm == 0:
            similarities.append(0.0)
        else:
            similarities.append(dot / (query_norm * row_norm))
    return similarities


def _load_metadata(
    paths: Sequence[str | Path], *, encoding: str
) -> Tuple[MetadataEntry, ...]:
    if not paths:
        return ()

    entries: List[MetadataEntry] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Metadata file does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            entries.extend(_load_metadata_csv(path, encoding=encoding))
        elif suffix == ".json":
            entries.extend(_load_metadata_json(path, encoding=encoding))
        else:
            raise ValueError(
                f"Unsupported metadata format for {path}. Expected .csv or .json."
            )
    return tuple(entries)


def _load_metadata_csv(path: Path, *, encoding: str) -> List[MetadataEntry]:
    loaded: List[MetadataEntry] = []
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = _metadata_entry_from_mapping(row, source=str(path))
            if entry:
                loaded.append(entry)
    return loaded


def _load_metadata_json(path: Path, *, encoding: str) -> List[MetadataEntry]:
    payload = json.loads(path.read_text(encoding=encoding))
    loaded: List[MetadataEntry] = []
    if isinstance(payload, list):
        iterable: Iterable = payload
    elif isinstance(payload, dict):
        iterable = payload.get("entries", [])
    else:
        raise ValueError(f"Unsupported JSON structure in metadata file: {path}")

    for item in iterable:
        if isinstance(item, dict):
            entry = _metadata_entry_from_mapping(item, source=str(path))
            if entry:
                loaded.append(entry)
    return loaded


def _metadata_entry_from_mapping(
    data: Mapping[str, object], *, source: str
) -> MetadataEntry | None:
    table = _extract_string(data, {"table", "table_name"})
    column = _extract_string(data, {"column", "column_name"})
    aliases = _extract_aliases(data, source=source)
    if not aliases:
        return None
    return MetadataEntry(table=table, column=column, aliases=tuple(aliases))


def _extract_string(data: Mapping[str, object], keys: set[str]) -> str | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_aliases(data: Mapping[str, object], *, source: str) -> List[str]:
    candidates: List[str] = []
    for key in ("japanese", "japanese_name", "logical_name", "aliases", "alias"):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            candidates.extend(_split_alias_string(value))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, str):
                    candidates.extend(_split_alias_string(item))
        else:
            raise ValueError(
                f"Unsupported alias type in metadata file {source}: {type(value)!r}"
            )
    return [alias for alias in (alias.strip() for alias in candidates) if alias]


def _split_alias_string(value: str) -> List[str]:
    parts = re.split(r"[,;\s]+", value)
    return [part for part in parts if part]


def main(argv: Sequence[str] | None = None) -> None:
    """Simple CLI for indexing folders and searching SQL similarity."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    index_parser = sub.add_parser("index", help="Vectorize a folder of .sql files")
    index_parser.add_argument("folder", type=str, help="Folder containing .sql files")
    index_parser.add_argument(
        "--output",
        type=str,
        default="sql_index.pkl",
        help="Where to store the generated index",
    )
    index_parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Text encoding used to read SQL files",
    )
    index_parser.add_argument(
        "--metadata",
        type=str,
        nargs="+",
        default=None,
        help="Paths to .csv or .json files containing table/column Japanese logical names.",
    )
    index_parser.add_argument(
        "--metadata-encoding",
        type=str,
        default=None,
        help="Encoding used to read metadata files (defaults to --encoding).",
    )

    search_parser = sub.add_parser(
        "search", help="Search the index for SQL files similar to a query"
    )
    search_parser.add_argument(
        "index",
        type=str,
        help="Path to the saved index file",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="SQL query string to compare",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar SQL files to return",
    )

    args = parser.parse_args(argv)

    if args.command == "index":
        index = vectorize_folder(
            args.folder,
            output=args.output,
            encoding=args.encoding,
            metadata=args.metadata,
            metadata_encoding=args.metadata_encoding,
        )
        metadata_path = Path(args.output).with_suffix(".json")
        metadata_path.write_text(json.dumps(index.to_dict(), indent=2))
        print(f"Indexed {len(index.file_paths)} SQL files. Index saved to {args.output}.")
    elif args.command == "search":
        index = load_index(args.index)
        results = index.search(args.query, top_k=args.top_k, include_scores=True)
        for path, score in results:
            print(f"{score:.4f}\t{path}")
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
