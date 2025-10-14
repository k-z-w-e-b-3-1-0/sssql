"""Tools for building and searching a TF-IDF index of SQL files."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SqlVectorIndex:
    """Represents a TF-IDF index of SQL files."""

    vectorizer: TfidfVectorizer
    matrix: np.ndarray
    file_paths: List[Path]

    @classmethod
    def from_folder(
        cls,
        folder: Path,
        *,
        encoding: str = "utf-8",
        stop_words: Sequence[str] | None = None,
    ) -> "SqlVectorIndex":
        """Create an index from all ``.sql`` files in ``folder``."""
        folder = folder.expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder}")

        sql_files = sorted(folder.rglob("*.sql"))
        if not sql_files:
            raise ValueError(f"No .sql files found under {folder}")

        documents = [_read_sql_file(path, encoding=encoding) for path in sql_files]
        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(documents).astype(np.float32)

        return cls(vectorizer=vectorizer, matrix=matrix, file_paths=sql_files)

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
        query_vec = self.vectorizer.transform([query]).astype(np.float32)
        similarities = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results: List[Tuple[Path, float] | Path] = []
        for idx in top_indices:
            path = self.file_paths[idx]
            score = float(similarities[idx])
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
) -> SqlVectorIndex:
    """Build an index from ``folder`` and save it to ``output``."""
    index = SqlVectorIndex.from_folder(
        Path(folder), encoding=encoding, stop_words=stop_words
    )
    index.save(Path(output))
    return index


def load_index(path: str | Path) -> SqlVectorIndex:
    """Convenience wrapper around :meth:`SqlVectorIndex.load`."""
    return SqlVectorIndex.load(Path(path))


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
            args.folder, output=args.output, encoding=args.encoding
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
