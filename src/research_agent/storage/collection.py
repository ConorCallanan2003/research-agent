"""Simple vector collection using SQLite for storage and hnswlib for search."""

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np


@dataclass
class Embedding:
    """An embedding with text and metadata."""

    vector: list[float]
    text: str
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: datetime.utcnow().timestamp())


@dataclass
class SearchResult:
    """A search result with distance."""

    text: str
    doc_id: str
    metadata: dict[str, Any]
    distance: float
    created_at: float


class Collection:
    """
    A vector collection backed by SQLite and hnswlib.

    Stores text and metadata in SQLite, vectors in hnswlib index.
    Both are persisted to disk.
    """

    def __init__(
        self,
        name: str,
        dimension: int,
        path: str | Path,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ):
        """
        Initialize or load a collection.

        Args:
            name: Collection name (used for filenames)
            dimension: Vector dimension
            path: Directory to store files
            ef_construction: hnswlib ef_construction parameter
            M: hnswlib M parameter
            ef_search: hnswlib ef parameter for search
        """
        self.name = name
        self.dimension = dimension
        self.path = Path(path)
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search

        self._lock = threading.Lock()
        self._db_path = self.path / f"{name}.db"
        self._index_path = self.path / f"{name}.index"

        # Ensure directory exists
        self.path.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self._init_db()

        # Initialize or load hnswlib index
        self._init_index()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY,
                    doc_id TEXT UNIQUE NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON embeddings(doc_id)")
            conn.commit()

    def _init_index(self) -> None:
        """Initialize or load hnswlib index."""
        self._index = hnswlib.Index(space="cosine", dim=self.dimension)

        if self._index_path.exists():
            # Load existing index
            self._index.load_index(str(self._index_path))
            self._index.set_ef(self.ef_search)
        else:
            # Create new index
            # Start with capacity for 1000 items, will resize as needed
            self._index.init_index(
                max_elements=1000,
                ef_construction=self.ef_construction,
                M=self.M,
            )
            self._index.set_ef(self.ef_search)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection."""
        return sqlite3.connect(self._db_path)

    def _resize_index_if_needed(self, new_count: int) -> None:
        """Resize index if we're running out of space."""
        current_max = self._index.get_max_elements()
        if new_count >= current_max:
            new_max = max(current_max * 2, new_count + 1000)
            self._index.resize_index(new_max)

    def add(self, embedding: Embedding) -> None:
        """Add a single embedding to the collection."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO embeddings (doc_id, text, metadata, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        embedding.doc_id,
                        embedding.text,
                        json.dumps(embedding.metadata),
                        embedding.created_at,
                    ),
                )
                row_id = cursor.lastrowid
                conn.commit()

            # Add to index
            self._resize_index_if_needed(row_id)
            vector = np.array([embedding.vector], dtype=np.float32)
            self._index.add_items(vector, [row_id])

            # Persist index
            self._index.save_index(str(self._index_path))

    def add_batch(self, embeddings: list[Embedding]) -> None:
        """Add multiple embeddings efficiently."""
        if not embeddings:
            return

        with self._lock:
            row_ids = []
            vectors = []

            with self._get_connection() as conn:
                for emb in embeddings:
                    cursor = conn.execute(
                        """
                        INSERT INTO embeddings (doc_id, text, metadata, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            emb.doc_id,
                            emb.text,
                            json.dumps(emb.metadata),
                            emb.created_at,
                        ),
                    )
                    row_ids.append(cursor.lastrowid)
                    vectors.append(emb.vector)
                conn.commit()

            # Add to index
            self._resize_index_if_needed(max(row_ids))
            vectors_np = np.array(vectors, dtype=np.float32)
            self._index.add_items(vectors_np, row_ids)

            # Persist index
            self._index.save_index(str(self._index_path))

    def search(self, query_vector: list[float], k: int = 10) -> list[SearchResult]:
        """
        Search for nearest neighbors.

        Args:
            query_vector: Query embedding
            k: Number of results to return

        Returns:
            List of SearchResult objects, sorted by distance (ascending)
        """
        current_count = self.count()
        if current_count == 0:
            return []

        # Can't return more than we have
        k = min(k, current_count)

        with self._lock:
            query = np.array([query_vector], dtype=np.float32)
            labels, distances = self._index.knn_query(query, k=k)

            row_ids = labels[0].tolist()
            dists = distances[0].tolist()

        # Fetch from SQLite
        results = []
        with self._get_connection() as conn:
            for row_id, distance in zip(row_ids, dists):
                cursor = conn.execute(
                    "SELECT doc_id, text, metadata, created_at FROM embeddings WHERE id = ?",
                    (row_id,),
                )
                row = cursor.fetchone()
                if row:
                    results.append(
                        SearchResult(
                            doc_id=row[0],
                            text=row[1],
                            metadata=json.loads(row[2]) if row[2] else {},
                            created_at=row[3],
                            distance=distance,
                        )
                    )

        return results

    def get_all(self, offset: int = 0, limit: int | None = None) -> list[Embedding]:
        """Get all embeddings (without vectors for efficiency)."""
        with self._get_connection() as conn:
            if limit:
                cursor = conn.execute(
                    "SELECT doc_id, text, metadata, created_at FROM embeddings LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            else:
                cursor = conn.execute(
                    "SELECT doc_id, text, metadata, created_at FROM embeddings OFFSET ?",
                    (offset,),
                )

            results = []
            for row in cursor.fetchall():
                results.append(
                    Embedding(
                        vector=[],  # Don't load vectors for efficiency
                        doc_id=row[0],
                        text=row[1],
                        metadata=json.loads(row[2]) if row[2] else {},
                        created_at=row[3],
                    )
                )
            return results

    def count(self) -> int:
        """Get the number of embeddings in the collection."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            return cursor.fetchone()[0]

    def delete(self, doc_id: str) -> bool:
        """
        Delete an embedding by doc_id.

        Note: hnswlib doesn't support deletion, so the vector remains
        in the index but won't be returned in search results.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM embeddings WHERE doc_id = ?",
                    (doc_id,),
                )
                conn.commit()
                return cursor.rowcount > 0

    def clear(self) -> None:
        """Delete all embeddings and reinitialize the index."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM embeddings")
                conn.commit()

            # Reinitialize index
            self._index = hnswlib.Index(space="cosine", dim=self.dimension)
            self._index.init_index(
                max_elements=1000,
                ef_construction=self.ef_construction,
                M=self.M,
            )
            self._index.set_ef(self.ef_search)
            self._index.save_index(str(self._index_path))
