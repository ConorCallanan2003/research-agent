"""FastAPI server for the Knowledge Store Explorer."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import hnswlib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from research_agent.config import Config
from research_agent.embeddings import QwenEmbedder

app = FastAPI(title="Knowledge Store Explorer")

# Embedder singleton
_embedder: QwenEmbedder | None = None


def get_embedder() -> QwenEmbedder:
    """Get or create the embedder singleton."""
    global _embedder
    if _embedder is None:
        _embedder = QwenEmbedder()
    return _embedder


@app.on_event("startup")
async def startup_event():
    """Load the embedding model in background on startup."""
    import threading

    def load_embedder():
        print("Loading embedding model in background...")
        embedder = get_embedder()
        if not embedder.is_loaded:
            embedder._load_model()
        print(f"Embedding model loaded on {embedder.device}")

    thread = threading.Thread(target=load_embedder, daemon=True)
    thread.start()

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def get_stores_dir() -> Path:
    """Get the knowledge stores directory."""
    return Config.KNOWLEDGE_STORE_DIR


def list_stores() -> list[dict]:
    """List all knowledge stores with metadata."""
    stores_dir = get_stores_dir()
    if not stores_dir.exists():
        return []

    stores = []
    for db_file in stores_dir.glob("*.db"):
        store_name = db_file.stem

        # Get finding count and date range
        try:
            with sqlite3.connect(db_file) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                count = cursor.fetchone()[0]

                cursor = conn.execute(
                    "SELECT MIN(created_at), MAX(created_at) FROM embeddings"
                )
                row = cursor.fetchone()
                min_ts, max_ts = row[0], row[1]

                created = datetime.fromtimestamp(min_ts).strftime("%Y-%m-%d %H:%M") if min_ts else "N/A"
                updated = datetime.fromtimestamp(max_ts).strftime("%Y-%m-%d %H:%M") if max_ts else "N/A"
        except Exception:
            count = 0
            created = "N/A"
            updated = "N/A"

        stores.append({
            "name": store_name,
            "finding_count": count,
            "created": created,
            "updated": updated,
        })

    # Sort by name
    stores.sort(key=lambda s: s["name"])
    return stores


def get_store_connection(store_name: str) -> sqlite3.Connection:
    """Get a connection to a store's SQLite database."""
    db_path = get_stores_dir() / f"{store_name}.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"Store '{store_name}' not found")
    return sqlite3.connect(db_path)


def get_findings(store_name: str, offset: int = 0, limit: int = 50) -> list[dict]:
    """Get findings from a store with pagination."""
    with get_store_connection(store_name) as conn:
        cursor = conn.execute(
            """
            SELECT id, doc_id, text, metadata, created_at
            FROM embeddings
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        findings = []
        for row in cursor.fetchall():
            metadata = json.loads(row[3]) if row[3] else {}
            findings.append({
                "row_id": row[0],
                "doc_id": row[1],
                "text": row[2],
                "text_preview": row[2][:200] + "..." if len(row[2]) > 200 else row[2],
                "metadata": metadata,
                "finding_type": metadata.get("finding_type", "unknown"),
                "source_url": metadata.get("source_url", ""),
                "title": metadata.get("title", "Untitled"),
                "created_at": datetime.fromtimestamp(row[4]).strftime("%Y-%m-%d %H:%M") if row[4] else "N/A",
            })

        return findings


def get_finding_count(store_name: str) -> int:
    """Get total finding count for a store."""
    with get_store_connection(store_name) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        return cursor.fetchone()[0]


def get_unique_sources(store_name: str) -> list[dict]:
    """Get unique sources from a store with finding counts."""
    with get_store_connection(store_name) as conn:
        cursor = conn.execute("SELECT metadata FROM embeddings")

        sources = {}  # url -> {title, count}
        for row in cursor.fetchall():
            metadata = json.loads(row[0]) if row[0] else {}
            url = metadata.get("source_url", "")
            title = metadata.get("title", "Untitled")

            if url:
                if url not in sources:
                    sources[url] = {"url": url, "title": title, "count": 0}
                sources[url]["count"] += 1

        # Sort by count descending
        source_list = list(sources.values())
        source_list.sort(key=lambda x: -x["count"])
        return source_list


def get_finding_by_id(store_name: str, doc_id: str) -> dict | None:
    """Get a single finding by doc_id."""
    with get_store_connection(store_name) as conn:
        cursor = conn.execute(
            "SELECT id, doc_id, text, metadata, created_at FROM embeddings WHERE doc_id = ?",
            (doc_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        metadata = json.loads(row[3]) if row[3] else {}
        return {
            "row_id": row[0],
            "doc_id": row[1],
            "text": row[2],
            "metadata": metadata,
            "finding_type": metadata.get("finding_type", "unknown"),
            "source_url": metadata.get("source_url", ""),
            "title": metadata.get("title", "Untitled"),
            "author": metadata.get("author"),
            "publication_date": metadata.get("publication_date"),
            "accessed_date": metadata.get("accessed_date"),
            "relevance_notes": metadata.get("relevance_notes", ""),
            "created_at": datetime.fromtimestamp(row[4]).strftime("%Y-%m-%d %H:%M") if row[4] else "N/A",
        }


def get_neighbors(store_name: str, doc_id: str, k: int = 10) -> list[dict]:
    """Get nearest neighbor findings for a given finding."""
    stores_dir = get_stores_dir()
    index_path = stores_dir / f"{store_name}.index"

    if not index_path.exists():
        return []

    # Get the row_id for this doc_id
    with get_store_connection(store_name) as conn:
        cursor = conn.execute(
            "SELECT id FROM embeddings WHERE doc_id = ?",
            (doc_id,),
        )
        row = cursor.fetchone()
        if not row:
            return []
        row_id = row[0]

        # Get total count for dimension detection
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        total_count = cursor.fetchone()[0]

    if total_count <= 1:
        return []

    # Load index and get the vector
    # We need to determine the dimension - read from config
    dimension = Config.EMBEDDING_DIM

    index = hnswlib.Index(space="cosine", dim=dimension)
    index.load_index(str(index_path))

    # Get the vector for this item
    try:
        vectors = index.get_items([row_id])
        if len(vectors) == 0:
            return []
        query_vector = vectors[0]
    except Exception:
        return []

    # Search for k+1 neighbors (includes self)
    search_k = min(k + 1, total_count)
    labels, distances = index.knn_query(np.array([query_vector], dtype=np.float32), k=search_k)

    # Get the findings, excluding self
    neighbor_findings = []
    with get_store_connection(store_name) as conn:
        for label, distance in zip(labels[0], distances[0]):
            if label == row_id:
                continue  # Skip self

            cursor = conn.execute(
                "SELECT doc_id, text, metadata, created_at FROM embeddings WHERE id = ?",
                (int(label),),
            )
            row = cursor.fetchone()
            if row:
                metadata = json.loads(row[2]) if row[2] else {}
                neighbor_findings.append({
                    "doc_id": row[0],
                    "text_preview": row[1][:150] + "..." if len(row[1]) > 150 else row[1],
                    "title": metadata.get("title", "Untitled"),
                    "finding_type": metadata.get("finding_type", "unknown"),
                    "distance": round(float(distance), 4),
                })

            if len(neighbor_findings) >= k:
                break

    # Sort by distance (closest/most similar first)
    neighbor_findings.sort(key=lambda x: -x["distance"])
    return neighbor_findings


def search_store(store_name: str, query: str, k: int = 10) -> list[dict]:
    """Semantic search across findings in a store."""
    stores_dir = get_stores_dir()
    index_path = stores_dir / f"{store_name}.index"

    if not index_path.exists():
        return []

    # Get total count
    with get_store_connection(store_name) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        total_count = cursor.fetchone()[0]

    if total_count == 0:
        return []

    # Embed the query
    embedder = get_embedder()
    query_vector = embedder.embed_single(query, is_query=True)

    # Load index and search
    dimension = Config.EMBEDDING_DIM
    index = hnswlib.Index(space="cosine", dim=dimension)
    index.load_index(str(index_path))

    search_k = min(k, total_count)
    labels, distances = index.knn_query(np.array([query_vector], dtype=np.float32), k=search_k)

    # Get the findings
    results = []
    with get_store_connection(store_name) as conn:
        for label, distance in zip(labels[0], distances[0]):
            cursor = conn.execute(
                "SELECT doc_id, text, metadata, created_at FROM embeddings WHERE id = ?",
                (int(label),),
            )
            row = cursor.fetchone()
            if row:
                metadata = json.loads(row[2]) if row[2] else {}
                results.append({
                    "doc_id": row[0],
                    "text_preview": row[1][:200] + "..." if len(row[1]) > 200 else row[1],
                    "title": metadata.get("title", "Untitled"),
                    "finding_type": metadata.get("finding_type", "unknown"),
                    "source_url": metadata.get("source_url", ""),
                    "distance": round(float(distance), 4),
                })

    # Sort by distance (most similar first)
    results.sort(key=lambda x: -x["distance"])
    return results


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page - list all stores."""
    stores = list_stores()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "stores": stores},
    )


@app.get("/stores/{store_name}", response_class=HTMLResponse)
async def store_detail(request: Request, store_name: str, page: int = 1):
    """Store detail page - list findings."""
    limit = 50
    offset = (page - 1) * limit

    findings = get_findings(store_name, offset=offset, limit=limit)
    total_count = get_finding_count(store_name)
    total_pages = (total_count + limit - 1) // limit
    sources = get_unique_sources(store_name)

    return templates.TemplateResponse(
        "store.html",
        {
            "request": request,
            "store_name": store_name,
            "findings": findings,
            "page": page,
            "total_pages": total_pages,
            "total_count": total_count,
            "sources": sources,
        },
    )


@app.get("/stores/{store_name}/findings/{doc_id}", response_class=HTMLResponse)
async def finding_detail(request: Request, store_name: str, doc_id: str):
    """Finding detail page."""
    finding = get_finding_by_id(store_name, doc_id)
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")

    return templates.TemplateResponse(
        "finding.html",
        {
            "request": request,
            "store_name": store_name,
            "finding": finding,
        },
    )


@app.get("/stores/{store_name}/findings/{doc_id}/neighbors", response_class=HTMLResponse)
async def finding_neighbors(request: Request, store_name: str, doc_id: str, k: int = 10):
    """Get neighbors partial (for htmx)."""
    neighbors = get_neighbors(store_name, doc_id, k=k)
    return templates.TemplateResponse(
        "partials/neighbors.html",
        {
            "request": request,
            "store_name": store_name,
            "neighbors": neighbors,
        },
    )


@app.get("/stores/{store_name}/search", response_class=HTMLResponse)
async def store_search(request: Request, store_name: str, q: str = "", k: int = 10):
    """Search findings in a store (htmx partial)."""
    if not q.strip():
        return HTMLResponse("<p><em>Enter a search query above.</em></p>")

    results = search_store(store_name, q, k=k)
    return templates.TemplateResponse(
        "partials/search_results.html",
        {
            "request": request,
            "store_name": store_name,
            "query": q,
            "results": results,
        },
    )


def main():
    """Run the server."""
    print("Starting Knowledge Store Explorer...")
    print(f"Knowledge stores directory: {get_stores_dir()}")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
