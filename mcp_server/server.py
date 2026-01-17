"""MCP Server for querying research knowledge stores."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp.server.fastmcp import FastMCP

from research_agent.config import Config
from research_agent.embeddings.qwen_embedder import QwenEmbedder
from research_agent.storage import Collection


# Create MCP server
mcp = FastMCP(
    name="research-knowledge",
    instructions="""This server provides access to research knowledge stores created by the Research Agent.

Use `list_knowledge_stores` to see available stores, then `query_knowledge_store` to search within a specific store.

Each store contains findings from a research session, with full citations and relevance information.""",
)

# Singleton embedder (lazy loaded)
_embedder: QwenEmbedder | None = None


def get_embedder() -> QwenEmbedder:
    """Get or create the singleton embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = QwenEmbedder()
    return _embedder


@mcp.tool()
def list_knowledge_stores() -> list[dict]:
    """
    List all available knowledge stores.

    Returns a list of stores with their names, paths, and finding counts.
    Use this to discover what research sessions are available to query.
    """
    stores = []
    store_dir = Config.KNOWLEDGE_STORE_DIR

    if not store_dir.exists():
        return []

    # Find all .db files (our custom storage format)
    for db_file in store_dir.glob("*.db"):
        store_name = db_file.stem

        try:
            # Initialize collection to get count
            collection = Collection(
                name=store_name,
                dimension=Config.EMBEDDING_DIM,
                path=store_dir,
            )
            count = collection.count()

            stores.append({
                "name": store_name,
                "path": str(db_file),
                "finding_count": count,
            })
        except Exception as e:
            stores.append({
                "name": store_name,
                "path": str(db_file),
                "error": str(e),
            })

    return stores


@mcp.tool()
def query_knowledge_store(
    store_name: str,
    query: str,
    k: int = 10,
) -> list[dict] | dict:
    """
    Semantic search within a specific knowledge store.

    Args:
        store_name: Name of the knowledge store (without .db extension).
                   Use list_knowledge_stores() to see available stores.
        query: Natural language search query. The search is semantic,
               so describe what you're looking for in plain language.
        k: Number of results to return (default: 10, max: 50)

    Returns:
        List of findings with text, citations, and relevance scores.
        Lower distance = more relevant.
    """
    store_dir = Config.KNOWLEDGE_STORE_DIR
    db_path = store_dir / f"{store_name}.db"

    if not db_path.exists():
        return {"error": f"Knowledge store '{store_name}' not found. Use list_knowledge_stores() to see available stores."}

    # Clamp k to reasonable range
    k = max(1, min(k, 50))

    try:
        # Get embedder and generate query embedding
        embedder = get_embedder()
        query_embedding = embedder.embed_single(query, is_query=True)

        # Initialize collection
        collection = Collection(
            name=store_name,
            dimension=embedder.dimension,
            path=store_dir,
        )

        # Search
        results = collection.search(query_embedding, k=k)

        # Format results
        formatted_results = []
        for result in results:
            metadata = result.metadata or {}
            formatted_results.append({
                "text": result.text,
                "source_url": metadata.get("source_url", ""),
                "title": metadata.get("title", ""),
                "author": metadata.get("author"),
                "publication_date": metadata.get("publication_date"),
                "accessed_date": metadata.get("accessed_date"),
                "relevance_notes": metadata.get("relevance_notes", ""),
                "distance": round(result.distance, 4),
                "relevance_score": round(1 - result.distance, 4),  # Higher = more relevant
            })

        return formatted_results

    except Exception as e:
        return {"error": f"Error querying store: {str(e)}"}


@mcp.tool()
def get_store_statistics(store_name: str) -> dict:
    """
    Get detailed statistics about a knowledge store.

    Args:
        store_name: Name of the knowledge store (without .db extension)

    Returns:
        Statistics including total findings, unique sources, and source URLs.
    """
    store_dir = Config.KNOWLEDGE_STORE_DIR
    db_path = store_dir / f"{store_name}.db"

    if not db_path.exists():
        return {"error": f"Knowledge store '{store_name}' not found."}

    try:
        collection = Collection(
            name=store_name,
            dimension=Config.EMBEDDING_DIM,
            path=store_dir,
        )

        count = collection.count()

        # Get all embeddings to extract unique sources
        source_urls = set()
        if count > 0:
            embeddings = collection.get_all(limit=count)
            for emb in embeddings:
                if emb.metadata and "source_url" in emb.metadata:
                    source_urls.add(emb.metadata["source_url"])

        return {
            "store_name": store_name,
            "total_findings": count,
            "unique_sources": len(source_urls),
            "source_urls": list(source_urls),
        }

    except Exception as e:
        return {"error": f"Error getting statistics: {str(e)}"}


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
