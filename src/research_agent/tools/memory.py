"""Vector-based memory store for research findings."""

import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from research_agent.config import Config
from research_agent.embeddings import QwenEmbedder
from research_agent.models.findings import Citation, Finding, FindingType
from research_agent.storage import Collection, Embedding


class MemoryTool:
    """Vector-based memory store for research findings."""

    def __init__(
        self,
        store_name: str,
        embedder: QwenEmbedder | None = None,
        storage_dir: str | Path | None = None,
    ):
        self._store_name = store_name
        self._embedder = embedder or QwenEmbedder()
        self._storage_dir = Path(storage_dir or Config.KNOWLEDGE_STORE_DIR)
        self._collection: Collection | None = None
        self._source_urls: set[str] = set()

    @property
    def store_path(self) -> Path:
        """Get the full path to the knowledge store."""
        return self._storage_dir / self._store_name

    def initialize(self) -> None:
        """Initialize or load the vector store."""
        # Ensure storage directory exists
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize collection
        self._collection = Collection(
            name=self._store_name,
            dimension=self._embedder.dimension,
            path=self._storage_dir,
        )

        # Load existing source URLs
        self._load_source_urls()

    def _load_source_urls(self) -> None:
        """Load source URLs from existing findings."""
        if self._collection is None:
            return

        try:
            count = self._collection.count()
            if count > 0:
                embeddings = self._collection.get_all(limit=count)
                for emb in embeddings:
                    if emb.metadata and "source_url" in emb.metadata:
                        self._source_urls.add(emb.metadata["source_url"])
        except Exception:
            pass

    def store_finding(self, finding: Finding) -> str:
        """
        Store a research finding with its embedding.

        Args:
            finding: The Finding object to store

        Returns:
            The document ID
        """
        if self._collection is None:
            self.initialize()

        # Generate embedding
        embedding_vector = self._embedder.embed_single(finding.text)

        # Generate document ID
        doc_id = str(uuid4())

        # Prepare metadata
        metadata = finding.to_storage_dict()

        # Create and store embedding
        embedding = Embedding(
            vector=embedding_vector,
            text=finding.text,
            doc_id=doc_id,
            metadata=metadata,
        )

        self._collection.add(embedding)

        # Track source URL
        self._source_urls.add(finding.citation.source_url)

        return doc_id

    def store_finding_from_dict(
        self,
        text: str,
        source_url: str,
        title: str,
        relevance_notes: str,
        finding_type: FindingType = "paraphrase",
        author: str | None = None,
        publication_date: str | None = None,
    ) -> str:
        """
        Store a finding from individual fields (for tool interface).

        Args:
            text: The finding text
            source_url: Source URL
            title: Page/article title
            relevance_notes: Why this finding is relevant
            finding_type: Type of finding (direct_quote, paraphrase, summary, synthesis)
            author: Author if known
            publication_date: Publication date if known

        Returns:
            The document ID
        """
        finding = Finding(
            text=text,
            citation=Citation(
                source_url=source_url,
                title=title,
                author=author,
                publication_date=publication_date,
            ),
            relevance_notes=relevance_notes,
            finding_type=finding_type,
        )
        return self.store_finding(finding)

    def search_findings(
        self, query: str, k: int = 10
    ) -> list[tuple[Finding, float]]:
        """
        Semantic search for relevant findings.

        Args:
            query: Semantic search query
            k: Number of results to return

        Returns:
            List of (Finding, distance) tuples. Lower distance = more similar.
        """
        if self._collection is None:
            self.initialize()

        # Generate query embedding
        query_vector = self._embedder.embed_single(query, is_query=True)

        # Search
        results = self._collection.search(query_vector, k=k)

        # Convert to Findings
        findings = []
        for result in results:
            finding = Finding.from_storage(
                text=result.text,
                metadata=result.metadata or {},
                doc_id=result.doc_id,
            )
            findings.append((finding, result.distance))

        return findings

    def get_all_findings(self) -> list[Finding]:
        """Retrieve all stored findings for final document generation."""
        if self._collection is None:
            self.initialize()

        count = self._collection.count()
        if count == 0:
            return []

        embeddings = self._collection.get_all(limit=count)

        findings = []
        for emb in embeddings:
            finding = Finding.from_storage(
                text=emb.text,
                metadata=emb.metadata or {},
                doc_id=emb.doc_id,
            )
            findings.append(finding)

        return findings

    def get_statistics(self) -> dict:
        """Get memory store statistics."""
        if self._collection is None:
            self.initialize()

        count = self._collection.count()

        return {
            "total_findings": count,
            "unique_sources": len(self._source_urls),
            "source_urls": list(self._source_urls),
            "store_name": self._store_name,
            "store_path": str(self.store_path),
        }

    def has_source(self, url: str) -> bool:
        """Check if a source URL has already been processed."""
        # Normalize URL for comparison
        normalized = url.rstrip("/").lower()
        for existing in self._source_urls:
            if existing.rstrip("/").lower() == normalized:
                return True
        return False


def generate_store_name(topic: str, detail_level: str) -> str:
    """
    Generate a knowledge store name from research brief.

    Args:
        topic: Research topic
        detail_level: Detail level (overview, moderate, comprehensive)

    Returns:
        Store name like "quantum_computing_20250117_143052_comprehensive"
    """
    # Slugify topic
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())
    slug = slug.strip("_")[:50]  # Limit length

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{slug}_{timestamp}_{detail_level}"
