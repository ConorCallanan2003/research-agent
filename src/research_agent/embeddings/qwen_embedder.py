"""Qwen3-Embedding-4B model wrapper with singleton pattern and lazy loading."""

import threading
from typing import ClassVar

import torch
from sentence_transformers import SentenceTransformer

from research_agent.config import Config


class QwenEmbedder:
    """
    Singleton pattern for Qwen3-Embedding-4B model.
    Lazy-loads the model on first use to avoid startup delays.
    """

    _instance: ClassVar["QwenEmbedder | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "QwenEmbedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._model = None
                    instance._initialized = False
                    instance._device = None
                    cls._instance = instance
        return cls._instance

    def _load_model(self) -> None:
        """Load the embedding model (called lazily on first embed)."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"  # Apple Silicon
            else:
                self._device = "cpu"

            # Determine dtype based on device
            if self._device == "cpu":
                dtype = torch.float32
            else:
                dtype = torch.float16

            self._model = SentenceTransformer(
                Config.EMBEDDING_MODEL,
                device=self._device,
                model_kwargs={"torch_dtype": dtype},
                tokenizer_kwargs={"padding_side": "left"},
            )
            self._initialized = True

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            is_query: If True, use query prompt for better retrieval

        Returns:
            List of embedding vectors
        """
        self._load_model()

        if is_query:
            # Use query prompt for retrieval queries
            embeddings = self._model.encode(
                texts,
                prompt_name="query",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            # Use default for documents/passages
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        return embeddings.tolist()

    def embed_single(self, text: str, is_query: bool = False) -> list[float]:
        """Convenience method for single text embedding."""
        return self.embed([text], is_query=is_query)[0]

    def embed_batch(
        self, texts: list[str], batch_size: int = 32, is_query: bool = False
    ) -> list[list[float]]:
        """
        Embed texts in batches for memory efficiency.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            is_query: If True, use query prompt

        Returns:
            List of embedding vectors
        """
        self._load_model()

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embed(batch, is_query=is_query)
            all_embeddings.extend(embeddings)

        return all_embeddings

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return Config.EMBEDDING_DIM

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._initialized

    @property
    def device(self) -> str | None:
        """Return the device the model is loaded on."""
        return self._device


def preload_embeddings() -> None:
    """
    Optionally preload embeddings at startup in a background thread.
    Call this early in application startup for faster first embedding.
    """

    def _load() -> None:
        embedder = QwenEmbedder()
        embedder._load_model()

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
