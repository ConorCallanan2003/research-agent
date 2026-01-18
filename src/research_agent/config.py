"""Configuration and environment management."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""

    # API Keys
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    KNOWLEDGE_STORE_DIR: Path = Path(
        os.getenv("KNOWLEDGE_STORE_DIR", str(PROJECT_ROOT / "knowledge_stores"))
    )

    # Model configuration
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    CLAUDE_FAST_MODEL: str = os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5-20251001")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
    EMBEDDING_DIM: int = 2560  # Qwen3-Embedding-4B dimension

    # Browser configuration
    # Headless mode supported with playwright-stealth (Yahoo search + consent dialog handling)
    BROWSER_HEADLESS: bool = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
    PAGE_TIMEOUT_MS: int = int(os.getenv("PAGE_TIMEOUT_MS", "30000"))

    # Agent configuration
    MAX_AGENT_TURNS: int = int(os.getenv("MAX_AGENT_TURNS", "100"))
    MAX_CONTENT_LENGTH: int = int(
        os.getenv("MAX_CONTENT_LENGTH", "15000")
    )  # Max chars per page

    @classmethod
    def validate(cls) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is not set")

        if not cls.KNOWLEDGE_STORE_DIR.exists():
            try:
                cls.KNOWLEDGE_STORE_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create knowledge store directory: {e}")

        return errors

    @classmethod
    def ensure_valid(cls) -> None:
        """Validate configuration and raise if invalid."""
        errors = cls.validate()
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
