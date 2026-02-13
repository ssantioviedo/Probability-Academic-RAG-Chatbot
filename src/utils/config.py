"""
Configuration module for loading environment variables.

This module provides a centralized configuration class that loads
and validates all environment variables required by the application.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Config:
    """
    Configuration class for managing application settings.
    
    Loads environment variables from .env file and provides
    typed access to all configuration values.
    
    Attributes:
        google_api_key: Google Gemini API key.
        chroma_db_path: Path to ChromaDB persistence directory.
        data_dir: Path to raw data directory.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        embedding_model: Sentence transformers model name.
        llm_model: LLM model name (e.g., gemini-pro).
        llm_temperature: LLM temperature setting.
        llm_max_tokens: Maximum tokens for LLM response.
        chunk_size: Size of text chunks in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        default_top_k: Default number of documents to retrieve.
        min_similarity_threshold: Minimum similarity score threshold.
    
    Example:
        >>> config = Config()
        >>> print(config.chroma_db_path)
        ./chroma_db
    """
    
    def __init__(self, env_file: Optional[str] = None) -> None:
        """
        Initialize configuration by loading environment variables.
        
        Args:
            env_file: Optional path to .env file. If not provided,
                     searches for .env in project root.
        """
        # Find project root (where .env should be)
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env in current directory or parent directories
            current_dir = Path.cwd()
            env_path = current_dir / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            else:
                # Try parent directories
                for parent in current_dir.parents:
                    env_path = parent / ".env"
                    if env_path.exists():
                        load_dotenv(env_path)
                        break
                else:
                    # If no .env found, still try to load from environment
                    load_dotenv()
        
        # API Keys
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
        
        # Paths
        self.chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.data_dir: str = os.getenv("DATA_DIR", "./data/raw")
        
        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
        
        # Embedding model
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", 
            "paraphrase-multilingual-mpnet-base-v2"
        )
        
        # LLM settings
        self.llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        
        # Chunking settings (in tokens, converted to chars by chunker)
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "625"))  # 625 tokens = 2500 chars
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "125"))  # 125 tokens = 500 chars
        self.parent_chunk_size: int = int(os.getenv("PARENT_CHUNK_SIZE", "1250"))  # 1250 tokens = 5000 chars
        self.child_chunk_size: int = int(os.getenv("CHILD_CHUNK_SIZE", "625"))  # Same as chunk_size
        self.chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "semantic")
        
        # Retrieval settings
        self.default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
        self.min_similarity_threshold: float = float(
            os.getenv("MIN_SIMILARITY_THRESHOLD", "0.5")
        )

        # Rate Limiting
        self.enable_session_limit: bool = os.getenv("ENABLE_SESSION_LIMIT", "False").lower() == "true"
        self.session_limit: int = int(os.getenv("SESSION_LIMIT", "5"))
        self.session_window_minutes: int = int(os.getenv("SESSION_WINDOW_MINUTES", "15"))
        
        # Advanced Retrieval settings
        self.chroma_db_path_obj = Path(self.chroma_db_path)
        self.sparse_index_path: Path = Path(self.data_dir).parent / "processed" / "bm25_index.pkl"
        self.reranker_model_name: str = os.getenv(
            "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        # RRF (Reciprocal Rank Fusion) parameters
        self.rrf_k_param: int = int(os.getenv("RRF_K_PARAM", "60"))  # RRF constant
        self.dense_fetch_multiplier: int = int(os.getenv("DENSE_FETCH_MULTIPLIER", "3"))
        self.sparse_fetch_multiplier: int = int(os.getenv("SPARSE_FETCH_MULTIPLIER", "3"))
        self.enable_query_expansion: bool = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    
    def validate(self) -> list[str]:
        """
        Validate that all required configuration values are set.
        
        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors: list[str] = []
        
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is not set in environment variables")
        
        if not Path(self.data_dir).exists():
            errors.append(f"DATA_DIR path does not exist: {self.data_dir}")
        
        if self.llm_temperature < 0 or self.llm_temperature > 1:
            errors.append(
                f"LLM_TEMPERATURE must be between 0 and 1, got: {self.llm_temperature}"
            )
        
        if self.chunk_size < 100:
            errors.append(
                f"CHUNK_SIZE should be at least 100, got: {self.chunk_size}"
            )
        
        if self.chunk_overlap >= self.chunk_size:
            errors.append(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        
        return errors
    
    def get_chroma_path(self) -> Path:
        """
        Get ChromaDB path as Path object, creating directory if needed.
        
        Returns:
            Path object for ChromaDB directory.
        """
        path = Path(self.chroma_db_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_data_path(self) -> Path:
        """
        Get data directory path as Path object.
        
        Returns:
            Path object for data directory.
        """
        return Path(self.data_dir)
    
    def __repr__(self) -> str:
        """Return string representation of config (hiding sensitive values)."""
        return (
            f"Config("
            f"api_key={'*' * 8 if self.google_api_key else 'NOT SET'}, "
            f"chroma_db_path='{self.chroma_db_path}', "
            f"data_dir='{self.data_dir}', "
            f"embedding_model='{self.embedding_model}', "
            f"llm_model='{self.llm_model}')"
        )


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance.
    
    Returns:
        Config instance loaded from environment.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config
