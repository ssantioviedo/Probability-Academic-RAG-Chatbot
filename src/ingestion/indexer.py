"""
ChromaDB Indexer module for creating and managing vector embeddings.

Handles embedding generation with sentence-transformers and
persistence to ChromaDB for semantic search capabilities.
"""

import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm

from src.utils.logger import get_logger, LoggerMixin, log_processing_stats
from src.ingestion.chunker import TextChunk
from src.retrieval.sparse import BM25Retriever


# Collection configuration
DEFAULT_COLLECTION_NAME = "probability_bibliography"
DEFAULT_BATCH_SIZE = 32


class ChromaIndexer(LoggerMixin):
    """
    Manages ChromaDB vector database for document embeddings.
    
    Handles embedding generation using sentence-transformers,
    batch processing, and persistence to disk.
    
    Attributes:
        persist_directory: Path to ChromaDB storage.
        collection_name: Name of the ChromaDB collection.
        embedding_model: Name of the sentence-transformers model.
        batch_size: Number of documents to embed per batch.
        client: ChromaDB client instance.
        collection: ChromaDB collection instance.
    
    Example:
        >>> indexer = ChromaIndexer(
        ...     persist_directory=Path("./chroma_db"),
        ...     embedding_model="paraphrase-multilingual-mpnet-base-v2"
        ... )
        >>> indexer.add_chunks(chunks)
        >>> indexer.get_statistics()
    """
    
    def __init__(
        self,
        persist_directory: Path,
        embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> None:
        """
        Initialize the ChromaDB indexer.
        
        Args:
            persist_directory: Directory for ChromaDB persistence.
            embedding_model: Sentence-transformers model name.
            collection_name: Name for the ChromaDB collection.
            batch_size: Batch size for embedding generation.
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            f"Initializing ChromaIndexer: model={embedding_model}, "
            f"collection={collection_name}"
        )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Academic bibliography for Probability & Statistics",
                "embedding_model": self.embedding_model,
                "created_at": datetime.now().isoformat(),
            }
        )
        
        self.logger.info(
            f"ChromaDB initialized: {self.collection.count()} existing documents"
        )
    
    def reset_collection(self) -> None:
        """
        Delete and recreate the collection.
        
        Use with caution - this deletes all indexed data.
        """
        self.logger.warning(f"Resetting collection: {self.collection_name}")
        
        # Delete existing collection
        self.client.delete_collection(self.collection_name)
        
        # Recreate collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Academic bibliography for Probability & Statistics",
                "embedding_model": self.embedding_model,
                "created_at": datetime.now().isoformat(),
            }
        )
        
        self.logger.info("Collection reset complete")
    
    def _prepare_metadata(self, metadata: dict) -> dict:
        """
        Prepare metadata for ChromaDB storage.
        
        ChromaDB requires metadata values to be str, int, float, or bool.
        
        Args:
            metadata: Raw metadata dictionary.
        
        Returns:
            Cleaned metadata dictionary.
        """
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned
    
    def add_chunks(
        self,
        chunks: list[TextChunk],
        show_progress: bool = True,
        build_sparse_index: bool = False,
        sparse_index_path: Optional[Path] = None
    ) -> dict:
        """
        Add text chunks to the ChromaDB collection.
        
        Generates embeddings in batches and stores them with metadata.
        Optionally builds and saves a BM25 index.
        
        Args:
            chunks: List of TextChunk objects to index.
            show_progress: Whether to show a progress bar.
            build_sparse_index: Whether to build and save BM25 index.
            sparse_index_path: Path to save BM25 index.
        
        Returns:
            Dictionary with indexing statistics.
        """
        if not chunks:
            self.logger.warning("No chunks to add")
            return {"added": 0, "skipped": 0, "errors": 0}
            
        # Build BM25 index if requested (before filtering for duplicates to ensure full corpus)
        if build_sparse_index and sparse_index_path:
            try:
                self.logger.info("Building BM25 index...")
                bm25_retriever = BM25Retriever(sparse_index_path)
                bm25_retriever.index_documents(chunks)
            except Exception as e:
                self.logger.error(f"Failed to build BM25 index: {e}")
        
        stats = {"added": 0, "skipped": 0, "errors": 0}
        start_time = time.time()
        
        # Get existing IDs to avoid duplicates
        existing_ids = set()
        try:
            # Get all existing IDs
            existing_data = self.collection.get()
            existing_ids = set(existing_data.get("ids", []))
        except Exception as e:
            self.logger.warning(f"Could not fetch existing IDs: {e}")
        
        # Filter out duplicates
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        skipped = len(chunks) - len(new_chunks)
        stats["skipped"] = skipped
        
        if skipped > 0:
            self.logger.info(f"Skipping {skipped} already indexed chunks")
        
        if not new_chunks:
            self.logger.info("All chunks already indexed")
            return stats
        
        # Process in batches
        batches = [
            new_chunks[i:i + self.batch_size]
            for i in range(0, len(new_chunks), self.batch_size)
        ]
        
        progress = tqdm(
            batches,
            desc="Indexing chunks",
            disable=not show_progress,
            unit="batch"
        )
        
        for batch in progress:
            try:
                # Prepare batch data
                ids = [chunk.chunk_id for chunk in batch]
                documents = [chunk.text for chunk in batch]
                metadatas = [
                    self._prepare_metadata(chunk.metadata) 
                    for chunk in batch
                ]
                
                # Add to collection (embeddings generated automatically)
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                
                stats["added"] += len(batch)
                
            except Exception as e:
                self.logger.error(f"Error indexing batch: {e}")
                stats["errors"] += len(batch)
        
        duration = time.time() - start_time
        
        log_processing_stats(
            self.logger,
            "Chunk indexing",
            stats["added"],
            duration,
            {"skipped": stats["skipped"], "errors": stats["errors"]}
        )
        
        return stats
    
    def get_existing_files(self) -> set[str]:
        """
        Get set of source files already in the index.
        
        Returns:
            Set of source filenames.
        """
        try:
            # Query for all unique source files
            results = self.collection.get(include=["metadatas"])
            
            if results and results.get("metadatas"):
                files = set()
                for metadata in results["metadatas"]:
                    if metadata and "source_file" in metadata:
                        files.add(metadata["source_file"])
                return files
            
            return set()
            
        except Exception as e:
            self.logger.error(f"Error getting existing files: {e}")
            return set()
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the indexed collection.
        
        Returns:
            Dictionary with collection statistics.
        """
        try:
            count = self.collection.count()
            
            # Get metadata breakdown
            results = self.collection.get(include=["metadatas"])
            
            stats = {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "persist_directory": str(self.persist_directory),
                "by_source_type": {},
                "by_author": {},
                "unique_files": set(),
                "languages": {},
            }
            
            if results and results.get("metadatas"):
                for metadata in results["metadatas"]:
                    if not metadata:
                        continue
                    
                    # Source type breakdown
                    source_type = metadata.get("source_type", "unknown")
                    stats["by_source_type"][source_type] = \
                        stats["by_source_type"].get(source_type, 0) + 1
                    
                    # Author breakdown
                    author = metadata.get("author", "unknown") or "unknown"
                    stats["by_author"][author] = \
                        stats["by_author"].get(author, 0) + 1
                    
                    # Unique files
                    source_file = metadata.get("source_file", "")
                    if source_file:
                        stats["unique_files"].add(source_file)
                    
                    # Language breakdown
                    language = metadata.get("language", "unknown")
                    stats["languages"][language] = \
                        stats["languages"].get(language, 0) + 1
            
            stats["total_files"] = len(stats["unique_files"])
            stats["unique_files"] = list(stats["unique_files"])
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def delete_by_file(self, source_file: str) -> int:
        """
        Delete all chunks from a specific source file.
        
        Args:
            source_file: Name of the source file to delete.
        
        Returns:
            Number of chunks deleted.
        """
        try:
            # Get IDs of chunks from this file
            results = self.collection.get(
                where={"source_file": source_file},
                include=["metadatas"]
            )
            
            if results and results.get("ids"):
                ids_to_delete = results["ids"]
                self.collection.delete(ids=ids_to_delete)
                self.logger.info(
                    f"Deleted {len(ids_to_delete)} chunks from {source_file}"
                )
                return len(ids_to_delete)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error deleting chunks: {e}")
            return 0
    
    def get_unique_authors(self) -> list[str]:
        """
        Get list of unique authors in the collection.
        
        Returns:
            List of author names.
        """
        stats = self.get_statistics()
        return list(stats.get("by_author", {}).keys())
    
    def get_last_indexed_date(self) -> Optional[str]:
        """
        Get the date when the collection was last indexed.
        
        Returns:
            ISO format date string or None.
        """
        try:
            collection_metadata = self.collection.metadata
            return collection_metadata.get("created_at")
        except Exception:
            return None


def create_indexer_from_config(config) -> ChromaIndexer:
    """
    Create a ChromaIndexer from application config.
    
    Args:
        config: Config object with ChromaDB settings.
    
    Returns:
        Configured ChromaIndexer instance.
    """
    return ChromaIndexer(
        persist_directory=Path(config.chroma_db_path),
        embedding_model=config.embedding_model,
        collection_name=DEFAULT_COLLECTION_NAME,
        batch_size=DEFAULT_BATCH_SIZE
    )
