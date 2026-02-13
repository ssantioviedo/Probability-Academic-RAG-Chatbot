"""
Sparse Retriever module using BM25.

Provides keyword-based search capabilities to complement
dense vector retrieval.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, TYPE_CHECKING

from rank_bm25 import BM25Okapi
from src.utils.logger import get_logger, LoggerMixin

if TYPE_CHECKING:
    from src.ingestion.chunker import TextChunk

class BM25Retriever(LoggerMixin):
    """
    Sparse retriever based on BM25Okapi.
    
    Attributes:
        index_path: Path to save/load the BM25 index.
        bm25: The BM25 object.
        chunks: List of stored chunks (needed to return results).
    """
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.bm25 = None
        self.chunks = []
        self.logger.info(f"BM25Retriever initialized with path: {index_path}")

    def index_documents(self, chunks: List['TextChunk']):
        """
        Build BM25 index from text chunks.
        
        Args:
            chunks: List of TextChunk objects.
        """
        self.logger.info(f"Building BM25 index for {len(chunks)} chunks...")
        
        # Simple tokenization: split by space and lowercase
        # In a production app, we might want a better tokenizer (e.g. NLTK/Spacy)
        tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.chunks = chunks
        
        self.save()
        self.logger.info("BM25 index built and saved.")

    def save(self):
        """Save the index and chunks to disk."""
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump({'bm25': self.bm25, 'chunks': self.chunks}, f)
        except Exception as e:
            self.logger.error(f"Failed to save BM25 index: {e}")

    def load(self) -> bool:
        """
        Load the index from disk.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.index_path.exists():
            self.logger.warning(f"BM25 index not found at {self.index_path}")
            return False
            
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.chunks = data['chunks']
            self.logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load BM25 index: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List['TextChunk']:
        """
        Search for documents using BM25.
        
        Args:
            query: Query string.
            top_k: Number of results to return.
            
        Returns:
            List of matching TextChunk objects.
        """
        if not self.bm25:
            self.logger.warning("BM25 index not loaded. Returning empty results.")
            return []
            
        tokenized_query = query.lower().split()
        # get_top_n returns the actual documents (chunks in our case)
        results = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        return results
