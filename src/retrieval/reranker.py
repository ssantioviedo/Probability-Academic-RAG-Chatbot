"""
Cross-Encoder Reranker module.

Re-scores retrieved documents using a Cross-Encoder model
to improve relevance.
"""

import math
from typing import List, Tuple

from sentence_transformers import CrossEncoder
from src.utils.logger import get_logger, LoggerMixin
from src.retrieval.models import RetrievedDocument

class Reranker(LoggerMixin):
    """
    Reranks document candidates using a Cross-Encoder.
    
    Attributes:
        model_name: Name of the Cross-Encoder model.
        model: The CrossEncoder model instance.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self.logger.info(f"Initializing Reranker with model: {model_name}")
        
    def load_model(self):
        """Lazy load the model to avoid overhead if not used."""
        if self.model is None:
            self.logger.info("Loading Cross-Encoder model...")
            self.model = CrossEncoder(self.model_name)
            self.logger.info("Cross-Encoder model loaded.")

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Apply sigmoid to convert logit to probability (0-1)."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def rerank(
        self, 
        query: str, 
        documents: List[RetrievedDocument], 
        top_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Rerank a list of retrieved documents.
        
        Args:
            query: The user query.
            documents: List of RetrievedDocument candidates.
            top_k: Number of top documents to return.
            
        Returns:
            List of reranked RetrievedDocument objects.
        """
        if not documents:
            return []
            
        self.load_model()
        
        # Prepare pairs for cross-encoder
        # We look at the actual text content (or parent content if that's what we want to judge)
        # Using context_text gives the model the full context to judge relevance
        pairs = [[query, doc.context_text] for doc in documents]
        
        try:
            # Predict scores
            scores = self.model.predict(pairs)
            
            # Update document scores (normalize logits to 0-1 via sigmoid)
            for doc, score in zip(documents, scores):
                doc.similarity = self._sigmoid(float(score))
                doc.distance = 0.0  # Clear distance as it's no longer cosine distance
                
            # Sort by score (descending)
            ranked_docs = sorted(documents, key=lambda x: x.similarity, reverse=True)
            
            return ranked_docs[:top_k]
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            # Fallback: return original top_k
            return documents[:top_k]
