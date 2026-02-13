"""
Data models for the retrieval module.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievedDocument:
    """
    Represents a document retrieved from the vector database.
    
    Attributes:
        text: Document text content.
        metadata: Document metadata.
        similarity: Cosine similarity score (0-1).
        distance: Distance metric from query.
        chunk_id: Unique identifier for the chunk.
        parent_content: Content of the parent document (if available).
    """
    text: str
    metadata: dict = field(default_factory=dict)
    similarity: float = 0.0
    distance: float = 0.0
    chunk_id: str = ""
    parent_content: Optional[str] = None
    
    @property
    def context_text(self) -> str:
        """Get the text to be used for context (parent if available)."""
        return self.parent_content if self.parent_content else self.text
    
    @property
    def source_file(self) -> str:
        """Get source filename."""
        return self.metadata.get("source_file", "Unknown")
    
    @property
    def page(self) -> int:
        """Get page number."""
        return self.metadata.get("page", 0)
    
    @property
    def author(self) -> str:
        """Get author name."""
        return self.metadata.get("author", "Unknown")
    
    @property
    def source_type(self) -> str:
        """Get source type."""
        return self.metadata.get("source_type", "unknown")
    
    @property
    def chapter(self) -> Optional[int]:
        """Get chapter number."""
        chapter = self.metadata.get("chapter", -1)
        return chapter if chapter != -1 else None
    
    def get_citation(self) -> str:
        """
        Generate a citation string for this document.
        
        Returns:
            Formatted citation string.
        """
        parts = []
        
        if self.author and self.author.lower() != "unknown":
            parts.append(self.author)
        
        if self.chapter:
            parts.append(f"Chapter {self.chapter}")
        
        if self.page:
            parts.append(f"page {self.page}")
        
        return ", ".join(parts) if parts else "Unknown source"
    
    def get_snippet(self, max_length: int = 200) -> str:
        """
        Get a preview snippet of the text.
        
        Args:
            max_length: Maximum snippet length.
        
        Returns:
            Truncated text snippet.
        """
        if len(self.text) <= max_length:
            return self.text
        
        return self.text[:max_length].rsplit(" ", 1)[0] + "..."
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "similarity": self.similarity,
            "chunk_id": self.chunk_id,
            "citation": self.get_citation(),
        }
