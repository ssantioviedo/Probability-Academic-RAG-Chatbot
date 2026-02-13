"""
Confidence Calculator module for response quality assessment.

Calculates confidence scores based on retrieval similarity
metrics to indicate how reliable the generated response is.
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger, LoggerMixin


class ConfidenceLevel(Enum):
    """Enumeration of confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass
class ConfidenceResult:
    """
    Result of confidence calculation.
    
    Attributes:
        level: Confidence level (HIGH, MEDIUM, LOW, INSUFFICIENT).
        score: Numerical confidence score (0-1).
        max_similarity: Maximum similarity among retrieved documents.
        avg_similarity: Average similarity of top documents.
        message: Human-readable confidence message.
        should_respond: Whether the system should generate a response.
    """
    level: ConfidenceLevel
    score: float
    max_similarity: float
    avg_similarity: float
    message: str
    should_respond: bool = True
    
    @property
    def emoji(self) -> str:
        """Get emoji indicator for confidence level."""
        emoji_map = {
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🔴",
            ConfidenceLevel.INSUFFICIENT: "⚫",
        }
        return emoji_map.get(self.level, "⚪")
    
    @property
    def color(self) -> str:
        """Get color code for UI display."""
        color_map = {
            ConfidenceLevel.HIGH: "#28a745",      # Green
            ConfidenceLevel.MEDIUM: "#ffc107",    # Yellow
            ConfidenceLevel.LOW: "#dc3545",       # Red
            ConfidenceLevel.INSUFFICIENT: "#6c757d",  # Gray
        }
        return color_map.get(self.level, "#6c757d")
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "score": self.score,
            "max_similarity": self.max_similarity,
            "avg_similarity": self.avg_similarity,
            "message": self.message,
            "should_respond": self.should_respond,
            "emoji": self.emoji,
        }


# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
MIN_RESPONSE_THRESHOLD = 0.50
TOP_N_FOR_AVERAGE = 3


class ConfidenceCalculator(LoggerMixin):
    """
    Calculates confidence scores for retrieval results.
    
    Uses similarity metrics from retrieved documents to
    assess how confident the system should be in its response.
    
    Attributes:
        high_threshold: Threshold for HIGH confidence.
        medium_threshold: Threshold for MEDIUM confidence.
        min_threshold: Minimum threshold to generate a response.
        top_n: Number of top documents to use for average.
    
    Example:
        >>> calculator = ConfidenceCalculator()
        >>> similarities = [0.85, 0.72, 0.68, 0.55, 0.42]
        >>> result = calculator.calculate(similarities)
        >>> print(f"{result.emoji} {result.level.value}: {result.score:.2f}")
    """
    
    def __init__(
        self,
        high_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
        medium_threshold: float = MEDIUM_CONFIDENCE_THRESHOLD,
        min_threshold: float = MIN_RESPONSE_THRESHOLD,
        top_n: int = TOP_N_FOR_AVERAGE
    ) -> None:
        """
        Initialize the confidence calculator.
        
        Args:
            high_threshold: Threshold for HIGH confidence.
            medium_threshold: Threshold for MEDIUM confidence.
            min_threshold: Minimum threshold to respond.
            top_n: Number of top docs for average calculation.
        """
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.min_threshold = min_threshold
        self.top_n = top_n
        
        self.logger.debug(
            f"ConfidenceCalculator initialized: "
            f"high={high_threshold}, medium={medium_threshold}, "
            f"min={min_threshold}"
        )
    
    def calculate(
        self,
        similarities: list[float]
    ) -> ConfidenceResult:
        """
        Calculate confidence from similarity scores.
        
        Args:
            similarities: List of similarity scores from retrieval.
        
        Returns:
            ConfidenceResult with level, score, and recommendations.
        """
        if not similarities:
            self.logger.warning("No similarities provided for confidence calculation")
            return ConfidenceResult(
                level=ConfidenceLevel.INSUFFICIENT,
                score=0.0,
                max_similarity=0.0,
                avg_similarity=0.0,
                message="No relevant documents found in the bibliography.",
                should_respond=False
            )
        
        # Calculate metrics
        max_sim = max(similarities)
        sorted_sims = sorted(similarities, reverse=True)
        top_n_sims = sorted_sims[:self.top_n]
        avg_top_n = sum(top_n_sims) / len(top_n_sims)
        
        self.logger.debug(
            f"Confidence metrics: max={max_sim:.3f}, avg_top{self.top_n}={avg_top_n:.3f}"
        )
        
        # Determine confidence level
        if max_sim < self.min_threshold:
            return ConfidenceResult(
                level=ConfidenceLevel.INSUFFICIENT,
                score=max_sim,
                max_similarity=max_sim,
                avg_similarity=avg_top_n,
                message=(
                    "The retrieved documents don't seem relevant enough. "
                    "Try rephrasing your question or being more specific."
                ),
                should_respond=False
            )
        
        if max_sim > self.high_threshold and avg_top_n > (self.medium_threshold + 0.05):
            return ConfidenceResult(
                level=ConfidenceLevel.HIGH,
                score=max_sim,
                max_similarity=max_sim,
                avg_similarity=avg_top_n,
                message=(
                    "High confidence: Found highly relevant content in the bibliography."
                ),
                should_respond=True
            )
        
        if max_sim > self.medium_threshold:
            return ConfidenceResult(
                level=ConfidenceLevel.MEDIUM,
                score=max_sim,
                max_similarity=max_sim,
                avg_similarity=avg_top_n,
                message=(
                    "Medium confidence: Found relevant content, "
                    "but you may want to verify with additional sources."
                ),
                should_respond=True
            )
        
        return ConfidenceResult(
            level=ConfidenceLevel.LOW,
            score=max_sim,
            max_similarity=max_sim,
            avg_similarity=avg_top_n,
            message=(
                "Low confidence: The bibliography has limited information on this topic. "
                "Consider consulting additional resources."
            ),
            should_respond=True
        )
    
    def should_respond(self, similarities: list[float]) -> bool:
        """
        Quick check if system should generate a response.
        
        Args:
            similarities: List of similarity scores.
        
        Returns:
            True if response should be generated.
        """
        if not similarities:
            return False
        return max(similarities) >= self.min_threshold
    


