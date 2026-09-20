# reranker/base.py
# Abstract base class for reranking strategies

from typing import List, Dict, Any, Tuple, Optional, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseReranker(ABC):
    """
    Abstract base class for any reranking strategy.
    All rerankers should inherit from this class.
    """

    def __init__(self, name: str = "base"):
        """
        Initialize the reranker
        
        Args:
            name: Identifier for the reranker
        """
        self.name = name
        logger.debug(f"Initialized {name} reranker")
    
    @abstractmethod
    def rerank(self, 
              query: str, 
              candidates: List[str], 
              metadata: Optional[List[Dict[str, Any]]] = None) -> List[Union[str, Tuple[str, float]]]:
        """
        Rerank a list of candidate texts based on the query.

        Args:
            query: User input question or goal
            candidates: Retrieved text passages
            metadata: Optional metadata for each candidate

        Returns:
            Reranked list (most relevant first), optionally with scores
        """
        pass
    
    def rerank_with_scores(self, 
                          query: str, 
                          candidates: List[str], 
                          metadata: Optional[List[Dict[str, Any]]] = None) -> List[Tuple[str, float]]:
        """
        Rerank candidates and return with scores

        Args:
            query: User input question or goal
            candidates: Retrieved text passages
            metadata: Optional metadata for each candidate

        Returns:
            List of (passage, score) tuples, sorted by score (descending)
        """
        results = self.rerank(query, candidates, metadata)
        
        # Handle case where rerank() already returns scores
        if results and isinstance(results[0], tuple):
            return results
            
        # Default scoring - linearly decreasing
        return [(passage, 1.0 - (i / len(results))) for i, passage in enumerate(results)]
    
    def get_explanation(self, 
                       query: str, 
                       candidate: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Get explanation for why a candidate was ranked as it was

        Args:
            query: Original query
            candidate: The text passage
            metadata: Optional metadata for the candidate

        Returns:
            Human-readable explanation string
        """
        return f"No detailed explanation available for {self.name} reranker."