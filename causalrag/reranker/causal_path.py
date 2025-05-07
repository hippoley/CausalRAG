# reranker/causal_path.py
# Reranks candidate text passages based on their overlap with causal path nodes and paths

import re
from typing import List, Dict, Any, Tuple, Optional, Union, Set
import logging
from collections import Counter
import numpy as np
from .base import BaseReranker

logger = logging.getLogger(__name__)

class CausalPathReranker(BaseReranker):
    """
    Reranker that scores candidates based on overlap with causal paths
    from the causal graph that are relevant to the query.
    """
    
    def __init__(self, 
                retriever,  # CausalPathRetriever - avoiding circular import
                name: str = "causal_path",
                node_match_weight: float = 1.0,
                path_match_weight: float = 2.0,
                semantic_match_weight: float = 0.5,
                min_node_length: int = 3):
        """
        Initialize the causal path reranker
        
        Args:
            retriever: CausalPathRetriever instance
            name: Identifier for the reranker
            node_match_weight: Weight for individual node matches
            path_match_weight: Weight for path structure matches
            semantic_match_weight: Weight for semantic similarity
            min_node_length: Minimum node text length to consider (to avoid short, common words)
        """
        super().__init__(name=name)
        self.retriever = retriever
        self.node_match_weight = node_match_weight
        self.path_match_weight = path_match_weight
        self.semantic_match_weight = semantic_match_weight
        self.min_node_length = min_node_length
        self.last_query_nodes = []  # Cache for explanations
        self.last_query_paths = []  # Cache for explanations
    
    def rerank(self, 
              query: str, 
              candidates: List[str], 
              metadata: Optional[List[Dict[str, Any]]] = None) -> List[Tuple[str, float]]:
        """
        Rerank candidates based on causal path relevance
        
        Args:
            query: User query
            candidates: List of candidate text passages
            metadata: Optional metadata for each candidate
            
        Returns:
            List of (passage, score) tuples sorted by score (descending)
        """
        if not candidates:
            return []
        
        # Get relevant causal nodes and paths
        try:
            path_nodes = self.retriever.retrieve_path_nodes(query)
            causal_paths = self.retriever.retrieve_paths(query, max_paths=3)
            
            # Cache for explanations
            self.last_query_nodes = path_nodes
            self.last_query_paths = causal_paths
            
            # If no causal information is found, return original order with minimal scores
            if not path_nodes:
                logger.warning(f"No causal nodes found for query: {query}")
                return [(p, 0.1) for p in candidates]
                
        except Exception as e:
            logger.error(f"Error retrieving causal information: {e}")
            return [(p, 0.1) for p in candidates]
        
        # Score candidates
        scored_candidates = []
        
        for i, passage in enumerate(candidates):
            # 1. Calculate node overlap score
            node_score = self._calculate_node_overlap(passage, path_nodes)
            
            # 2. Calculate path structure score
            path_score = self._calculate_path_structure(passage, causal_paths)
            
            # 3. Calculate semantic match (if metadata available)
            semantic_score = 0.0
            if metadata and i < len(metadata) and 'score' in metadata[i]:
                semantic_score = metadata[i]['score']
            
            # Combine scores
            final_score = (
                node_score * self.node_match_weight + 
                path_score * self.path_match_weight + 
                semantic_score * self.semantic_match_weight
            )
            
            scored_candidates.append((passage, final_score))
        
        # Normalize scores to [0,1] range
        if scored_candidates:
            scores = [score for _, score in scored_candidates]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max(max_score - min_score, 1e-5)  # Avoid division by zero
            
            normalized_candidates = []
            for passage, score in scored_candidates:
                normalized_score = (score - min_score) / score_range
                normalized_candidates.append((passage, normalized_score))
                
            scored_candidates = normalized_candidates
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def _calculate_node_overlap(self, passage: str, nodes: List[str]) -> float:
        """
        Calculate how many causal nodes appear in the passage
        
        Args:
            passage: Text passage
            nodes: List of causal node texts
            
        Returns:
            Overlap score
        """
        if not nodes:
            return 0.0
            
        # Prepare passage for matching
        passage_lower = passage.lower()
        
        # Count matches
        match_count = 0
        for node in nodes:
            if len(node) >= self.min_node_length and node.lower() in passage_lower:
                match_count += 1
        
        # Normalize by total nodes
        return match_count / len(nodes)
    
    def _calculate_path_structure(self, passage: str, paths: List[List[str]]) -> float:
        """
        Calculate how well the passage preserves causal path structures
        
        Args:
            passage: Text passage
            paths: List of causal paths (each path is a list of node texts)
            
        Returns:
            Path structure preservation score
        """
        if not paths:
            return 0.0
            
        passage_lower = passage.lower()
        path_scores = []
        
        for path in paths:
            if len(path) < 2:
                continue
                
            # Check for sequential pairs of concepts in the passage
            pair_matches = 0
            total_pairs = len(path) - 1
            
            for i in range(total_pairs):
                cause = path[i].lower()
                effect = path[i+1].lower()
                
                # Skip if either node is too short
                if len(cause) < self.min_node_length or len(effect) < self.min_node_length:
                    continue
                
                # Check if both nodes appear in passage
                if cause in passage_lower and effect in passage_lower:
                    # Check if they appear in the correct order
                    cause_pos = passage_lower.find(cause)
                    effect_pos = passage_lower.find(effect)
                    
                    if cause_pos < effect_pos:
                        # Bonus for correct order
                        pair_matches += 1.5
                    else:
                        # Partial credit for having both concepts
                        pair_matches += 0.5
            
            if total_pairs > 0:
                path_scores.append(pair_matches / total_pairs)
        
        # Return average path score or 0 if no valid paths
        return sum(path_scores) / len(path_scores) if path_scores else 0.0
    
    def get_explanation(self, 
                       query: str, 
                       candidate: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Explain why a passage received its ranking
        
        Args:
            query: Original query
            candidate: Text passage
            metadata: Optional metadata
            
        Returns:
            Human-readable explanation
        """
        # Calculate the node overlap
        node_matches = []
        for node in self.last_query_nodes:
            if len(node) >= self.min_node_length and node.lower() in candidate.lower():
                node_matches.append(node)
        
        # Check for path structure preservation
        path_matches = []
        for path in self.last_query_paths:
            if len(path) < 2:
                continue
                
            for i in range(len(path) - 1):
                cause = path[i].lower()
                effect = path[i+1].lower()
                
                if (len(cause) >= self.min_node_length and 
                    len(effect) >= self.min_node_length and
                    cause in candidate.lower() and 
                    effect in candidate.lower()):
                    
                    path_matches.append(f"{cause} → {effect}")
        
        # Build explanation
        explanation = [f"Causal ranking explanation for passage:"]
        
        if node_matches:
            explanation.append(f"\nMatched concepts ({len(node_matches)}/{len(self.last_query_nodes)}):")
            for node in node_matches[:5]:  # Limit to 5 for readability
                explanation.append(f"- {node}")
            if len(node_matches) > 5:
                explanation.append(f"- ... and {len(node_matches) - 5} more")
        else:
            explanation.append("\nNo direct concept matches found.")
        
        if path_matches:
            explanation.append(f"\nPreserved causal relationships:")
            for path in path_matches[:3]:  # Limit to 3 for readability
                explanation.append(f"- {path}")
            if len(path_matches) > 3:
                explanation.append(f"- ... and {len(path_matches) - 3} more")
        else:
            explanation.append("\nNo causal relationships preserved.")
        
        return "\n".join(explanation)