# retriever/hybrid.py
# Combines vector-based retrieval with causal path constraints for more accurate retrieval

from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import logging
from collections import Counter
import numpy as np
import heapq

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retriever that combines vector-based semantic search with causal path constraints.
    This creates a more focused retrieval by prioritizing passages that contain concepts
    from relevant causal paths.
    """
    
    def __init__(self, 
                vector_retriever,  # VectorStoreRetriever
                graph_retriever,   # CausalPathRetriever
                semantic_weight: float = 0.4,
                causal_weight: float = 0.6,
                reranking_factor: int = 2,
                min_causal_matches: int = 1,
                cache_results: bool = True):
        """
        Initialize hybrid retriever
        
        Args:
            vector_retriever: Vector-based retriever instance
            graph_retriever: Causal graph path retriever
            semantic_weight: Weight for semantic similarity (0-1)
            causal_weight: Weight for causal relevance (0-1)
            reranking_factor: How many more results to fetch from vector store for reranking
            min_causal_matches: Minimum causal nodes that must match for high priority
            cache_results: Whether to cache results for repeated queries
        """
        self.vector = vector_retriever
        self.graph = graph_retriever
        self.semantic_weight = semantic_weight
        self.causal_weight = causal_weight
        self.reranking_factor = reranking_factor
        self.min_causal_matches = min_causal_matches
        self.cache_results = cache_results
        
        # Results cache
        self.query_cache = {}
        self.last_query = None
        self.last_results = None
        
        # Normalize weights
        total = self.semantic_weight + self.causal_weight
        if total != 1.0:
            self.semantic_weight /= total
            self.causal_weight /= total
    
    def score_passage(self, 
                     passage: str, 
                     path_nodes: List[str], 
                     causal_paths: List[List[str]], 
                     semantic_score: float = 0.0) -> Tuple[float, Dict[str, Any]]:
        """
        Score a passage based on causal and semantic relevance
        
        Args:
            passage: Text passage to score
            path_nodes: List of causal node texts
            causal_paths: List of causal paths (each a list of nodes)
            semantic_score: Semantic similarity score (0-1)
            
        Returns:
            Tuple of (combined_score, details_dict)
        """
        # Convert passage to lowercase for case-insensitive matching
        passage_lower = passage.lower()
        
        # Count matching nodes
        matched_nodes = []
        for node in path_nodes:
            if node.lower() in passage_lower:
                matched_nodes.append(node)
        
        node_match_score = len(matched_nodes) / max(len(path_nodes), 1)
        
        # Check for causal path preservation
        path_matches = []
        for path in causal_paths:
            if len(path) < 2:
                continue
                
            # Count sequential pairs that appear in the passage
            sequential_matches = 0
            for i in range(len(path) - 1):
                cause = path[i].lower()
                effect = path[i+1].lower()
                
                if cause in passage_lower and effect in passage_lower:
                    # Check if they appear in the correct order
                    cause_pos = passage_lower.find(cause)
                    effect_pos = passage_lower.find(effect)
                    
                    if cause_pos < effect_pos:
                        sequential_matches += 1
                        path_matches.append((cause, effect))
            
            # Weight by how much of the path is preserved
            if sequential_matches > 0:
                # More weight to longer matching sequences
                path_match_score = sequential_matches / (len(path) - 1)
            else:
                path_match_score = 0
        
        # Combine path match scores across all paths
        if causal_paths and len(path_matches) > 0:
            overall_path_score = len(path_matches) / sum(len(p) - 1 for p in causal_paths if len(p) > 1)
        else:
            overall_path_score = 0
        
        # Combine node and path scores for overall causal score
        causal_score = 0.7 * node_match_score + 0.3 * overall_path_score
        
        # Final weighted score
        combined_score = (
            self.semantic_weight * semantic_score + 
            self.causal_weight * causal_score
        )
        
        # Return score and details for explanation
        return combined_score, {
            "matched_nodes": matched_nodes,
            "node_score": node_match_score,
            "path_matches": path_matches,
            "path_score": overall_path_score,
            "causal_score": causal_score,
            "semantic_score": semantic_score,
            "combined_score": combined_score
        }
    
    def retrieve(self, 
                query: str, 
                top_k: int = 5, 
                include_scores: bool = False, 
                include_details: bool = False) -> Union[
                    List[str], 
                    List[Tuple[str, float]], 
                    List[Dict[str, Any]]
                ]:
        """
        Retrieve passages using hybrid semantic+causal approach
        
        Args:
            query: User query
            top_k: Number of results to return
            include_scores: Whether to include scores in results
            include_details: Whether to include detailed scoring info
            
        Returns:
            List of passages, (passage, score) tuples, or detailed dicts
        """
        # Check cache first
        if self.cache_results and query in self.query_cache:
            cached_results = self.query_cache[query]
            
            # Filter by top_k and format as requested
            return self._format_results(cached_results[:top_k], include_scores, include_details)
            
        # Get more results than needed for reranking
        expanded_k = top_k * self.reranking_factor
        
        # Get semantic results with scores
        try:
            semantic_results = self.vector.search(query, top_k=expanded_k, include_scores=True)
        except Exception as e:
            logger.error(f"Error retrieving vector results: {e}")
            semantic_results = []
        
        # If no semantic results, try loading from cache
        if not semantic_results and self.cache_results and self.last_results:
            logger.info("Using cached results from last query")
            if include_details:
                return self.last_results[:top_k]
            else:
                return self._format_results(self.last_results[:top_k], include_scores, False)
        
        # Get causal information
        try:
            path_nodes = self.graph.retrieve_path_nodes(query)
            causal_paths = self.graph.retrieve_paths(query, max_paths=3)
        except Exception as e:
            logger.error(f"Error retrieving causal paths: {e}")
            path_nodes = []
            causal_paths = []
        
        # Score passages by combining semantic and causal signals
        scored_results = []
        
        for passage, semantic_score in semantic_results:
            score, details = self.score_passage(
                passage, path_nodes, causal_paths, semantic_score
            )
            
            # Create result object
            result = {
                "passage": passage,
                "score": score,
                "details": details
            }
            
            scored_results.append(result)
        
        # Sort by score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Cache results
        if self.cache_results:
            self.query_cache[query] = scored_results
            self.last_query = query
            self.last_results = scored_results
        
        # Return formatted results
        return self._format_results(scored_results[:top_k], include_scores, include_details)
    
    def _format_results(self, 
                       results: List[Dict[str, Any]], 
                       include_scores: bool, 
                       include_details: bool) -> Union[
                           List[str], 
                           List[Tuple[str, float]], 
                           List[Dict[str, Any]]
                       ]:
        """Format results based on requested output type"""
        if include_details:
            return results
        elif include_scores:
            return [(r["passage"], r["score"]) for r in results]
        else:
            return [r["passage"] for r in results]
    
    def get_explanation(self, query: str, passage: str) -> str:
        """
        Get human-readable explanation for why a passage was retrieved
        
        Args:
            query: Original query
            passage: Retrieved passage
            
        Returns:
            Explanation text
        """
        # Check if we have cached details
        if self.last_query == query and self.last_results:
            for result in self.last_results:
                if result["passage"] == passage:
                    details = result["details"]
                    
                    explanation = [f"Hybrid retrieval explanation for: {query}"]
                    
                    # Semantic component
                    explanation.append(f"\nSemantic relevance score: {details['semantic_score']:.2f} (weight: {self.semantic_weight:.2f})")
                    
                    # Causal component
                    explanation.append(f"\nCausal relevance score: {details['causal_score']:.2f} (weight: {self.causal_weight:.2f})")
                    
                    # Matched nodes
                    if details["matched_nodes"]:
                        explanation.append(f"\nMatched causal concepts ({len(details['matched_nodes'])} concepts):")
                        for node in details["matched_nodes"]:
                            explanation.append(f"- {node}")
                    
                    # Path matches
                    if details["path_matches"]:
                        explanation.append(f"\nPreserved causal relationships:")
                        for cause, effect in details["path_matches"]:
                            explanation.append(f"- {cause} → {effect}")
                    
                    # Overall score
                    explanation.append(f"\nOverall score: {details['combined_score']:.2f}")
                    
                    return "\n".join(explanation)
        
        # If not found in cache, provide generic explanation
        return f"This passage was retrieved as relevant to the query: {query}. No detailed scoring information is available."
    
    def clear_cache(self):
        """Clear the results cache"""
        self.query_cache = {}
        self.last_query = None
        self.last_results = None
