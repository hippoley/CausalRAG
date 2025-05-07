# causal_graph/retriever.py
# Responsible for finding query-relevant nodes or subgraphs from causal graph

import networkx as nx
from typing import List, Tuple, Dict, Set, Optional, Any, Union
import torch
from sentence_transformers import util
import logging
import heapq

logger = logging.getLogger(__name__)

class CausalPathRetriever:
    """Retrieve relevant causal paths from the causal graph"""
    
    def __init__(self, builder: 'CausalGraphBuilder'):
        """
        Initialize the causal path retriever
        
        Args:
            builder: CausalGraphBuilder containing the graph and embeddings
        """
        self.graph = builder.get_graph()
        self.node_embeddings = builder.node_embeddings
        self.node_text = builder.node_text
        self.encoder = builder.encoder
        self.builder = builder
    
    def retrieve_nodes(self, 
                      query: str, 
                      top_k: int = 5, 
                      threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Retrieve the most semantically relevant nodes to a query
        
        Args:
            query: User query text
            top_k: Number of top nodes to retrieve
            threshold: Minimum similarity score threshold
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        if not self.encoder or not self.node_embeddings:
            logger.warning("Encoder or node embeddings not available")
            return []
        
        try:
            # Encode query
            q_emb = self.encoder.encode(query, convert_to_tensor=True)
            
            # Calculate similarity with all nodes
            scores = {}
            for node_id, emb in self.node_embeddings.items():
                try:
                    sim = util.pytorch_cos_sim(q_emb, emb).item()
                    if sim >= threshold:
                        scores[node_id] = sim
                except Exception as e:
                    logger.error(f"Error calculating similarity for node {node_id}: {e}")
            
            # Sort by similarity score
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        except Exception as e:
            logger.error(f"Error retrieving nodes: {e}")
            return []
    
    def retrieve_path_nodes(self, 
                          query: str, 
                          top_k: int = 5, 
                          max_hops: int = 2,
                          include_similar: bool = True) -> List[str]:
        """
        Retrieve a set of nodes related to the query through causal paths
        
        Args:
            query: User query text
            top_k: Number of top seed nodes to consider
            max_hops: Maximum path length from seed nodes
            include_similar: Whether to include semantically similar nodes
            
        Returns:
            List of node IDs in the causal neighborhood
        """
        # Get top nodes by semantic similarity
        top_nodes = self.retrieve_nodes(query, top_k=top_k)
        
        # Collect all nodes in paths starting from or ending at top nodes
        path_nodes = set()
        seed_nodes = [node_id for node_id, _ in top_nodes]
        
        for node_id in seed_nodes:
            # Add the node itself
            path_nodes.add(node_id)
            
            # Add nodes reachable within max_hops (effects)
            descendants = self._get_descendants(node_id, max_hops)
            path_nodes.update(descendants)
            
            # Add nodes that can reach this node within max_hops (causes)
            ancestors = self._get_ancestors(node_id, max_hops)
            path_nodes.update(ancestors)
        
        # Optionally add semantically similar nodes that weren't in the top-k
        if include_similar and len(seed_nodes) > 0 and self.encoder:
            try:
                # Get average embedding of seed nodes
                seed_embs = [self.node_embeddings[n] for n in seed_nodes if n in self.node_embeddings]
                if seed_embs:
                    avg_emb = torch.mean(torch.stack(seed_embs), dim=0)
                    
                    # Find similar nodes
                    for node_id, emb in self.node_embeddings.items():
                        if node_id not in path_nodes:
                            sim = util.pytorch_cos_sim(avg_emb, emb).item()
                            if sim > 0.8:  # High similarity threshold
                                path_nodes.add(node_id)
            except Exception as e:
                logger.error(f"Error including similar nodes: {e}")
        
        # Convert to list and return
        return list(path_nodes)
    
    def _get_descendants(self, node: str, max_hops: int) -> Set[str]:
        """Get all descendants within max_hops"""
        descendants = set()
        current = {node}
        
        for _ in range(max_hops):
            next_nodes = set()
            for n in current:
                try:
                    next_nodes.update(self.graph.successors(n))
                except Exception:
                    pass  # Node might not exist
            
            descendants.update(next_nodes)
            current = next_nodes
            
            if not current:
                break
        
        return descendants
    
    def _get_ancestors(self, node: str, max_hops: int) -> Set[str]:
        """Get all ancestors within max_hops"""
        ancestors = set()
        current = {node}
        
        for _ in range(max_hops):
            next_nodes = set()
            for n in current:
                try:
                    next_nodes.update(self.graph.predecessors(n))
                except Exception:
                    pass  # Node might not exist
            
            ancestors.update(next_nodes)
            current = next_nodes
            
            if not current:
                break
        
        return ancestors
    
    def retrieve_paths(self, 
                      query: str, 
                      max_paths: int = 5,
                      min_path_length: int = 2,
                      max_path_length: int = 4) -> List[List[str]]:
        """
        Retrieve complete causal paths relevant to the query
        
        Args:
            query: User query
            max_paths: Maximum number of paths to return
            min_path_length: Minimum path length to include
            max_path_length: Maximum path length to consider
            
        Returns:
            List of paths, where each path is a list of node IDs
        """
        # Get relevant nodes
        relevant_nodes = self.retrieve_path_nodes(query, top_k=5, max_hops=1)
        
        if len(relevant_nodes) < 2:
            return []
        
        # Find all simple paths between relevant nodes
        paths = []
        for i, src in enumerate(relevant_nodes):
            for tgt in relevant_nodes[i+1:]:
                if src != tgt:
                    try:
                        # Try to find paths in both directions
                        for direction in [(src, tgt), (tgt, src)]:
                            start, end = direction
                            for path in nx.all_simple_paths(
                                self.graph, start, end, 
                                cutoff=max_path_length
                            ):
                                if len(path) >= min_path_length:
                                    # Convert IDs to text for readability
                                    text_path = [self.node_text.get(n, n) for n in path]
                                    paths.append((path, text_path))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
        
        # Sort paths by relevance (currently length, could be improved)
        sorted_paths = sorted(paths, key=lambda p: len(p[0]))
        
        # Return the text version of the paths
        return [text_path for _, text_path in sorted_paths[:max_paths]]
    
    def get_causal_explanation(self, query: str) -> str:
        """
        Generate a textual explanation of causal relationships relevant to the query
        
        Args:
            query: User query
            
        Returns:
            Human-readable explanation of causal relationships
        """
        paths = self.retrieve_paths(query, max_paths=3)
        
        if not paths:
            return "No relevant causal relationships found."
        
        explanation = f"Causal relationships relevant to '{query}':\n\n"
        
        for i, path in enumerate(paths):
            explanation += f"{i+1}. {' → '.join(path)}\n"
        
        return explanation
    
    def highlight_subgraph(self, query: str) -> nx.DiGraph:
        """
        Extract a query-relevant subgraph for visualization
        
        Args:
            query: User query
            
        Returns:
            NetworkX DiGraph containing the relevant subgraph
        """
        nodes = self.retrieve_path_nodes(query)
        return self.graph.subgraph(nodes)