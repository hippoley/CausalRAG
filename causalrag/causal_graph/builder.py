# causal_graph/builder.py
# Responsible for parsing documents into causal triples and building the causal graph

import networkx as nx
import json
import os
from typing import List, Tuple, Dict, Optional, Set, Any, Union
from sentence_transformers import SentenceTransformer, util
import torch
import re
import logging

logger = logging.getLogger(__name__)

class CausalTripleExtractor:
    """Extract causal relationships from text using different methods"""
    
    def __init__(self, method: str = "hybrid", llm_interface=None):
        """
        Initialize the causal triple extractor
        
        Args:
            method: Extraction method ("rule", "llm", or "hybrid")
            llm_interface: Optional LLM interface for extraction
        """
        self.method = method
        self.llm_interface = llm_interface
        
        # Rule-based patterns for causal extraction (can be expanded)
        self.causal_patterns = [
            r"([\w\s]+)\s+causes\s+([\w\s]+)",
            r"([\w\s]+)\s+leads to\s+([\w\s]+)",
            r"([\w\s]+)\s+results in\s+([\w\s]+)",
            r"because of\s+([\w\s]+),\s+([\w\s]+)",
            r"([\w\s]+)\s+is caused by\s+([\w\s]+)",
            r"if\s+([\w\s]+),\s+then\s+([\w\s]+)",
            r"([\w\s]+)\s+contributes to\s+([\w\s]+)",
            r"([\w\s]+)\s+influences\s+([\w\s]+)"
        ]
    
    def extract(self, text: str) -> List[Tuple[str, str, Optional[float]]]:
        """
        Extract causal triples from text
        
        Args:
            text: Input text document
            
        Returns:
            List of (cause, effect, confidence) tuples
        """
        if self.method == "rule":
            return self._rule_based_extraction(text)
        elif self.method == "llm":
            return self._llm_based_extraction(text)
        else:  # hybrid
            rule_triples = self._rule_based_extraction(text)
            
            # If rule-based extraction finds nothing and LLM interface is available, try LLM
            if not rule_triples and self.llm_interface:
                return self._llm_based_extraction(text)
            return rule_triples
    
    def _rule_based_extraction(self, text: str) -> List[Tuple[str, str, Optional[float]]]:
        """Extract causal triples using rule-based patterns"""
        triples = []
        
        # Clean and prepare text
        clean_text = text.replace("\n", " ").strip()
        
        # Apply patterns
        for pattern in self.causal_patterns:
            matches = re.finditer(pattern, clean_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    # First pattern item is cause, second is effect
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                    
                    # Filter out spurious or too short matches
                    if len(cause) > 2 and len(effect) > 2:
                        triples.append((cause, effect, 0.8))  # Confidence score for rule-based
        
        return triples
    
    def _llm_based_extraction(self, text: str) -> List[Tuple[str, str, Optional[float]]]:
        """Extract causal triples using LLM"""
        if not self.llm_interface:
            logger.warning("LLM interface not provided, cannot perform LLM-based extraction")
            return []
        
        prompt = f"""Extract all causal relationships from the text below as a JSON list of objects.
Each object should have 'cause', 'effect', and 'confidence' (0.0-1.0) fields.
Focus only on strong causal relationships, not merely correlations or temporal sequences.

TEXT:
{text}

OUTPUT FORMAT:
[
  {{"cause": "...", "effect": "...", "confidence": 0.9}},
  ...
]

CAUSAL RELATIONSHIPS:"""
        
        try:
            response = self.llm_interface.generate(prompt, temperature=0.1, json_mode=True)
            
            # Parse JSON response
            if isinstance(response, str):
                # Find JSON in the response
                json_str = response.strip()
                if not json_str.startswith("["):
                    # Try to find the JSON array in the text
                    start_idx = json_str.find("[")
                    end_idx = json_str.rfind("]") + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = json_str[start_idx:end_idx]
                
                try:
                    triples_data = json.loads(json_str)
                    return [(item["cause"], item["effect"], item.get("confidence", 0.7)) 
                            for item in triples_data]
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM response as JSON: {response}")
                    return []
            elif isinstance(response, dict) or isinstance(response, list):
                # Direct JSON response
                if isinstance(response, dict):
                    # Handle case where response might be wrapped
                    triples_data = response.get("triples", [])
                else:
                    triples_data = response
                
                return [(item["cause"], item["effect"], item.get("confidence", 0.7)) 
                        for item in triples_data]
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                return []
                
        except Exception as e:
            logger.error(f"Error during LLM extraction: {e}")
            return []


class CausalGraphBuilder:
    """Build and maintain a causal graph from extracted triples"""
    
    def __init__(self, 
                model_name: str = "all-MiniLM-L6-v2",
                normalize_nodes: bool = True,
                confidence_threshold: float = 0.5,
                extractor_method: str = "hybrid",
                llm_interface = None):
        """
        Initialize the causal graph builder
        
        Args:
            model_name: Name of sentence transformer model for embeddings
            normalize_nodes: Whether to normalize node names (merge similar concepts)
            confidence_threshold: Minimum confidence to include triples
            extractor_method: Method for extracting causal relationships
            llm_interface: Optional LLM interface for extraction
        """
        self.graph = nx.DiGraph()
        self.node_text = {}  # Mapping from node ID to original text
        self.node_variants = {}  # Mapping from node ID to list of variant texts
        self.confidence_threshold = confidence_threshold
        self.normalize_nodes = normalize_nodes
        
        # Load embedding model
        try:
            self.encoder = SentenceTransformer(model_name)
            self.node_embeddings = {}  # Node ID to embedding mapping
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            self.encoder = None
            self.node_embeddings = {}
        
        # Initialize extractor
        self.extractor = CausalTripleExtractor(
            method=extractor_method,
            llm_interface=llm_interface
        )
    
    def add_triples(self, triples: List[Tuple[str, str, Optional[float]]]):
        """
        Add triples to the causal graph
        
        Args:
            triples: List of (cause, effect, confidence) tuples
        """
        for triple in triples:
            if len(triple) >= 2:
                cause, effect = triple[0], triple[1]
                # Get confidence if available, otherwise default to 1.0
                confidence = triple[2] if len(triple) > 2 and triple[2] is not None else 1.0
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Normalize nodes if enabled
                if self.normalize_nodes:
                    cause_id = self._get_or_create_node(cause)
                    effect_id = self._get_or_create_node(effect)
                else:
                    cause_id = cause
                    effect_id = effect
                    # Store original text
                    self.node_text[cause_id] = cause
                    self.node_text[effect_id] = effect
                
                # Add edge with confidence as attribute
                self.graph.add_edge(cause_id, effect_id, weight=confidence)
                
                # Update embeddings if encoder is available
                if self.encoder:
                    for node_id in [cause_id, effect_id]:
                        if node_id not in self.node_embeddings:
                            try:
                                text = self.node_text[node_id]
                                self.node_embeddings[node_id] = self.encoder.encode(
                                    text, convert_to_tensor=True
                                )
                            except Exception as e:
                                logger.error(f"Error encoding node {node_id}: {e}")
    
    def _get_or_create_node(self, text: str) -> str:
        """
        Get existing node ID for similar text or create new node
        
        Args:
            text: Node text to find or create
            
        Returns:
            Node ID (normalized if enabled)
        """
        if not self.encoder:
            return text  # Can't normalize without encoder
        
        # Encode the new text
        try:
            text_emb = self.encoder.encode(text, convert_to_tensor=True)
        except:
            return text  # Fall back to using the text directly
        
        # Check similarity with existing nodes
        best_match = None
        best_score = 0.0
        
        for node_id, emb in self.node_embeddings.items():
            try:
                score = util.pytorch_cos_sim(text_emb, emb).item()
                if score > 0.85 and score > best_score:  # High similarity threshold
                    best_match = node_id
                    best_score = score
            except:
                continue
        
        if best_match:
            # Add this text as a variant
            if best_match not in self.node_variants:
                self.node_variants[best_match] = []
            self.node_variants[best_match].append(text)
            return best_match
        else:
            # Create new node
            node_id = text
            self.node_text[node_id] = text
            try:
                self.node_embeddings[node_id] = text_emb
            except:
                pass
            return node_id
    
    def index_documents(self, docs: List[str]) -> int:
        """
        Process documents and update the causal graph
        
        Args:
            docs: List of document texts
            
        Returns:
            Number of triples added to the graph
        """
        initial_edge_count = self.graph.number_of_edges()
        
        for doc in docs:
            triples = self.extractor.extract(doc)
            self.add_triples(triples)
        
        new_edges = self.graph.number_of_edges() - initial_edge_count
        return new_edges
    
    def get_graph(self) -> nx.DiGraph:
        """Get the current causal graph"""
        return self.graph
    
    def get_node_variants(self, node_id: str) -> List[str]:
        """Get all text variants for a node"""
        variants = self.node_variants.get(node_id, [])
        return [self.node_text[node_id]] + variants
    
    def get_embedding(self, node_id: str) -> Optional[torch.Tensor]:
        """Get the embedding for a node"""
        return self.node_embeddings.get(node_id)
    
    def describe_graph(self) -> str:
        """Get a text description of the causal graph"""
        if self.graph.number_of_edges() == 0:
            return "Empty causal graph (no causal relationships found)"
        
        edge_texts = []
        for a, b, data in self.graph.edges(data=True):
            confidence = data.get('weight', 1.0)
            a_text = self.node_text.get(a, a)
            b_text = self.node_text.get(b, b)
            edge_texts.append(f"{a_text} → {b_text} (confidence: {confidence:.2f})")
        
        return "\n".join(edge_texts)
    
    def save(self, filepath: str):
        """Save the causal graph to a file"""
        data = {
            'nodes': dict(self.node_text),
            'variants': dict(self.node_variants),
            'edges': [(a, b, dict(data)) for a, b, data in self.graph.edges(data=True)]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load a causal graph from a file"""
        if not os.path.exists(filepath):
            logger.error(f"Graph file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reset current graph
            self.graph = nx.DiGraph()
            self.node_text = data.get('nodes', {})
            self.node_variants = data.get('variants', {})
            
            # Rebuild embeddings
            self.node_embeddings = {}
            if self.encoder:
                for node_id, text in self.node_text.items():
                    try:
                        self.node_embeddings[node_id] = self.encoder.encode(
                            text, convert_to_tensor=True
                        )
                    except:
                        pass
            
            # Add edges
            for a, b, data in data.get('edges', []):
                self.graph.add_edge(a, b, **data)
            
            return True
        except Exception as e:
            logger.error(f"Error loading graph from {filepath}: {e}")
            return False