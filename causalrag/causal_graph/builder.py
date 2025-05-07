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
            r"([\w\s]+)\s+influences\s+([\w\s]+)",
            r"([\w\s]+)\s+leads\s+to\s+([\w\s]+)",
            r"([\w\s]+)\s+triggers\s+([\w\s]+)",
            r"([\w\s]+)\s+induces\s+([\w\s]+)",
            r"([\w\s]+)\s+drives\s+([\w\s]+)"
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
            # In hybrid mode, always try both methods and combine
            rule_triples = self._rule_based_extraction(text)
            
            # If we have LLM interface available, use it
            if self.llm_interface:
                llm_triples = self._llm_based_extraction(text)
                
                # Combine and deduplicate triples
                combined_triples = rule_triples.copy()
                
                # Add LLM triples that aren't duplicates
                existing_pairs = {(c, e) for c, e, _ in rule_triples}
                for cause, effect, conf in llm_triples:
                    if (cause, effect) not in existing_pairs:
                        combined_triples.append((cause, effect, conf))
                    
                return combined_triples
            
            return rule_triples
    
    def _rule_based_extraction(self, text: str) -> List[Tuple[str, str, Optional[float]]]:
        """Extract causal triples using rule-based patterns"""
        triples = []
        
        # Clean and prepare text
        clean_text = text.replace("\n", " ").strip()
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', clean_text)
        
        for sentence in sentences:
            # Apply patterns to each sentence
            for pattern in self.causal_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        # First pattern item is cause, second is effect
                        cause = match.group(1).strip()
                        effect = match.group(2).strip()
                        
                        # Filter out spurious or too short matches
                        if len(cause) > 2 and len(effect) > 2:
                            # Confidence based on pattern match and context
                            conf = 0.8
                            # Reduce confidence for very general statements
                            if len(cause) < 5 or len(effect) < 5:
                                conf = 0.6
                            
                            triples.append((cause, effect, conf))
        
        return triples
    
    def _llm_based_extraction(self, text: str) -> List[Tuple[str, str, Optional[float]]]:
        """Extract causal triples using LLM"""
        if not self.llm_interface:
            logger.warning("LLM interface not provided, cannot perform LLM-based extraction")
            return []
        
        # Split long texts into chunks for processing
        chunks = self._split_text_into_chunks(text, max_length=3000)
        all_triples = []
        
        for chunk in chunks:
            # Create the LLM prompt for causal extraction
            prompt = self._create_causal_extraction_prompt(chunk)
            
            try:
                response = self.llm_interface.generate(
                    prompt, 
                    temperature=0.1, 
                    json_mode=True
                )
                
                # Parse triples from response and add to results
                chunk_triples = self._parse_llm_response(response)
                all_triples.extend(chunk_triples)
                
            except Exception as e:
                logger.error(f"Error during LLM extraction for chunk: {e}")
        
        # Deduplicate and return combined triples
        return self._deduplicate_triples(all_triples)
    
    def _create_causal_extraction_prompt(self, text: str) -> str:
        """Create a structured prompt for causal extraction"""
        return f"""Extract all causal relationships from the text below as a JSON list of objects.
Each object should have 'cause', 'effect', and 'confidence' (0.0-1.0) fields.
Focus only on causal relationships, not merely correlations or temporal sequences.

TYPES OF CAUSAL RELATIONSHIPS TO IDENTIFY:
1. Direct causation: A directly causes B (e.g., "smoking causes lung cancer")
2. Mediated causation: A causes B through C (e.g., "smoking damages lung tissue, which leads to cancer")
3. Enabling causation: A creates conditions for B to occur (e.g., "drought creates conditions for wildfires")
4. Preventive causation: A prevents or reduces B (e.g., "vaccines prevent disease")
5. Contributory causation: A contributes to or increases risk of B (e.g., "pollution contributes to respiratory problems")

CONFIDENCE SCORING CRITERIA (0.0-1.0):
- 0.9-1.0: Explicit, unambiguous causal claim with strong causal language (causes, leads to, results in)
- 0.7-0.8: Clear causal relationship but with less explicit language (contributes to, influences)
- 0.5-0.6: Implied causation requiring some inference (associated with + mechanism described)
- 0.3-0.4: Suggested but uncertain causation (may cause, could lead to, is linked to)
- < 0.3: Primarily correlational or too uncertain to include

EXTRACTION GUIDELINES:
1. Focus on factual causal claims, not hypothetical or counterfactual statements
2. Extract the most specific cause and effect possible, avoiding overly general concepts
3. For complex causal chains (A→B→C→D), extract all individual links (A→B, B→C, C→D)
4. When multiple causes lead to the same effect, extract each relationship separately
5. Normalize forms but preserve key content (e.g., "global warming" vs "climate change")
6. Capture directional relationships correctly (what causes what)
7. When confidence is below 0.5, only include if the relationship is particularly significant

TEXT:
{text}

OUTPUT FORMAT:
[
  {{"cause": "climate change", "effect": "rising sea levels", "confidence": 0.95}},
  {{"cause": "rising sea levels", "effect": "coastal flooding", "confidence": 0.9}},
  {{"cause": "deforestation", "effect": "increased atmospheric CO2", "confidence": 0.85}},
  {{"cause": "regular exercise", "effect": "reduced risk of heart disease", "confidence": 0.8}}
]

Ensure your response contains ONLY the valid JSON array. Do not include any other explanation or text.
CAUSAL RELATIONSHIPS:"""
    
    def _parse_llm_response(self, response: Union[str, List, Dict]) -> List[Tuple[str, str, Optional[float]]]:
        """Parse the LLM response into a list of causal triples"""
        triples = []
        
        try:
            # Handle different response types
            if isinstance(response, str):
                # Extract JSON from text response
                json_str = response.strip()
                
                # Find JSON array by looking for the first [ and last ]
                start_idx = json_str.find("[")
                end_idx = json_str.rfind("]") + 1
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]
                    
                    # Clean any trailing or leading non-JSON content
                    json_str = json_str.strip()
                    
                    # Try to parse the JSON
                    try:
                        triples_data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Try to fix common JSON issues
                        fixed_json = self._fix_json_errors(json_str)
                        try:
                            triples_data = json.loads(fixed_json)
                            logger.info(f"Successfully fixed JSON parsing issue")
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse LLM response as JSON after fixing: {e}")
                            logger.debug(f"Raw response: {response[:500]}...")
                            return []
                else:
                    logger.error(f"No valid JSON array found in response")
                    logger.debug(f"Raw response: {response[:500]}...")
                    return []
                
            elif isinstance(response, list):
                triples_data = response
            elif isinstance(response, dict):
                # Handle case where response might be wrapped
                if "triples" in response:
                    triples_data = response["triples"]
                elif "data" in response: 
                    triples_data = response["data"]
                elif "results" in response:
                    triples_data = response["results"]
                elif "cause" in response and "effect" in response:
                    # Handle case of single triple returned as dict
                    triples_data = [response]
                else:
                    logger.error(f"Unexpected response structure: {list(response.keys())}")
                    return []
            else:
                logger.error(f"Unexpected response type: {type(response)}")
                return []
            
            # Validate triples_data is a list
            if not isinstance(triples_data, list):
                logger.error(f"Expected list of triples, got {type(triples_data)}")
                return []
            
            # Convert to our triple format
            for item in triples_data:
                if not isinstance(item, dict):
                    logger.warning(f"Expected dict for triple, got {type(item)}")
                    continue
                
                # Extract cause and effect with robust error handling
                cause = self._safe_extract_field(item, "cause")
                effect = self._safe_extract_field(item, "effect")
                
                # Skip invalid triples
                if not cause or not effect:
                    continue
                
                # Extract confidence with fallback
                confidence = self._safe_extract_confidence(item)
                
                # Validate and clean
                if len(cause) > 2 and len(effect) > 2:
                    # Normalize triple
                    cause = self._normalize_text(cause)
                    effect = self._normalize_text(effect)
                    
                    # Add to results
                    triples.append((cause, effect, confidence))
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Log extraction statistics
        logger.info(f"Extracted {len(triples)} causal relationships")
        return triples
    
    def _safe_extract_field(self, item: Dict, field: str) -> str:
        """Safely extract a field from a triple dict with multiple fallbacks"""
        # Try the standard field name
        if field in item and item[field]:
            return str(item[field]).strip()
        
        # Try alternative field names
        alternatives = {
            "cause": ["source", "from", "antecedent", "reason", "origin"],
            "effect": ["target", "to", "consequent", "result", "destination", "outcome"]
        }
        
        if field in alternatives:
            for alt in alternatives[field]:
                if alt in item and item[alt]:
                    return str(item[alt]).strip()
        
        # Handle capitalized or uppercase versions
        for key in item.keys():
            if key.lower() == field.lower():
                return str(item[key]).strip()
        
        return ""
    
    def _safe_extract_confidence(self, item: Dict) -> float:
        """Extract confidence value with fallbacks and validation"""
        # Try standard field name
        for field in ["confidence", "weight", "score", "probability", "certainty"]:
            if field in item:
                try:
                    conf = float(item[field])
                    # Ensure in 0-1 range
                    return max(0.0, min(1.0, conf))
                except (ValueError, TypeError):
                    pass
        
        # Handle capitalized or uppercase versions
        for key in item.keys():
            if key.lower() == "confidence":
                try:
                    conf = float(item[key])
                    return max(0.0, min(1.0, conf))
                except (ValueError, TypeError):
                    pass
        
        # Default confidence
        return 0.7
    
    def _normalize_text(self, text: str) -> str:
        """Clean and normalize text for consistent processing"""
        # Trim whitespace and quotes
        text = text.strip().strip('"\'')
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure text doesn't end with punctuation
        text = re.sub(r'[.,;:!?]$', '', text)
        
        return text
    
    def _fix_json_errors(self, json_str: str) -> str:
        """Attempt to fix common JSON formatting errors"""
        # Replace single quotes with double quotes
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'(\s*})(\s*,\s*{)', r'\1,\2', json_str)
        
        # Fix trailing commas in arrays
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix missing commas between objects
        json_str = re.sub(r'}\s*{', '},{', json_str)
        
        # Fix unquoted property names
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        return json_str
    
    def _split_text_into_chunks(self, text: str, max_length: int = 3000) -> List[str]:
        """Split long text into manageable chunks for LLM processing"""
        if len(text) <= max_length:
            return [text]
        
        # Try to split on paragraph boundaries
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_length:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too long, split it further
                if len(para) > max_length:
                    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', para)
                    sent_chunk = ""
                    
                    for sent in sentences:
                        if len(sent_chunk) + len(sent) + 1 <= max_length:
                            if sent_chunk:
                                sent_chunk += " "
                            sent_chunk += sent
                        else:
                            if sent_chunk:
                                chunks.append(sent_chunk)
                            sent_chunk = sent
                    
                    if sent_chunk:
                        current_chunk = sent_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _deduplicate_triples(self, triples: List[Tuple[str, str, Optional[float]]]) -> List[Tuple[str, str, Optional[float]]]:
        """Remove duplicate triples, keeping the one with higher confidence"""
        if not triples:
            return []
        
        # Use dictionary to track best confidence for each cause-effect pair
        unique_triples = {}
        
        for cause, effect, conf in triples:
            key = (cause.lower(), effect.lower())
            
            if key not in unique_triples or conf > unique_triples[key][2]:
                unique_triples[key] = (cause, effect, conf)
        
        return list(unique_triples.values())

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
    
    def index_documents(self, docs: List[str], batch_size: int = 5, show_progress: bool = True) -> int:
        """
        Process documents and update the causal graph
        
        Args:
            docs: List of document texts
            batch_size: Number of documents to process in each batch
            show_progress: Whether to show a progress bar
            
        Returns:
            Number of triples added to the graph
        """
        initial_edge_count = self.graph.number_of_edges()
        initial_node_count = self.graph.number_of_nodes()
        
        # Use tqdm for progress tracking if available and requested
        try:
            from tqdm import tqdm
            doc_iterator = tqdm(docs, desc="Building causal graph") if show_progress else docs
        except ImportError:
            doc_iterator = docs
            if show_progress:
                logger.info(f"Processing {len(docs)} documents (tqdm not available for progress bar)")
        
        # Process documents in batches
        all_triples = []
        batch_count = 0
        
        # Create batches
        doc_batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
        
        for batch in doc_batches:
            batch_triples = []
            batch_count += 1
            
            # Process each document in the batch
            for doc in batch:
                if not doc or len(doc.strip()) < 10:  # Skip empty or very short docs
                    continue
                    
                try:
                    doc_triples = self.extractor.extract(doc)
                    batch_triples.extend(doc_triples)
                except Exception as e:
                    logger.error(f"Error extracting triples from document: {str(e)}")
            
            # Add batch triples to the graph
            self.add_triples(batch_triples)
            all_triples.extend(batch_triples)
            
            # Log batch progress
            if batch_count % 5 == 0 or batch_count == len(doc_batches):
                logger.info(f"Processed batch {batch_count}/{len(doc_batches)}: " 
                          f"found {len(batch_triples)} causal relationships")
        
        # Log overall results
        new_edges = self.graph.number_of_edges() - initial_edge_count
        new_nodes = self.graph.number_of_nodes() - initial_node_count
        
        logger.info(f"Indexing complete: {len(docs)} documents processed")
        logger.info(f"Added {new_nodes} new nodes and {new_edges} new relationships to graph")
        logger.info(f"Graph now has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
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
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the extracted causal relationships
        
        Returns:
            Dictionary with statistics about the causal graph
        """
        if not self.graph:
            return {"error": "Graph is empty"}
        
        # Basic graph statistics
        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()
        
        # Node degree statistics
        in_degrees = [d for n, d in self.graph.in_degree()]
        out_degrees = [d for n, d in self.graph.out_degree()]
        
        # Find highest confidence relationships
        edge_confidences = []
        for u, v, data in self.graph.edges(data=True):
            conf = data.get('weight', 0.5)
            edge_confidences.append((u, v, conf))
        
        # Sort by confidence
        edge_confidences.sort(key=lambda x: x[2], reverse=True)
        
        # Get top confidence relationships
        top_relationships = []
        if edge_confidences:
            for u, v, conf in edge_confidences[:10]:  # Top 10
                u_text = self.node_text.get(u, str(u))
                v_text = self.node_text.get(v, str(v))
                top_relationships.append({
                    "cause": u_text,
                    "effect": v_text,
                    "confidence": conf
                })
        
        # Identify central nodes (highest total degree)
        total_degrees = {n: self.graph.in_degree(n) + self.graph.out_degree(n) 
                        for n in self.graph.nodes()}
        central_nodes = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)
        
        central_concepts = []
        if central_nodes:
            for node, degree in central_nodes[:10]:  # Top 10
                node_text = self.node_text.get(node, str(node))
                central_concepts.append({
                    "concept": node_text,
                    "connections": degree,
                    "in_degree": self.graph.in_degree(node),
                    "out_degree": self.graph.out_degree(node)
                })
        
        # Measure graph connectivity
        try:
            import networkx as nx
            is_dag = nx.is_directed_acyclic_graph(self.graph)
            has_cycles = not is_dag
            weakly_connected_components = len(list(nx.weakly_connected_components(self.graph)))
        except:
            is_dag = "Unknown"
            has_cycles = "Unknown"
            weakly_connected_components = "Unknown"
        
        return {
            "graph_statistics": {
                "nodes": nodes,
                "edges": edges,
                "density": edges / (nodes * (nodes - 1)) if nodes > 1 else 0,
                "is_dag": is_dag,
                "has_cycles": has_cycles,
                "components": weakly_connected_components
            },
            "degree_statistics": {
                "max_in_degree": max(in_degrees) if in_degrees else 0,
                "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
                "max_out_degree": max(out_degrees) if out_degrees else 0,
                "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0
            },
            "top_relationships": top_relationships,
            "central_concepts": central_concepts
        }
    
    def visualize_graph(self, 
                      output_path: Optional[str] = None, 
                      format: str = "html", 
                      max_nodes: int = 100,
                      min_edge_weight: float = 0.0,
                      highlight_nodes: List[str] = None,
                      title: str = "Causal Knowledge Graph") -> Optional[str]:
        """
        Visualize the causal graph as an interactive network diagram
        
        Args:
            output_path: Path to save the visualization file
            format: Output format ("html", "png", "json", "networkx")
            max_nodes: Maximum number of nodes to include (for readability)
            min_edge_weight: Minimum edge weight to include
            highlight_nodes: List of node IDs to highlight
            title: Title for the visualization
            
        Returns:
            Path to the output file or network data if format is "networkx"
        """
        if not self.graph.nodes():
            logger.error("Cannot visualize empty graph")
            return None
        
        # Create a subgraph with the most important nodes if graph is large
        graph_to_viz = self.graph
        if self.graph.number_of_nodes() > max_nodes:
            # Calculate node importance as sum of edge weights
            node_importance = {}
            for node in self.graph.nodes():
                in_edges = [(u, v, d.get('weight', 0.5)) for u, v, d in self.graph.in_edges(node, data=True)]
                out_edges = [(u, v, d.get('weight', 0.5)) for u, v, d in self.graph.out_edges(node, data=True)]
                total_weight = sum(w for _, _, w in in_edges + out_edges)
                node_importance[node] = total_weight
            
            # Sort nodes by importance
            important_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            important_node_ids = [n for n, _ in important_nodes]
            
            # Create subgraph with important nodes
            graph_to_viz = self.graph.subgraph(important_node_ids)
            
            logger.info(f"Created visualization subgraph with {graph_to_viz.number_of_nodes()} nodes " 
                      f"(from original {self.graph.number_of_nodes()} nodes)")
        
        # Apply edge weight filter if specified
        if min_edge_weight > 0:
            edges_to_remove = [(u, v) for u, v, d in graph_to_viz.edges(data=True) 
                              if d.get('weight', 0.5) < min_edge_weight]
            graph_to_viz.remove_edges_from(edges_to_remove)
            
            # Remove isolated nodes
            isolated_nodes = [n for n in graph_to_viz.nodes() if graph_to_viz.degree(n) == 0]
            graph_to_viz.remove_nodes_from(isolated_nodes)
            
            logger.info(f"Filtered edges with weight < {min_edge_weight}, " 
                      f"resulting in {graph_to_viz.number_of_edges()} edges and {graph_to_viz.number_of_nodes()} nodes")
            
        # Format output based on specified format
        try:
            if format.lower() == "networkx":
                return graph_to_viz
                
            elif format.lower() == "json":
                # Create a JSON representation of the graph
                nodes_data = []
                for node in graph_to_viz.nodes():
                    node_data = {
                        "id": str(node),
                        "label": self.node_text.get(node, str(node)),
                        "in_degree": graph_to_viz.in_degree(node),
                        "out_degree": graph_to_viz.out_degree(node)
                    }
                    if highlight_nodes and node in highlight_nodes:
                        node_data["highlighted"] = True
                    nodes_data.append(node_data)
                
                edges_data = []
                for u, v, data in graph_to_viz.edges(data=True):
                    edge_data = {
                        "source": str(u),
                        "target": str(v),
                        "weight": data.get('weight', 0.5)
                    }
                    edges_data.append(edge_data)
                
                graph_data = {
                    "nodes": nodes_data,
                    "edges": edges_data,
                    "metadata": {
                        "title": title,
                        "node_count": len(nodes_data),
                        "edge_count": len(edges_data)
                    }
                }
                
                if output_path:
                    with open(output_path, 'w') as f:
                        json.dump(graph_data, f, indent=2)
                    logger.info(f"Saved graph visualization as JSON to {output_path}")
                    return output_path
                else:
                    return json.dumps(graph_data)
                    
            elif format.lower() in ["html", "png"]:
                try:
                    # Try using pyvis for interactive HTML visualization
                    from pyvis.network import Network
                    
                    # Create network
                    net = Network(height="750px", width="100%", notebook=False, directed=True)
                    net.heading = title
                    
                    # Add nodes
                    for node in graph_to_viz.nodes():
                        node_text = self.node_text.get(node, str(node))
                        node_options = {}
                        
                        # Highlight nodes if specified
                        if highlight_nodes and node in highlight_nodes:
                            node_options = {
                                "color": "#ff6347",
                                "borderWidth": 2,
                                "borderWidthSelected": 4
                            }
                        
                        net.add_node(node, label=node_text, title=node_text, **node_options)
                    
                    # Add edges with weights
                    for u, v, data in graph_to_viz.edges(data=True):
                        weight = data.get('weight', 0.5)
                        width = 1 + 4 * weight  # Scale width based on weight
                        title = f"{self.node_text.get(u, str(u))} → {self.node_text.get(v, str(v))} ({weight:.2f})"
                        net.add_edge(u, v, width=width, title=title, value=weight)
                    
                    # Set physics options for better layout
                    net.set_options("""
                    {
                      "physics": {
                        "forceAtlas2Based": {
                          "gravitationalConstant": -50,
                          "centralGravity": 0.01,
                          "springLength": 100,
                          "springConstant": 0.08
                        },
                        "solver": "forceAtlas2Based",
                        "stabilization": {
                          "enabled": true,
                          "iterations": 100
                        }
                      },
                      "edges": {
                        "color": {
                          "inherit": true
                        },
                        "smooth": {
                          "enabled": true,
                          "type": "continuous"
                        },
                        "arrows": {
                          "to": {
                            "enabled": true,
                            "scaleFactor": 0.5
                          }
                        }
                      }
                    }
                    """)
                    
                    # Write HTML file
                    if not output_path:
                        output_path = "causal_graph.html"
                    
                    # Save as HTML
                    net.save_graph(output_path)
                    logger.info(f"Saved interactive graph visualization to {output_path}")
                    
                    # If PNG was requested, try to convert HTML to PNG
                    if format.lower() == "png" and output_path.endswith('.html'):
                        try:
                            # Try using playwright first
                            from playwright.sync_api import sync_playwright
                            with sync_playwright() as p:
                                browser = p.chromium.launch()
                                page = browser.new_page()
                                page.goto(f"file://{os.path.abspath(output_path)}")
                                page.wait_for_timeout(3000)  # Wait for force-directed layout to stabilize
                                page.screenshot(path=output_path.replace('.html', '.png'), full_page=True)
                                browser.close()
                            logger.info(f"Saved static graph image to {output_path.replace('.html', '.png')}")
                            return output_path.replace('.html', '.png')
                        except ImportError:
                            # Fall back to selenium
                            import selenium.webdriver
                            from selenium.webdriver.chrome.options import Options
                            chrome_options = Options()
                            chrome_options.add_argument("--headless")
                            chrome_options.add_argument("--disable-gpu")
                            driver = selenium.webdriver.Chrome(options=chrome_options)
                            driver.get(f"file://{os.path.abspath(output_path)}")
                            driver.implicitly_wait(3)
                            driver.save_screenshot(output_path.replace('.html', '.png'))
                            driver.quit()
                            logger.info(f"Saved static graph image to {output_path.replace('.html', '.png')}")
                            return output_path.replace('.html', '.png')
                        except Exception as e:
                            logger.error(f"Error converting HTML to PNG: {e}")
                            logger.info(f"Using HTML visualization instead at {output_path}")
                    
                    return output_path
                    
                except ImportError:
                    # Fall back to networkx drawing capabilities
                    import matplotlib.pyplot as plt
                    
                    plt.figure(figsize=(12, 10))
                    plt.title(title)
                    
                    # Get node positions using spring layout
                    pos = nx.spring_layout(graph_to_viz, seed=42)
                    
                    # Draw nodes
                    node_colors = ['#ff6347' if highlight_nodes and n in highlight_nodes else '#6495ed' 
                                 for n in graph_to_viz.nodes()]
                    nx.draw_networkx_nodes(graph_to_viz, pos, node_color=node_colors, alpha=0.8, node_size=700)
                    
                    # Draw edges with width proportional to weight
                    edges = graph_to_viz.edges(data=True)
                    edge_widths = [1 + 3 * data.get('weight', 0.5) for _, _, data in edges]
                    nx.draw_networkx_edges(graph_to_viz, pos, width=edge_widths, alpha=0.7, 
                                         arrows=True, arrowsize=15, arrowstyle='->')
                    
                    # Draw labels
                    labels = {n: self.node_text.get(n, str(n)) for n in graph_to_viz.nodes()}
                    nx.draw_networkx_labels(graph_to_viz, pos, labels=labels, font_size=10, font_weight='bold')
                    
                    plt.axis('off')
                    
                    # Save to file
                    if not output_path:
                        output_path = "causal_graph.png"
                    
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Saved graph visualization to {output_path}")
                    return output_path
            
            else:
                logger.error(f"Unsupported visualization format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error visualizing graph: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None