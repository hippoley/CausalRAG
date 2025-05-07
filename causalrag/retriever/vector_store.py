# retriever/vector_store.py
# Provides vector-based semantic retrieval with multiple backend options

from typing import List, Dict, Tuple, Optional, Union, Any
import os
import logging
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class VectorStoreRetriever:
    """
    Vector-based retrieval component supporting multiple backends
    including FAISS, in-memory, and external vector stores
    """
    
    def __init__(self, 
                model_name: str = "all-MiniLM-L6-v2",
                backend: str = "faiss",
                dimension: Optional[int] = None,
                batch_size: int = 32,
                cache_dir: Optional[str] = None):
        """
        Initialize the vector store retriever
        
        Args:
            model_name: Name of sentence transformer model for encodings
            backend: Retrieval backend ("faiss", "memory", "weaviate", etc.)
            dimension: Vector dimension (if None, determined from model)
            batch_size: Batch size for encoding
            cache_dir: Optional directory to cache vectors
        """
        self.model_name = model_name
        self.backend = backend.lower()
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        # Initialize encoder
        try:
            self.encoder = SentenceTransformer(model_name)
            # Get dimension from model if not specified
            self.dimension = dimension or self.encoder.get_sentence_embedding_dimension()
            logger.info(f"Initialized encoder: {model_name} (dim={self.dimension})")
        except Exception as e:
            logger.error(f"Error initializing encoder: {e}")
            self.encoder = None
            self.dimension = dimension or 384  # Default fallback
        
        # Initialize backend
        self.passages = []
        self.metadata = []
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected vector store backend"""
        if self.backend == "faiss":
            try:
                import faiss
                # Use L2 distance by default
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info("Initialized FAISS backend")
            except ImportError:
                logger.error("FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu)")
                # Fall back to in-memory
                self.backend = "memory"
                self.vectors = []
        
        elif self.backend == "memory":
            # Simple in-memory storage
            self.vectors = []
            logger.info("Initialized in-memory vector backend")
        
        elif self.backend == "weaviate":
            try:
                import weaviate
                # Weaviate setup would go here - connection details would be passed in constructor
                logger.info("Weaviate backend initialized")
                self.weaviate_client = None  # Would be properly initialized with credentials
            except ImportError:
                logger.error("Weaviate not installed. Install with: pip install weaviate-client")
                # Fall back to in-memory
                self.backend = "memory"
                self.vectors = []
        
        elif self.backend == "pinecone":
            try:
                import pinecone
                # Pinecone setup would go here
                logger.info("Pinecone backend initialized")
                self.pinecone_index = None  # Would be properly initialized with credentials
            except ImportError:
                logger.error("Pinecone not installed. Install with: pip install pinecone-client")
                # Fall back to in-memory
                self.backend = "memory"
                self.vectors = []
        
        else:
            logger.warning(f"Unknown backend: {self.backend}, falling back to in-memory")
            self.backend = "memory"
            self.vectors = []
    
    def index_corpus(self, 
                    texts: List[str], 
                    metadata: Optional[List[Dict[str, Any]]] = None,
                    ids: Optional[List[str]] = None,
                    store_original: bool = True):
        """
        Index a corpus of texts
        
        Args:
            texts: List of text passages to index
            metadata: Optional metadata for each passage
            ids: Optional unique IDs for each passage
            store_original: Whether to store original text
        """
        if not texts:
            logger.warning("Empty corpus provided for indexing")
            return
        
        if not self.encoder:
            logger.error("Encoder not initialized, cannot index corpus")
            return
        
        # Prepare metadata if provided
        if metadata and len(metadata) != len(texts):
            logger.warning(f"Metadata length ({len(metadata)}) doesn't match texts ({len(texts)})")
            metadata = None
            
        # Generate metadata if not provided
        if metadata is None:
            metadata = [{"id": str(i), "position": i} for i in range(len(texts))]
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        elif len(ids) != len(texts):
            logger.warning(f"IDs length ({len(ids)}) doesn't match texts ({len(texts)})")
            ids = [str(i) for i in range(len(texts))]
        
        # Store original texts if requested
        if store_original:
            self.passages = texts
            self.metadata = metadata
        
        # Encode texts in batches
        try:
            all_vectors = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                batch_vectors = self.encoder.encode(batch, convert_to_numpy=True)
                all_vectors.append(batch_vectors)
            
            embeddings = np.vstack(all_vectors)
            
            # Index vectors based on backend
            self._add_to_backend(embeddings, ids, metadata)
            
            # Cache vectors if directory specified
            if self.cache_dir:
                self._cache_vectors(embeddings, ids, metadata)
                
            logger.info(f"Indexed {len(texts)} passages")
            
        except Exception as e:
            logger.error(f"Error encoding and indexing corpus: {e}")
    
    def _add_to_backend(self, embeddings: np.ndarray, ids: List[str], metadata: List[Dict[str, Any]]):
        """Add vectors to the backend"""
        if self.backend == "faiss":
            self.index.add(embeddings)
            
        elif self.backend == "memory":
            self.vectors = embeddings
            
        elif self.backend == "weaviate":
            # Example Weaviate batch import
            if hasattr(self, 'weaviate_client') and self.weaviate_client:
                # Implementation would depend on Weaviate schema
                pass
                
        elif self.backend == "pinecone":
            # Example Pinecone batch import
            if hasattr(self, 'pinecone_index') and self.pinecone_index:
                # Implementation would depend on Pinecone setup
                pass
    
    def _cache_vectors(self, embeddings: np.ndarray, ids: List[str], metadata: List[Dict[str, Any]]):
        """Cache vectors to disk"""
        if not self.cache_dir:
            return
            
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Save vectors
            np.save(os.path.join(self.cache_dir, "vectors.npy"), embeddings)
            
            # Save metadata and IDs
            with open(os.path.join(self.cache_dir, "metadata.json"), "w") as f:
                json.dump({
                    "ids": ids,
                    "metadata": metadata,
                    "model": self.model_name,
                    "dimension": self.dimension
                }, f)
                
            logger.info(f"Cached vectors to {self.cache_dir}")
            
        except Exception as e:
            logger.error(f"Error caching vectors: {e}")
    
    def load_cached(self, cache_dir: Optional[str] = None):
        """
        Load cached vectors from disk
        
        Args:
            cache_dir: Directory containing cached vectors (overrides self.cache_dir)
        
        Returns:
            Success flag
        """
        dir_path = cache_dir or self.cache_dir
        if not dir_path:
            logger.error("No cache directory specified")
            return False
            
        try:
            # Load vectors
            vectors_path = os.path.join(dir_path, "vectors.npy")
            if not os.path.exists(vectors_path):
                logger.error(f"Vectors file not found: {vectors_path}")
                return False
                
            embeddings = np.load(vectors_path)
            
            # Load metadata
            meta_path = os.path.join(dir_path, "metadata.json")
            if not os.path.exists(meta_path):
                logger.error(f"Metadata file not found: {meta_path}")
                return False
                
            with open(meta_path, "r") as f:
                data = json.load(f)
                ids = data.get("ids", [])
                metadata = data.get("metadata", [])
                
                # Verify model compatibility
                cached_model = data.get("model")
                cached_dim = data.get("dimension")
                
                if cached_model != self.model_name:
                    logger.warning(f"Cached model ({cached_model}) different from current ({self.model_name})")
                
                if cached_dim != self.dimension:
                    logger.error(f"Dimension mismatch: cached={cached_dim}, current={self.dimension}")
                    return False
            
            # Add to backend
            self._add_to_backend(embeddings, ids, metadata)
            
            # Store passages if available
            if "passages" in data:
                self.passages = data["passages"]
                
            self.metadata = metadata
            
            logger.info(f"Loaded {len(embeddings)} vectors from cache")
            return True
            
        except Exception as e:
            logger.error(f"Error loading cached vectors: {e}")
            return False
    
    def search(self, 
              query: str, 
              top_k: int = 5, 
              threshold: Optional[float] = None,
              include_scores: bool = False) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Search for passages similar to query
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Optional similarity threshold
            include_scores: Whether to include similarity scores in results
            
        Returns:
            List of passages or (passage, score) tuples
        """
        if not self.encoder:
            logger.error("Encoder not initialized, cannot search")
            return []
            
        try:
            # Encode query
            q_emb = self.encoder.encode(query, convert_to_numpy=True)
            
            # Search based on backend
            if self.backend == "faiss":
                return self._search_faiss(q_emb, top_k, threshold, include_scores)
                
            elif self.backend == "memory":
                return self._search_memory(q_emb, top_k, threshold, include_scores)
                
            elif self.backend == "weaviate":
                # Would implement Weaviate search
                logger.warning("Weaviate search not fully implemented")
                return []
                
            elif self.backend == "pinecone":
                # Would implement Pinecone search
                logger.warning("Pinecone search not fully implemented")
                return []
                
            else:
                logger.error(f"Unknown backend: {self.backend}")
                return []
                
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def _search_faiss(self, 
                     q_emb: np.ndarray, 
                     top_k: int, 
                     threshold: Optional[float], 
                     include_scores: bool) -> Union[List[str], List[Tuple[str, float]]]:
        """Search using FAISS backend"""
        if not hasattr(self, 'index') or self.index.ntotal == 0:
            return []
            
        # Add batch dimension if needed
        if len(q_emb.shape) == 1:
            q_emb = q_emb.reshape(1, -1)
            
        # Search index
        distances, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.passages):
                continue  # Invalid index
                
            distance = distances[0][i]
            
            # Apply threshold if specified
            if threshold is not None and distance > threshold:
                continue
                
            if include_scores:
                # Convert distance to similarity score (1.0 is best)
                similarity = 1.0 / (1.0 + distance)
                results.append((self.passages[idx], similarity))
            else:
                results.append(self.passages[idx])
                
        return results
    
    def _search_memory(self, 
                      q_emb: np.ndarray, 
                      top_k: int, 
                      threshold: Optional[float], 
                      include_scores: bool) -> Union[List[str], List[Tuple[str, float]]]:
        """Search using in-memory backend"""
        if not hasattr(self, 'vectors') or len(self.vectors) == 0:
            return []
            
        # Calculate cosine similarity
        scores = []
        q_emb_norm = q_emb / np.linalg.norm(q_emb)
        
        for i, vec in enumerate(self.vectors):
            vec_norm = vec / np.linalg.norm(vec)
            similarity = np.dot(q_emb_norm, vec_norm)
            
            # Apply threshold if specified
            if threshold is not None and similarity < threshold:
                continue
                
            scores.append((i, similarity))
            
        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = []
        for idx, score in scores[:top_k]:
            if include_scores:
                results.append((self.passages[idx], score))
            else:
                results.append(self.passages[idx])
                
        return results
    
    def search_with_metadata(self, 
                           query: str, 
                           top_k: int = 5, 
                           threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for passages and return with metadata
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Optional similarity threshold
            
        Returns:
            List of dicts with passage, metadata and score
        """
        # Get search results with scores
        results = self.search(query, top_k, threshold, include_scores=True)
        
        # Format results with metadata
        output = []
        for i, (passage, score) in enumerate(results):
            # Find matching passage in original list
            try:
                idx = self.passages.index(passage)
                metadata = self.metadata[idx] if idx < len(self.metadata) else {}
            except ValueError:
                metadata = {}
                
            output.append({
                "passage": passage,
                "score": score,
                "metadata": metadata,
                "rank": i + 1
            })
            
        return output