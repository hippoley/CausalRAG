# retriever/base.py
# Base class and interfaces for CausalRAG retrievers

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple


class BaseRetriever(ABC):
    """Base class for all retrieval components in CausalRAG"""
    
    def __init__(self, **kwargs):
        """
        Initialize the retriever with common parameters
        
        Args:
            **kwargs: Additional retriever-specific parameters
        """
        self.name = self.__class__.__name__
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve passages relevant to the query
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional retrieval parameters
            
        Returns:
            List of retrieved passages with metadata
        """
        pass
    
    @abstractmethod
    def index_corpus(self, documents: List[Union[str, Dict[str, Any]]], **kwargs) -> bool:
        """
        Index a corpus of documents for retrieval
        
        Args:
            documents: List of documents (text or dictionaries with text and metadata)
            **kwargs: Additional indexing parameters
            
        Returns:
            True if indexing was successful
        """
        pass
    
    def save_index(self, path: str) -> bool:
        """
        Save the retriever's index to a file
        
        Args:
            path: Path to save the index
            
        Returns:
            True if saving was successful
        """
        # Default implementation - override in subclasses
        return False
    
    def load_index(self, path: str) -> bool:
        """
        Load the retriever's index from a file
        
        Args:
            path: Path to load the index from
            
        Returns:
            True if loading was successful
        """
        # Default implementation - override in subclasses
        return False
    
    def process_query(self, query: str) -> str:
        """
        Preprocess the query before retrieval
        
        Args:
            query: Raw query string
            
        Returns:
            Processed query
        """
        # Default implementation - simple cleanup
        return query.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever
        
        Returns:
            Dictionary with retriever statistics
        """
        # Default implementation - basic info
        return {
            "name": self.name,
            "type": "BaseRetriever",
        }


class EmbeddingRetriever(BaseRetriever):
    """Base class for retrievers that use embeddings"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", **kwargs):
        """
        Initialize an embedding-based retriever
        
        Args:
            embedding_model: Name or path of embedding model
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
    
    @abstractmethod
    def encode_query(self, query: str) -> Any:
        """
        Encode a query into an embedding vector
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        pass
    
    @abstractmethod
    def encode_documents(self, documents: List[str]) -> List[Any]:
        """
        Encode documents into embedding vectors
        
        Args:
            documents: List of document texts
            
        Returns:
            List of document embeddings
        """
        pass


class KeywordRetriever(BaseRetriever):
    """Base class for keyword-based retrievers"""
    
    def __init__(self, tokenizer=None, **kwargs):
        """
        Initialize a keyword-based retriever
        
        Args:
            tokenizer: Optional custom tokenizer
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words/tokens
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.tokenizer:
            return self.tokenizer(text)
        else:
            # Default simple tokenization
            return text.lower().split() 