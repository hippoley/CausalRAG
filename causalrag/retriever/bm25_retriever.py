# retriever/bm25_retriever.py
# BM25-based keyword retrieval component

from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import re
import math
import numpy as np
from collections import Counter
from .base import BaseRetriever

logger = logging.getLogger(__name__)

class BM25Retriever(BaseRetriever):
    """
    Retriever using BM25 algorithm for keyword-based search.
    Good baseline and complement to semantic search.
    """
    
    def __init__(self, 
                k1: float = 1.5, 
                b: float = 0.75,
                name: str = "bm25"):
        """
        Initialize BM25 retriever
        
        Args:
            k1: Term saturation parameter
            b: Length normalization parameter
            name: Identifier for this retriever
        """
        super().__init__(name=name)
        self.k1 = k1
        self.b = b
        self.passages = []
        self.doc_freqs = Counter()  # df
        self.term_freqs = []  # tf
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.vocabulary = set()
        self.num_docs = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Convert text to lowercase tokens"""
        text = text.lower()
        # Simple tokenization - split on non-alphanumeric
        return re.findall(r'\w+', text)
    
    def index_documents(self, texts: List[str]):
        """
        Index documents for BM25 retrieval
        
        Args:
            texts: List of text passages to index
        """
        self.passages = texts
        self.num_docs = len(texts)
        self.doc_freqs = Counter()
        self.term_freqs = []
        self.doc_lengths = []
        self.vocabulary = set()