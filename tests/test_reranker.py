import unittest
from unittest.mock import MagicMock, patch
from causalrag.reranker.base import BaseReranker
from causalrag.reranker.causal_path import CausalPathReranker

class TestRerankers(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create mock causal path retriever
        self.mock_retriever = MagicMock()
        
        # Sample paths and nodes that our mock will return
        self.test_paths = [
            ["climate change", "rising sea levels", "coastal flooding"],
            ["deforestation", "carbon capture reduction", "CO2 increase"]
        ]
        
        self.test_nodes = ["climate change", "rising sea levels", "coastal flooding", 
                          "deforestation", "carbon capture", "CO2"]
        
        # Configure mock to return our test data
        self.mock_retriever.retrieve_paths.return_value = self.test_paths
        self.mock_retriever.retrieve_path_nodes.return_value = self.test_nodes
        
        # Create reranker with mock retriever
        self.reranker = CausalPathReranker(self.mock_retriever)
        
        # Sample documents for reranking
        self.test_docs = [
            "Climate patterns have been changing in recent decades.",
            "Rising sea levels caused by climate change lead to coastal flooding.",
            "Deforestation reduces the ability of forests to capture carbon.",
            "Coral reefs are affected by ocean temperature changes.",
            "Coastal cities are implementing flood protection measures."
        ]
    
    def test_base_reranker(self):
        """Test basic functionality of the abstract base reranker"""
        # BaseReranker is abstract, so we test through a minimal concrete implementation
        class SimpleReranker(BaseReranker):
            def score_document(self, query, doc):
                # Simple implementation that just checks for word overlap
                query_words = set(query.lower().split())
                doc_words = set(doc.lower().split())
                return len(query_words.intersection(doc_words))
        
        reranker = SimpleReranker()
        query = "climate change effects"
        
        # Test reranking
        ranked = reranker.rerank(query, self.test_docs)
        
        # Verify result is a list of the same length
        self.assertEqual(len(ranked), len(self.test_docs))
        
        # First document should have "climate" in it
        self.assertIn("climate", ranked[0].lower())
    
    def test_causal_path_reranker(self):
        """Test the causal path aware reranker"""
        query = "How does climate change affect coastal areas?"
        
        # Verify our mock was configured correctly
        self.mock_retriever.retrieve_paths.assert_not_called()
        
        # Perform reranking
        ranked_docs = self.reranker.rerank(query, self.test_docs)
        
        # Verify mock was called with our query
        self.mock_retriever.retrieve_paths.assert_called_with(query, max_paths=3)
        
        # We expect document with causal path elements to be ranked higher
        # The document with climate change->sea levels->flooding should be first
        self.assertIn("rising sea levels", ranked_docs[0].lower())
        self.assertIn("coastal flooding", ranked_docs[0].lower())
        
    def test_document_scoring(self):
        """Test the document scoring mechanism"""
        query = "What causes coastal flooding?"
        
        # Configure our mock
        self.mock_retriever.retrieve_paths.return_value = [
            ["climate change", "rising sea levels", "coastal flooding"]
        ]
        
        # Test scoring of individual documents
        doc1 = "Climate change is causing rising sea levels."
        doc2 = "Coastal flooding is a problem in many cities."
        doc3 = "Climate change leads to rising sea levels, which causes coastal flooding."
        
        # Score each document
        score1 = self.reranker.score_document(query, doc1)
        score2 = self.reranker.score_document(query, doc2)
        score3 = self.reranker.score_document(query, doc3)
        
        # Document with complete causal path should score highest
        self.assertGreater(score3, score1)
        self.assertGreater(score3, score2)
        
    def test_reranking_with_weights(self):
        """Test reranking with custom weights"""
        # Create reranker with custom weights
        reranker = CausalPathReranker(
            self.mock_retriever,
            node_match_weight=1.0,
            path_match_weight=3.0  # Heavily favor path matches
        )
        
        query = "How does climate change cause flooding?"
        
        # Configure mock return value
        self.mock_retriever.retrieve_paths.return_value = [
            ["climate change", "rising sea levels", "coastal flooding"]
        ]
        
        # Sample documents
        docs = [
            "Climate change and flooding are environmental issues.",  # Contains nodes but no path
            "Rising sea levels are causing coastal flooding worldwide.",  # Contains partial path
            "Climate change leads to rising sea levels, causing coastal flooding."  # Complete path
        ]
        
        # Rerank
        ranked = reranker.rerank(query, docs)
        
        # With high path weight, complete path should be first
        self.assertIn("climate change leads to rising sea levels", ranked[0].lower())

if __name__ == '__main__':
    unittest.main() 