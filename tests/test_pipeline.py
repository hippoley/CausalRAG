"""
Basic tests for the CausalRAG pipeline.
"""

import unittest
import os
import tempfile
import shutil

from causalrag import CausalRAGPipeline

class TestPipeline(unittest.TestCase):
    """Test the core pipeline functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_docs = [
            "Climate change causes rising sea levels, which leads to coastal flooding.",
            "Deforestation reduces carbon capture, increasing atmospheric CO2.",
        ]
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        pipeline = CausalRAGPipeline()
        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(pipeline.graph_builder)
        self.assertIsNotNone(pipeline.vector_retriever)
    
    def test_document_indexing(self):
        """Test document indexing functionality."""
        pipeline = CausalRAGPipeline()
        result = pipeline.index(self.test_docs)
        
        # Check that documents were indexed
        self.assertEqual(len(self.test_docs), result.get('documents', 0))
        
        # Verify graph has nodes
        graph = pipeline.graph_builder.get_graph()
        self.assertGreater(graph.number_of_nodes(), 0)
    
    def test_query_execution(self):
        """Test basic query execution."""
        pipeline = CausalRAGPipeline()
        pipeline.index(self.test_docs)
        
        result = pipeline.run("What causes coastal flooding?")
        
        # Check result structure
        self.assertIn("answer", result)
        self.assertIn("context", result)
        
        # Answer should be a non-empty string
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 0)
    
    def test_save_and_load(self):
        """Test saving and loading functionality."""
        # Create and index with a pipeline
        pipeline1 = CausalRAGPipeline()
        pipeline1.index(self.test_docs)
        
        # Save to temp directory
        save_path = os.path.join(self.temp_dir, "causalrag_test")
        os.makedirs(save_path, exist_ok=True)
        pipeline1.save(save_path)
        
        # Create a new pipeline and load the saved state
        pipeline2 = CausalRAGPipeline(
            graph_path=os.path.join(save_path, "causal_graph.json"),
            index_path=save_path
        )
        
        # Test query on loaded pipeline
        result = pipeline2.run("What is climate change?")
        self.assertIn("answer", result)

if __name__ == "__main__":
    unittest.main() 