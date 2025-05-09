import unittest
import os
import tempfile
import json
import networkx as nx
from causalrag.causal_graph.builder import CausalGraphBuilder
from causalrag.causal_graph.retriever import CausalPathRetriever

class TestCausalGraph(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.builder = CausalGraphBuilder()
        self.test_documents = [
            "Climate change causes rising sea levels, which leads to coastal flooding.",
            "Deforestation reduces carbon capture, increasing atmospheric CO2.",
            "Higher CO2 levels accelerate global warming, exacerbating climate change.",
            "Coastal flooding damages infrastructure and causes population displacement."
        ]
        # Build graph for testing
        self.builder.index_documents(self.test_documents)
        self.retriever = CausalPathRetriever(self.builder)
        
    def test_graph_construction(self):
        """Test if the graph is properly constructed from documents"""
        # Check if the graph is created and has nodes
        self.assertIsInstance(self.builder.graph, nx.DiGraph)
        self.assertGreater(len(self.builder.graph.nodes), 0)
        self.assertGreater(len(self.builder.graph.edges), 0)
        
        # Check if specific causal relationships are captured
        self.assertTrue(self.builder.has_causal_relation("climate change", "rising sea levels"))
        self.assertTrue(self.builder.has_causal_relation("rising sea levels", "coastal flooding"))
        self.assertTrue(self.builder.has_causal_relation("coastal flooding", "infrastructure damage"))
        
    def test_save_load_graph(self):
        """Test saving and loading graph functionality"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            # Save the graph
            self.builder.save_graph(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new builder and load graph
            new_builder = CausalGraphBuilder()
            new_builder.load_graph(tmp_path)
            
            # Compare graphs
            self.assertEqual(len(self.builder.graph.nodes), len(new_builder.graph.nodes))
            self.assertEqual(len(self.builder.graph.edges), len(new_builder.graph.edges))
            self.assertTrue(new_builder.has_causal_relation("climate change", "rising sea levels"))
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    def test_causal_path_retrieval(self):
        """Test retrieval of causal paths relevant to a query"""
        # Test path retrieval
        paths = self.retriever.retrieve_paths("What are the effects of climate change?", top_k=3)
        
        # Check if paths are retrieved
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)
        
        # Check if paths start with climate change
        has_climate_path = False
        for path in paths:
            if "climate change" in path[0].lower():
                has_climate_path = True
                break
                
        self.assertTrue(has_climate_path, "Should find paths starting with climate change")
        
    def test_query_relevance(self):
        """Test node relevance to query"""
        query = "How does climate change affect coastal infrastructure?"
        relevant_nodes = self.retriever.get_relevant_nodes(query, top_k=5)
        
        # Check if relevant nodes are retrieved
        self.assertIsInstance(relevant_nodes, list)
        self.assertGreater(len(relevant_nodes), 0)
        
        # Check if specific relevant nodes are present
        relevant_terms = ["climate change", "coastal flooding", "infrastructure"]
        for term in relevant_terms:
            found = False
            for node_info in relevant_nodes:
                if term in node_info["node"].lower():
                    found = True
                    break
            self.assertTrue(found, f"Should find {term} in relevant nodes")

if __name__ == '__main__':
    unittest.main() 