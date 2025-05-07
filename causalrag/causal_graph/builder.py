# causal_graph/builder.py
# Responsible for parsing documents into causal triples and building the causal graph

import networkx as nx
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer, util

# Stub extractor — replace with LLM call or rule-based method
def extract_causal_triples(text: str) -> List[Tuple[str, str]]:
    """
    Extracts causal pairs from text. Placeholder method.
    :param text: Input text document
    :return: List of (cause, effect) tuples
    """
    if "influence" in text.lower():
        return [("Influence Tactics", "Buyer Attention"), ("Buyer Attention", "Contract Award")]
    return []

class CausalGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_text = {}  # Optional mapping to full surface text
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.node_embeddings = {}

    def add_triples(self, triples: List[Tuple[str, str]]):
        for cause, effect in triples:
            self.graph.add_edge(cause, effect)
            for node in [cause, effect]:
                if node not in self.node_embeddings:
                    self.node_embeddings[node] = self.encoder.encode(node, convert_to_tensor=True)
                    self.node_text[node] = node

    def index_documents(self, docs: List[str]):
        for doc in docs:
            triples = extract_causal_triples(doc)
            self.add_triples(triples)

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def get_embedding(self, node: str):
        return self.node_embeddings.get(node)

    def describe_graph(self) -> str:
        return "\n".join([f"{a} → {b}" for a, b in self.graph.edges()])

