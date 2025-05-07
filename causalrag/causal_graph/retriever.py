# causal_graph/retriever.py
# Responsible for finding query-relevant nodes or subgraphs from causal graph

class CausalPathRetriever:
    def __init__(self, builder: CausalGraphBuilder):
        self.graph = builder.get_graph()
        self.node_embeddings = builder.node_embeddings
        self.encoder = builder.encoder

    def retrieve_path_nodes(self, query: str, top_k: int = 3, max_hops: int = 2) -> List[str]:
        q_emb = self.encoder.encode(query, convert_to_tensor=True)
        sims = {
            node: util.pytorch_cos_sim(q_emb, emb).item()
            for node, emb in self.node_embeddings.items()
        }
        top_nodes = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_k]
        path_nodes = set()
        for node, _ in top_nodes:
            path_nodes.add(node)
            for _ in range(max_hops):
                path_nodes.update(self.graph.successors(node))
                path_nodes.update(self.graph.predecessors(node))
        return list(path_nodes)

    def highlight_path(self, node_list: List[str]) -> List[Tuple[str, str]]:
        return [(a, b) for a, b in self.graph.edges() if a in node_list and b in node_list]
