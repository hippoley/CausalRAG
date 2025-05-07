# reranker/causal_path.py
# Reranks candidate text passages based on their overlap with causal path nodes

class CausalPathReranker:
    def __init__(self, retriever: CausalPathRetriever):
        self.retriever = retriever

    def rerank(self, query: str, candidates: List[str]) -> List[str]:
        path_nodes = self.retriever.retrieve_path_nodes(query)
        scored = []
        for passage in candidates:
            match_score = sum(1 for node in path_nodes if node.lower() in passage.lower())
            scored.append((match_score, passage))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]