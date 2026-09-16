from __future__ import annotations

import logging
from typing import List, Set, Tuple

import networkx as nx
import numpy as np

from causalrag.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


class CausalPathRetriever:
    """Retrieve query-relevant causal nodes and paths from a graph."""

    def __init__(self, builder: "CausalGraphBuilder") -> None:
        self.graph = builder.get_graph()
        self.node_embeddings = builder.node_embeddings
        self.node_text = builder.node_text
        self.embedding_provider = getattr(builder, "embedding_provider", None)
        self.builder = builder

    def retrieve_nodes(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[str, float]]:
        if self.embedding_provider is None or not self.node_embeddings:
            logger.warning("Embedding provider or node embeddings not available")
            return []
        try:
            query_embedding = self.embedding_provider.embed_one(query)
            scores = []
            for node_id, embedding in self.node_embeddings.items():
                score = cosine_similarity(query_embedding, embedding)
                if score >= threshold:
                    scores.append((node_id, score))
            return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
        except Exception as exc:
            logger.error("Error retrieving causal nodes: %s", exc)
            return []

    def retrieve_path_nodes(
        self,
        query: str,
        top_k: int = 5,
        max_hops: int = 2,
        include_similar: bool = True,
    ) -> List[str]:
        top_nodes = self.retrieve_nodes(query, top_k=top_k)
        seed_nodes = [node_id for node_id, _score in top_nodes]
        path_nodes: Set[str] = set(seed_nodes)

        for node_id in seed_nodes:
            path_nodes.update(self._get_descendants(node_id, max_hops))
            path_nodes.update(self._get_ancestors(node_id, max_hops))

        if include_similar and seed_nodes and self.node_embeddings:
            seed_embeddings = [
                np.asarray(self.node_embeddings[node_id], dtype=np.float32)
                for node_id in seed_nodes
                if node_id in self.node_embeddings
            ]
            if seed_embeddings:
                centroid = np.mean(np.vstack(seed_embeddings), axis=0)
                for node_id, embedding in self.node_embeddings.items():
                    if node_id not in path_nodes and cosine_similarity(centroid, embedding) > 0.8:
                        path_nodes.add(node_id)

        return list(path_nodes)

    def _get_descendants(self, node: str, max_hops: int) -> Set[str]:
        descendants: Set[str] = set()
        current = {node}
        for _ in range(max_hops):
            next_nodes: Set[str] = set()
            for item in current:
                if item in self.graph:
                    next_nodes.update(self.graph.successors(item))
            descendants.update(next_nodes)
            current = next_nodes
            if not current:
                break
        return descendants

    def _get_ancestors(self, node: str, max_hops: int) -> Set[str]:
        ancestors: Set[str] = set()
        current = {node}
        for _ in range(max_hops):
            next_nodes: Set[str] = set()
            for item in current:
                if item in self.graph:
                    next_nodes.update(self.graph.predecessors(item))
            ancestors.update(next_nodes)
            current = next_nodes
            if not current:
                break
        return ancestors

    def retrieve_paths(
        self,
        query: str,
        max_paths: int = 5,
        min_path_length: int = 2,
        max_path_length: int = 4,
    ) -> List[List[str]]:
        relevant_nodes = self.retrieve_path_nodes(query, top_k=5, max_hops=1)
        if len(relevant_nodes) < 2:
            return []

        paths = []
        seen = set()
        for index, source in enumerate(relevant_nodes):
            for target in relevant_nodes[index + 1 :]:
                if source == target:
                    continue
                for start, end in ((source, target), (target, source)):
                    try:
                        for path in nx.all_simple_paths(
                            self.graph, start, end, cutoff=max_path_length
                        ):
                            if len(path) < min_path_length:
                                continue
                            key = tuple(path)
                            if key in seen:
                                continue
                            seen.add(key)
                            text_path = [self.node_text.get(node, node) for node in path]
                            paths.append((path, text_path))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

        paths.sort(key=lambda item: len(item[0]))
        return [text_path for _path, text_path in paths[:max_paths]]

    def get_causal_explanation(self, query: str) -> str:
        paths = self.retrieve_paths(query, max_paths=3)
        if not paths:
            return "No relevant causal relationships found."
        lines = [f"Causal relationships relevant to '{query}':", ""]
        lines.extend(f"{index}. {' → '.join(path)}" for index, path in enumerate(paths, start=1))
        return "\n".join(lines)

    def highlight_subgraph(self, query: str) -> nx.DiGraph:
        return self.graph.subgraph(self.retrieve_path_nodes(query))
