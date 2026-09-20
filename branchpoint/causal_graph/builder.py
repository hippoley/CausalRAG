from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from branchpoint.embeddings import EmbeddingProvider, cosine_similarity, create_embedding_provider

logger = logging.getLogger(__name__)


class CausalTripleExtractor:
    """Extract directional causal claims with rule, LLM, or hybrid methods."""

    def __init__(self, method: str = "hybrid", llm_interface: Any = None) -> None:
        if method not in {"rule", "llm", "hybrid"}:
            raise ValueError("extractor method must be rule, llm, or hybrid")
        self.method = method
        self.llm_interface = llm_interface
        self.causal_patterns = [
            r"([\w\s-]+?)\s+causes\s+([\w\s-]+)",
            r"([\w\s-]+?)\s+leads\s+to\s+([\w\s-]+)",
            r"([\w\s-]+?)\s+results\s+in\s+([\w\s-]+)",
            r"([\w\s-]+?)\s+contributes\s+to\s+([\w\s-]+)",
            r"([\w\s-]+?)\s+triggers\s+([\w\s-]+)",
            r"([\w\s-]+?)\s+drives\s+([\w\s-]+)",
            r"if\s+([\w\s-]+?),\s*then\s+([\w\s-]+)",
        ]

    def extract(self, text: str) -> List[Tuple[str, str, float]]:
        if self.method == "rule":
            return self._rule_based_extraction(text)
        if self.method == "llm":
            return self._llm_based_extraction(text)
        return self._deduplicate_triples(
            self._rule_based_extraction(text) + self._llm_based_extraction(text)
        )

    def _rule_based_extraction(self, text: str) -> List[Tuple[str, str, float]]:
        triples: List[Tuple[str, str, float]] = []
        clean_text = text.replace("\n", " ").strip()
        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        for sentence in sentences:
            for pattern in self.causal_patterns:
                for match in re.finditer(pattern, sentence, re.IGNORECASE):
                    cause = self._normalize_text(match.group(1))
                    effect = self._normalize_text(match.group(2))
                    if len(cause) > 2 and len(effect) > 2:
                        triples.append((cause, effect, 0.8))
        return self._deduplicate_triples(triples)

    def _llm_based_extraction(self, text: str) -> List[Tuple[str, str, float]]:
        if self.llm_interface is None:
            return []
        triples: List[Tuple[str, str, float]] = []
        for chunk in self._split_text_into_chunks(text):
            prompt = (
                "Extract causal claims from the text as a JSON array. Each item must "
                "contain cause, effect, confidence (0..1). Include directional causal "
                "claims only, not correlation or mere temporal order. Return JSON only.\n\n"
                f"TEXT:\n{chunk}"
            )
            try:
                response = self.llm_interface.generate(
                    prompt, temperature=0.1, json_mode=True
                )
                triples.extend(self._parse_llm_response(response))
            except Exception as exc:
                logger.warning("Causal LLM extraction failed: %s", exc)
        return self._deduplicate_triples(triples)

    def _parse_llm_response(
        self, response: Union[str, List[Any], Dict[str, Any]]
    ) -> List[Tuple[str, str, float]]:
        if isinstance(response, str):
            raw = response.strip()
            start = raw.find("[")
            end = raw.rfind("]")
            if start < 0 or end < start:
                return []
            try:
                payload = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []
        elif isinstance(response, list):
            payload = response
        elif isinstance(response, dict):
            payload = response.get("triples") or response.get("data") or [response]
        else:
            return []

        triples = []
        for item in payload if isinstance(payload, list) else []:
            if not isinstance(item, dict):
                continue
            cause = self._normalize_text(str(item.get("cause", "")))
            effect = self._normalize_text(str(item.get("effect", "")))
            if not cause or not effect:
                continue
            try:
                confidence = float(item.get("confidence", 0.7))
            except (TypeError, ValueError):
                confidence = 0.7
            triples.append((cause, effect, max(0.0, min(1.0, confidence))))
        return triples

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text.strip().strip("\"'"))
        return re.sub(r"[.,;:!?]+$", "", text)

    @staticmethod
    def _split_text_into_chunks(text: str, max_length: int = 3000) -> List[str]:
        if len(text) <= max_length:
            return [text]
        chunks = []
        cursor = 0
        while cursor < len(text):
            end = min(len(text), cursor + max_length)
            if end < len(text):
                split = text.rfind(" ", cursor, end)
                if split > cursor:
                    end = split
            chunks.append(text[cursor:end].strip())
            cursor = end
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _deduplicate_triples(
        triples: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float]]:
        best: Dict[Tuple[str, str], Tuple[str, str, float]] = {}
        for cause, effect, confidence in triples:
            key = (cause.casefold(), effect.casefold())
            if key not in best or confidence > best[key][2]:
                best[key] = (cause, effect, confidence)
        return list(best.values())


class CausalGraphBuilder:
    """Build and maintain a causal graph using an injectable embedding provider."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        normalize_nodes: bool = True,
        confidence_threshold: float = 0.5,
        extractor_method: str = "hybrid",
        llm_interface: Any = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_provider_name: str = "openai",
        embedding_api_key: Optional[str] = None,
        node_similarity_threshold: float = 0.85,
    ) -> None:
        self.graph = nx.DiGraph()
        self.node_text: Dict[str, str] = {}
        self.node_variants: Dict[str, List[str]] = {}
        self.node_embeddings: Dict[str, np.ndarray] = {}
        self.confidence_threshold = confidence_threshold
        self.normalize_nodes = normalize_nodes
        self.node_similarity_threshold = node_similarity_threshold
        self.embedding_provider_name = embedding_provider_name
        self.model_name = model_name

        self.embedding_provider = embedding_provider
        if self.embedding_provider is None and normalize_nodes:
            self.embedding_provider = create_embedding_provider(
                provider=embedding_provider_name,
                model=model_name,
                api_key=embedding_api_key,
            )

        # Compatibility alias for callers that previously accessed builder.encoder.
        self.encoder = self.embedding_provider
        self.extractor = CausalTripleExtractor(
            method=extractor_method, llm_interface=llm_interface
        )

    def add_triples(self, triples: List[Tuple[str, str, Optional[float]]]) -> None:
        for triple in triples:
            if len(triple) < 2:
                continue
            cause, effect = str(triple[0]), str(triple[1])
            confidence = (
                float(triple[2])
                if len(triple) > 2 and triple[2] is not None
                else 1.0
            )
            if confidence < self.confidence_threshold:
                continue

            if self.normalize_nodes:
                cause_id = self._get_or_create_node(cause)
                effect_id = self._get_or_create_node(effect)
            else:
                cause_id, effect_id = cause, effect
                self.node_text.setdefault(cause_id, cause)
                self.node_text.setdefault(effect_id, effect)
                self._ensure_embedding(cause_id)
                self._ensure_embedding(effect_id)

            self.graph.add_edge(cause_id, effect_id, weight=confidence)

    def _ensure_embedding(self, node_id: str) -> None:
        if self.embedding_provider is None or node_id in self.node_embeddings:
            return
        text = self.node_text.get(node_id, node_id)
        self.node_embeddings[node_id] = self.embedding_provider.embed_one(text)

    def _get_or_create_node(self, text: str) -> str:
        normalized = CausalTripleExtractor._normalize_text(text)
        if self.embedding_provider is None:
            self.node_text.setdefault(normalized, normalized)
            return normalized

        embedding = self.embedding_provider.embed_one(normalized)
        best_match: Optional[str] = None
        best_score = -1.0
        for node_id, existing_embedding in self.node_embeddings.items():
            score = cosine_similarity(embedding, existing_embedding)
            if score >= self.node_similarity_threshold and score > best_score:
                best_match, best_score = node_id, score

        if best_match is not None:
            variants = self.node_variants.setdefault(best_match, [])
            if normalized != self.node_text.get(best_match) and normalized not in variants:
                variants.append(normalized)
            return best_match

        node_id = normalized
        self.node_text[node_id] = normalized
        self.node_embeddings[node_id] = np.asarray(embedding, dtype=np.float32)
        return node_id

    def index_documents(
        self,
        docs: List[str],
        batch_size: int = 5,
        show_progress: bool = True,
    ) -> int:
        initial_edges = self.graph.number_of_edges()
        iterable = docs
        if show_progress:
            try:
                from tqdm import tqdm

                iterable = tqdm(docs, desc="Building causal graph")
            except ImportError:
                pass
        for document in iterable:
            if document and len(document.strip()) >= 3:
                self.add_triples(self.extractor.extract(document))
        return self.graph.number_of_edges() - initial_edges

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def get_node_variants(self, node_id: str) -> List[str]:
        primary = self.node_text.get(node_id, node_id)
        return [primary] + list(self.node_variants.get(node_id, []))

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        return self.node_embeddings.get(node_id)

    def describe_graph(self) -> str:
        if self.graph.number_of_edges() == 0:
            return "Empty causal graph (no causal relationships found)"
        lines = []
        for cause, effect, data in self.graph.edges(data=True):
            lines.append(
                f"{self.node_text.get(cause, cause)} → {self.node_text.get(effect, effect)} "
                f"(confidence: {float(data.get('weight', 1.0)):.2f})"
            )
        return "\n".join(lines)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "nodes": self.node_text,
            "variants": self.node_variants,
            "edges": [
                (cause, effect, dict(attributes))
                for cause, effect, attributes in self.graph.edges(data=True)
            ],
            "embedding": {
                "provider": self.embedding_provider_name,
                "model": getattr(self.embedding_provider, "model_name", self.model_name),
            },
        }
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.graph = nx.DiGraph()
            self.node_text = data.get("nodes", {})
            self.node_variants = data.get("variants", {})
            self.node_embeddings = {}
            for cause, effect, attributes in data.get("edges", []):
                self.graph.add_edge(cause, effect, **attributes)
            if self.embedding_provider is not None and self.node_text:
                node_ids = list(self.node_text)
                matrix = self.embedding_provider.embed(
                    [self.node_text[node_id] for node_id in node_ids]
                )
                self.node_embeddings = {
                    node_id: np.asarray(matrix[index], dtype=np.float32)
                    for index, node_id in enumerate(node_ids)
                }
            return True
        except Exception as exc:
            logger.error("Error loading graph from %s: %s", filepath, exc)
            return False

    # Historical compatibility aliases.
    save_graph = save
    load_graph = load

    def get_extraction_statistics(self) -> Dict[str, Any]:
        weights = [
            float(data.get("weight", 1.0))
            for _cause, _effect, data in self.graph.edges(data=True)
        ]
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "average_confidence": float(np.mean(weights)) if weights else 0.0,
            "embedding_provider": self.embedding_provider_name,
            "embedding_model": getattr(
                self.embedding_provider, "model_name", self.model_name
            ),
        }
