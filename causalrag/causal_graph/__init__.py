"""Causal graph extraction and retrieval components.

Core graph building and path retrieval stay lightweight. Visualization helpers
are loaded only when explicitly requested.
"""

from .builder import CausalGraphBuilder
from .retriever import CausalPathRetriever


def __getattr__(name):
    if name == "CausalGraphExplainer":
        try:
            from .explainer import CausalGraphExplainer
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "Graph visualization requires optional visualization dependencies. "
                "Install them with: pip install 'causalrag[visualization]'"
            ) from exc
        return CausalGraphExplainer
    raise AttributeError("module 'causalrag.causal_graph' has no attribute %r" % name)


__all__ = ["CausalGraphBuilder", "CausalPathRetriever"]
