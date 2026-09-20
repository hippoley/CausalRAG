"""External interfaces for CausalRAG.

The v0.3 agent API is available from ``causalrag.interface.agent_api`` without
loading the legacy RAG server. The old ``app`` surface is kept lazily for
compatibility and requires both API and RAG dependencies.
"""


def __getattr__(name):
    if name == "app":
        try:
            from .api import app
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "The legacy RAG API requires optional API and RAG dependencies. "
                "Install them with: pip install 'causalrag[api,rag]'"
            ) from exc
        return app
    raise AttributeError("module 'causalrag.interface' has no attribute %r" % name)


__all__ = []
