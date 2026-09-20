"""External interfaces for Branchpoint.

The v0.3 agent API is available from ``branchpoint.interface.agent_api`` without
loading the legacy retrieval server. The old ``app`` surface is kept lazily for
compatibility and requires both API and retrieval dependencies.
"""


def __getattr__(name):
    if name == "app":
        try:
            from .api import app
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "The legacy retrieval API requires optional API and retrieval dependencies. "
                "Install them with: pip install 'branchpoint[api,retrieval]'"
            ) from exc
        return app
    raise AttributeError("module 'branchpoint.interface' has no attribute %r" % name)


__all__ = []
