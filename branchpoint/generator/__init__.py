"""Text-generation adapters for CausalRAG.

The lightweight agent runtime only needs ``LLMInterface``. Prompt-building
helpers are loaded lazily because they belong to the optional legacy RAG path.
"""

from .llm_interface import LLMInterface


def __getattr__(name):
    if name in {"build_prompt", "PromptBuilder"}:
        try:
            from .prompt_builder import build_prompt, PromptBuilder
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "PromptBuilder belongs to the optional RAG layer. Install it "
                "with: pip install 'causalrag[rag]'"
            ) from exc
        return {"build_prompt": build_prompt, "PromptBuilder": PromptBuilder}[name]
    raise AttributeError("module 'causalrag.generator' has no attribute %r" % name)


__all__ = ["LLMInterface"]
