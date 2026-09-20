"""Text-generation adapters for Branchpoint.

The lightweight agent runtime only needs ``LLMInterface``. Prompt-building
helpers are loaded lazily because they belong to the optional legacy retrieval path.
"""

from .llm_interface import LLMInterface


def __getattr__(name):
    if name in {"build_prompt", "PromptBuilder"}:
        try:
            from .prompt_builder import build_prompt, PromptBuilder
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "PromptBuilder belongs to the optional retrieval layer. Install it "
                "with: pip install 'branchpoint[retrieval]'"
            ) from exc
        return {"build_prompt": build_prompt, "PromptBuilder": PromptBuilder}[name]
    raise AttributeError("module 'branchpoint.generator' has no attribute %r" % name)


__all__ = ["LLMInterface"]
