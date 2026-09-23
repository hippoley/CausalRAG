from .openai_agents import (
    ApprovalOutcome,
    OpenAIAgentsApprovalAdapter,
    OpenAIAgentsResolution,
    OpenAIAgentsToolDecision,
)

__all__ = [
    "ApprovalOutcome",
    "OpenAIAgentsApprovalAdapter",
    "OpenAIAgentsResolution",
    "OpenAIAgentsToolDecision",
    "LangChainBranchpointError",
    "LangChainBranchpointMiddleware",
    "langchain_branchpoint_stack",
    "langchain_human_in_the_loop",
]


_LANGCHAIN_EXPORTS = {
    "LangChainBranchpointError",
    "LangChainBranchpointMiddleware",
    "langchain_branchpoint_stack",
    "langchain_human_in_the_loop",
}


def __getattr__(name):
    if name in _LANGCHAIN_EXPORTS:
        from .langchain import (
            LangChainBranchpointError,
            LangChainBranchpointMiddleware,
            langchain_branchpoint_stack,
            langchain_human_in_the_loop,
        )

        return {
            "LangChainBranchpointError": LangChainBranchpointError,
            "LangChainBranchpointMiddleware": LangChainBranchpointMiddleware,
            "langchain_branchpoint_stack": langchain_branchpoint_stack,
            "langchain_human_in_the_loop": langchain_human_in_the_loop,
        }[name]
    raise AttributeError(f"module 'branchpoint.integrations' has no attribute {name!r}")
