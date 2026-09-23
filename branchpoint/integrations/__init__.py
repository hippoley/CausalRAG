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
    "langchain_tool_schema",
]


_LANGCHAIN_EXPORTS = {
    "LangChainBranchpointError",
    "LangChainBranchpointMiddleware",
    "langchain_branchpoint_stack",
    "langchain_human_in_the_loop",
    "langchain_tool_schema",
}


def __getattr__(name):
    if name in _LANGCHAIN_EXPORTS:
        from .langchain import (
            LangChainBranchpointError,
            LangChainBranchpointMiddleware,
            langchain_branchpoint_stack,
            langchain_human_in_the_loop,
            langchain_tool_schema,
        )

        return {
            "LangChainBranchpointError": LangChainBranchpointError,
            "LangChainBranchpointMiddleware": LangChainBranchpointMiddleware,
            "langchain_branchpoint_stack": langchain_branchpoint_stack,
            "langchain_human_in_the_loop": langchain_human_in_the_loop,
            "langchain_tool_schema": langchain_tool_schema,
        }[name]
    raise AttributeError(f"module 'branchpoint.integrations' has no attribute {name!r}")
