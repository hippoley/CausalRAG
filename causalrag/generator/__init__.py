"""
Text generation components for producing causally-aware answers.

This package provides prompt building utilities and interfaces to 
language models for generating high-quality answers.
"""

from .prompt_builder import build_prompt, PromptBuilder
from .llm_interface import LLMInterface

__all__ = [
    'build_prompt',
    'PromptBuilder',
    'LLMInterface'
] 