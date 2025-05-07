# generator/prompt_builder.py
# Creates prompts for LLM input with causal context and natural language summarization

from typing import List, Dict, Optional, Tuple, Union
import os
import re
import jinja2
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builds LLM prompts with rich causal information and natural language summaries"""
    
    def __init__(self, 
                template_style: str = "detailed", 
                templates_dir: Optional[str] = None,
                llm_interface = None):
        """
        Initialize prompt builder with specific template style and optional LLM
        
        Args:
            template_style: Style of template ("basic", "detailed", "structured", "chain_of_thought")
            templates_dir: Directory containing custom template files
            llm_interface: Optional LLM interface for causal summarization
        """
        self.template_style = template_style
        self.llm_interface = llm_interface
        
        # Setup Jinja2 template environment
        if templates_dir:
            self.template_loader = jinja2.FileSystemLoader(templates_dir)
        else:
            # Default to package templates directory
            package_dir = Path(__file__).parent.parent
            self.template_loader = jinja2.FileSystemLoader(package_dir / "templates")
            
        self.template_env = jinja2.Environment(
            loader=self.template_loader,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Try to load template file based on style
        template_filename = f"causal_prompt_{template_style}.txt"
        try:
            self.template = self.template_env.get_template(template_filename)
        except jinja2.exceptions.TemplateNotFound:
            # Fall back to default template
            logger.warning(f"Template {template_filename} not found, using default")
            self.template = self.template_env.get_template("causal_prompt.txt")
    
    def build_prompt(self, 
                    query: str, 
                    passages: List[str], 
                    causal_paths: Optional[List[List[str]]] = None,
                    causal_graph_summary: Optional[str] = None,
                    example_triples: Optional[List[Tuple[str, str, float]]] = None) -> str:
        """
        Build a prompt incorporating retrieved passages and causal information
        
        Args:
            query: User question
            passages: Retrieved text passages
            causal_paths: List of causal paths (each path is list of connected nodes)
            causal_graph_summary: Optional summary of the local causal graph structure
            example_triples: Optional list of (cause, effect, confidence) examples
            
        Returns:
            Complete prompt string for LLM
        """
        # Generate natural language causal path summaries if LLM is available
        path_summaries = None
        if causal_paths and self.llm_interface:
            path_summaries = self._generate_causal_summaries(causal_paths, query)
        
        # Prioritize generated summaries or fall back to template-based rendering
        if self._is_jinja_template():
            return self._render_template(
                query, 
                passages, 
                causal_paths, 
                causal_graph_summary,
                path_summaries,
                example_triples
            )
        else:
            # Use programmatic prompt building based on style
            if self.template_style == "basic":
                return self._build_basic_prompt(query, passages, causal_paths, path_summaries)
            elif self.template_style == "detailed":
                return self._build_detailed_prompt(query, passages, causal_paths, 
                                                causal_graph_summary, path_summaries)
            elif self.template_style == "chain_of_thought":
                return self._build_cot_prompt(query, passages, causal_paths, 
                                             causal_graph_summary, path_summaries)
            else:  # structured
                return self._build_structured_prompt(query, passages, causal_paths, 
                                                   causal_graph_summary, path_summaries)
    
    def _is_jinja_template(self) -> bool:
        """Check if we're using a Jinja template file"""
        return self.template is not None
    
    def _render_template(self, 
                        query: str, 
                        passages: List[str], 
                        causal_paths: Optional[List[List[str]]], 
                        causal_graph_summary: Optional[str],
                        path_summaries: Optional[List[str]],
                        example_triples: Optional[List[Tuple[str, str, float]]]) -> str:
        """Render the prompt using Jinja template"""
        try:
            return self.template.render(
                query=query,
                passages=passages,
                causal_paths=causal_paths,
                causal_graph_summary=causal_graph_summary,
                path_summaries=path_summaries,
                example_triples=example_triples
            )
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            # Fall back to basic prompt
            return self._build_basic_prompt(query, passages, causal_paths, path_summaries)
    
    def _generate_causal_summaries(self, 
                                  causal_paths: List[List[str]], 
                                  query: str) -> List[str]:
        """
        Generate natural language summaries of causal paths
        
        Args:
            causal_paths: List of causal paths
            query: The original query for context
            
        Returns:
            List of natural language summaries
        """
        if not self.llm_interface:
            return None
        
        summaries = []
        
        try:
            # First, create a concise version for all paths to get context
            all_paths_text = "\n".join([" → ".join(path) for path in causal_paths])
            
            # Generate an overall summary first
            overview_prompt = f"""Summarize the following causal relationships as they relate to: "{query}"
            
Causal paths:
{all_paths_text}

Provide a concise summary (1-2 sentences) that captures the key causal mechanisms:"""
            
            overview = self.llm_interface.generate(overview_prompt, 
                                                 temperature=0.3, 
                                                 max_tokens=150)
            
            # Now generate summaries for individual paths
            for i, path in enumerate(causal_paths):
                path_text = " → ".join(path)
                
                # For very short paths (2-3 nodes), we might not need a complex summary
                if len(path) <= 3:
                    summaries.append(self._rewrite_as_natural_language(path_text))
                    continue
                
                # For longer paths, ask LLM for a natural language summary
                summary_prompt = f"""Convert this causal relationship path into a natural language explanation:
                
Causal path: {path_text}

Write a concise explanation using simple natural language that maintains all the causal relationships:"""
                
                summary = self.llm_interface.generate(summary_prompt, 
                                                    temperature=0.3, 
                                                    max_tokens=100)
                
                # Clean up and add to list
                summary = summary.strip()
                summaries.append(summary)
            
            # Add the overview as the first summary
            summaries.insert(0, overview.strip())
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error generating causal summaries: {e}")
            return None
    
    def _rewrite_as_natural_language(self, path_text: str) -> str:
        """Convert a simple causal path into natural language without using LLM"""
        # Replace arrows with connecting phrases
        text = path_text.replace(" → ", " leads to ")
        
        # Ensure proper sentence format
        if not text[0].isupper():
            text = text[0].upper() + text[1:]
        if not text.endswith('.'):
            text += "."
            
        return text
    
    def _build_basic_prompt(self, 
                           query: str, 
                           passages: List[str], 
                           causal_paths: Optional[List[List[str]]],
                           path_summaries: Optional[List[str]]) -> str:
        """Basic prompt format with minimal instructions"""
        prompt = "Answer the following question using the provided context.\n\n"
        
        # Add causal paths and summaries if available
        if causal_paths and len(causal_paths) > 0:
            prompt += "Causal relationships:\n"
            
            if path_summaries and len(path_summaries) > 0:
                # First summary is overview
                if len(path_summaries) > 1:
                    prompt += f"Overview: {path_summaries[0]}\n\n"
                
                # Add path-specific summaries
                for i, path in enumerate(causal_paths):
                    prompt += f"[{i+1}] {' → '.join(path)}\n"
                    if i+1 < len(path_summaries):
                        prompt += f"   Explanation: {path_summaries[i+1]}\n"
            else:
                # Just show the paths without summaries
                for i, path in enumerate(causal_paths):
                    prompt += f"[{i+1}] {' → '.join(path)}\n"
            
            prompt += "\n"
        
        # Add context passages
        prompt += "Context:\n"
        for i, p in enumerate(passages):
            prompt += f"[{i+1}] {p}\n"
        
        prompt += f"\nQuestion: {query}\nAnswer:"
        return prompt
    
    def _build_detailed_prompt(self, 
                              query: str, 
                              passages: List[str], 
                              causal_paths: Optional[List[List[str]]], 
                              causal_graph_summary: Optional[str],
                              path_summaries: Optional[List[str]]) -> str:
        """Detailed prompt with explicit instructions on using causal information"""
        prompt = """You are a causal reasoning assistant. Answer the following question using:
1. The provided context passages
2. The causal relationships between concepts
3. Your understanding of how causes lead to effects

Ensure your answer reflects the causal mechanisms described in the context.
"""
        
        # Add causal information
        if causal_paths and len(causal_paths) > 0:
            # Add overview summary if available
            if path_summaries and len(path_summaries) > 0:
                prompt += f"\nCausal overview: {path_summaries[0]}\n"
            
            prompt += "\nRelevant causal relationships:\n"
            for i, path in enumerate(causal_paths):
                prompt += f"[{i+1}] {' → '.join(path)}\n"
                # Add path-specific summary if available (skip the overview)
                if path_summaries and i+1 < len(path_summaries):
                    prompt += f"   In other words: {path_summaries[i+1]}\n"
            
            if causal_graph_summary:
                prompt += f"\nCausal graph summary: {causal_graph_summary}\n"
        
        # Add context
        prompt += "\nContext passages:\n"
        for i, p in enumerate(passages):
            prompt += f"[{i+1}] {p.strip()}\n"
        
        prompt += f"\nQuestion: {query}\n\nAnswer (explain the causal relationships that lead to your conclusion):"
        return prompt
    
    def _build_structured_prompt(self, 
                               query: str, 
                               passages: List[str], 
                               causal_paths: Optional[List[List[str]]], 
                               causal_graph_summary: Optional[str],
                               path_summaries: Optional[List[str]]) -> str:
        """Structured prompt that asks for step-by-step causal reasoning"""
        prompt = """You are a causal reasoning assistant that explains complex relationships between concepts.
For the following question, provide a structured answer that:
1. Identifies the key causal factors involved
2. Explains how these factors relate through causal mechanisms
3. Provides a final answer that follows from this causal chain

Use the provided context passages and causal relationship information.
"""
        
        # Add causal information with explicit instructions
        if causal_paths and len(causal_paths) > 0:
            # Add overview summary if available
            if path_summaries and len(path_summaries) > 0:
                prompt += f"\nSUMMARY OF CAUSAL MECHANISMS: {path_summaries[0]}\n"
            
            prompt += "\nRelevant causal pathways:\n"
            for i, path in enumerate(causal_paths):
                prompt += f"[{i+1}] {' → '.join(path)}\n"
                # Add path-specific summary if available (skip the overview)
                if path_summaries and i+1 < len(path_summaries):
                    prompt += f"   Natural language: {path_summaries[i+1]}\n"
            
            if causal_graph_summary:
                prompt += f"\nCausal graph structure: {causal_graph_summary}\n"
            
            prompt += "\nImportant: Use these causal pathways to structure your reasoning."
        
        # Add context
        prompt += "\nContext passages:\n"
        for i, p in enumerate(passages):
            prompt += f"[{i+1}] {p.strip()}\n"
        
        prompt += f"""
Question: {query}

Your structured causal answer:
1. Causal factors:
2. Causal mechanisms:
3. Conclusion:
"""
        return prompt
    
    def _build_cot_prompt(self, 
                        query: str, 
                        passages: List[str], 
                        causal_paths: Optional[List[List[str]]], 
                        causal_graph_summary: Optional[str],
                        path_summaries: Optional[List[str]]) -> str:
        """Chain-of-thought prompt for detailed causal reasoning"""
        prompt = """You are an expert in causal reasoning who answers complex questions by tracing causal mechanisms.
For the question below, think step-by-step through the causal chains involved:

1. First identify all key concepts from the question
2. For each causal relationship relevant to these concepts:
   - Examine what causes what
   - Consider the strength and direction of the relationship
   - Look for mediators and moderators of the relationship
3. Then trace through the most plausible causal paths
4. Finally, synthesize these relationships into a cohesive explanation

Reference only information from the provided context and causal relationships.
"""
        
        # Add causal information with detailed explanations
        if causal_paths and len(causal_paths) > 0:
            if path_summaries and len(path_summaries) > 0:
                prompt += f"\nKEY INSIGHT ABOUT THESE CAUSAL MECHANISMS: {path_summaries[0]}\n"
            
            prompt += "\nCAUSAL RELATIONSHIPS TO CONSIDER:\n"
            for i, path in enumerate(causal_paths):
                prompt += f"[{i+1}] {' → '.join(path)}\n"
                if path_summaries and i+1 < len(path_summaries):
                    prompt += f"   Explanation: {path_summaries[i+1]}\n"
            
            if causal_graph_summary:
                prompt += f"\nGlobal causal structure: {causal_graph_summary}\n"
        
        # Add context
        prompt += "\nREFERENCE CONTEXTS:\n"
        for i, p in enumerate(passages):
            prompt += f"[{i+1}] {p.strip()}\n"
        
        prompt += f"""
QUESTION: {query}

STEP-BY-STEP REASONING:
1) Key concepts in this question are:
2) Relevant causal relationships from the context:
3) Tracing the causal chain:
4) Therefore, the answer is:
"""
        return prompt


def build_prompt(
    query: str, 
    passages: List[str], 
    causal_paths: Optional[List[List[str]]] = None,
    template_style: str = "detailed",
    llm_interface = None
) -> str:
    """
    Utility function to quickly build a prompt without instantiating PromptBuilder
    
    Args:
        query: User question
        passages: Retrieved text passages
        causal_paths: List of causal paths (each path is list of connected nodes)
        template_style: Style of template to use
        llm_interface: Optional LLM interface for summarization
        
    Returns:
        Complete prompt string for LLM
    """
    builder = PromptBuilder(template_style=template_style, llm_interface=llm_interface)
    return builder.build_prompt(query, passages, causal_paths)