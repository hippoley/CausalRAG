# generator/prompt_builder.py
# Creates prompts for LLM input with optional causal context

from typing import List, Dict, Optional, Tuple, Union

class PromptBuilder:
    def __init__(self, template_style: str = "detailed"):
        """
        Initialize prompt builder with specific template style
        
        Args:
            template_style: Style of template to use ("basic", "detailed", "structured")
        """
        self.template_style = template_style
        
    def build_prompt(self, 
                    query: str, 
                    passages: List[str], 
                    causal_paths: Optional[List[List[str]]] = None,
                    causal_graph_summary: Optional[str] = None) -> str:
        """
        Build a prompt incorporating both retrieved passages and causal information
        
        Args:
            query: User question
            passages: Retrieved text passages
            causal_paths: List of causal paths (each path is list of connected nodes)
            causal_graph_summary: Optional summary of the local causal graph structure
            
        Returns:
            Complete prompt string for LLM
        """
        if self.template_style == "basic":
            return self._build_basic_prompt(query, passages, causal_paths)
        elif self.template_style == "detailed":
            return self._build_detailed_prompt(query, passages, causal_paths, causal_graph_summary)
        else:  # structured
            return self._build_structured_prompt(query, passages, causal_paths, causal_graph_summary)
    
    def _build_basic_prompt(self, query: str, passages: List[str], causal_paths: Optional[List[List[str]]]) -> str:
        """Basic prompt format with minimal instructions"""
        prompt = "Answer the following question using the provided context.\n\n"
        
        # Add causal paths if available
        if causal_paths and len(causal_paths) > 0:
            prompt += "Causal relationships:\n"
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
                              causal_graph_summary: Optional[str]) -> str:
        """Detailed prompt with explicit instructions on using causal information"""
        prompt = """You are a causal reasoning assistant. Answer the following question using:
1. The provided context passages
2. The causal relationships between concepts
3. Your understanding of how causes lead to effects

Ensure your answer reflects the causal mechanisms described in the context.
"""
        
        # Add causal information
        if causal_paths and len(causal_paths) > 0:
            prompt += "\nRelevant causal relationships:\n"
            for i, path in enumerate(causal_paths):
                prompt += f"[{i+1}] {' → '.join(path)}\n"
            
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
                               causal_graph_summary: Optional[str]) -> str:
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
            prompt += "\nRelevant causal pathways:\n"
            for i, path in enumerate(causal_paths):
                prompt += f"[{i+1}] {' → '.join(path)}\n"
            
            if causal_graph_summary:
                prompt += f"\nCausal graph structure: {causal_graph_summary}\n"
            
            prompt += "\nUse these causal pathways to structure your reasoning."
        
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