# generator/llm_interface.py
# Handles communication with local or remote LLMs

from typing import Optional, Dict, List, Any, Union
import os
import openai
from openai import OpenAI  # New OpenAI client
import json
import logging

class LLMInterface:
    """Interface for interacting with various LLM providers"""
    
    def __init__(self, 
                model: str = "gpt-4", 
                api_key: Optional[str] = None,
                provider: str = "openai",
                system_message: Optional[str] = None):
        """
        Initialize LLM interface
        
        Args:
            model: Model identifier to use
            api_key: API key (if None, will try to read from env variables)
            provider: LLM provider ("openai", "anthropic", "local")
            system_message: Optional system message to use with the model
        """
        self.model = model
        self.provider = provider.lower()
        self.system_message = system_message or "You are a helpful AI assistant with expertise in causal reasoning."
        
        # Setup API credentials
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == "local":
            # Setup for local models (e.g., through LM Studio API)
            self.base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
            self.client = OpenAI(base_url=self.base_url, api_key="not-needed")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.3, 
                max_tokens: int = 800,
                stream: bool = False,
                json_mode: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Generate text completion using the configured LLM
        
        Args:
            prompt: Prompt text to send to the model
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            json_mode: Whether to request JSON output format
            
        Returns:
            Generated text or structured response
        """
        try:
            if self.provider == "openai":
                return self._generate_openai(prompt, temperature, max_tokens, stream, json_mode)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt, temperature, max_tokens, stream)
            elif self.provider == "local":
                return self._generate_local(prompt, temperature, max_tokens, stream)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logging.error(f"Error generating completion: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_openai(self, 
                        prompt: str, 
                        temperature: float, 
                        max_tokens: int, 
                        stream: bool,
                        json_mode: bool) -> str:
        """Generate using OpenAI API"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_mode and "gpt-4" in self.model:  # Only available for certain models
            kwargs["response_format"] = {"type": "json_object"}
        
        if stream:
            response_stream = self.client.chat.completions.create(**kwargs, stream=True)
            chunks = []
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
                    # You could yield here if implementing a generator
            return "".join(chunks)
        else:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str, temperature: float, max_tokens: int, stream: bool) -> str:
        """Generate using Anthropic API"""
        try:
            import anthropic
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text
        except ImportError:
            return "Anthropic package not installed. Install with: pip install anthropic"
        
    def _generate_local(self, prompt: str, temperature: float, max_tokens: int, stream: bool) -> str:
        """Generate using local model API"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )