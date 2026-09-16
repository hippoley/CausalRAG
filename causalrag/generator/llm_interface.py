# generator/llm_interface.py
# Handles communication with local or remote LLMs.

from typing import Optional, Dict, Any, Union
import json
import logging
import os

from openai import OpenAI


class LLMInterface:
    """Small provider adapter used by the causal reasoner.

    OpenAI defaults to the current GPT-5.6 family and the Responses API. Local
    OpenAI-compatible servers keep using Chat Completions for compatibility.
    """

    def __init__(
        self,
        model: str = "gpt-5.6-terra",
        api_key: Optional[str] = None,
        provider: str = "openai",
        system_message: Optional[str] = None,
    ):
        self.model = model
        self.provider = provider.lower()
        self.system_message = system_message or (
            "You are a causal reasoning assistant. Separate observations from "
            "causal claims, expose uncertainty, and prefer reversible information-"
            "gathering actions before risky interventions."
        )

        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        elif self.provider == "local":
            self.base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")
            self.client = OpenAI(base_url=self.base_url, api_key="not-needed")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800,
        stream: bool = False,
        json_mode: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        try:
            if self.provider == "openai":
                return self._generate_openai(
                    prompt, temperature, max_tokens, stream, json_mode
                )
            if self.provider == "anthropic":
                return self._generate_anthropic(prompt, temperature, max_tokens, stream)
            if self.provider == "local":
                return self._generate_local(prompt, temperature, max_tokens, stream)
            raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as exc:
            logging.error("Error generating completion: %s", exc)
            return f"Error generating response: {exc}"

    def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
        json_mode: bool,
    ) -> str:
        # GPT-5.6 and later are routed through Responses API. Keep the older
        # Chat Completions path for explicit legacy model IDs and streaming.
        use_responses = self.model.startswith("gpt-5") and not stream
        if use_responses:
            input_text = prompt
            if json_mode:
                input_text += (
                    "\n\nReturn only valid JSON. Do not wrap it in markdown fences "
                    "or add prose outside the JSON value."
                )
            response = self.client.responses.create(
                model=self.model,
                instructions=self.system_message,
                input=input_text,
                max_output_tokens=max_tokens,
            )
            return response.output_text

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        if stream:
            response_stream = self.client.chat.completions.create(**kwargs, stream=True)
            chunks = []
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            return "".join(chunks)

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _generate_anthropic(
        self, prompt: str, temperature: float, max_tokens: int, stream: bool
    ) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.messages.create(
                model=self.model,
                system=self.system_message,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.content[0].text
        except ImportError:
            return "Anthropic package not installed. Install with: pip install 'causalrag[anthropic]'"

    def _generate_local(
        self, prompt: str, temperature: float, max_tokens: int, stream: bool
    ) -> str:
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            chunks = []
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            return "".join(chunks)
        return response.choices[0].message.content or ""
