# generator/llm_interface.py
# Handles communication with local or remote LLMs.

from typing import Optional, Dict, Any, Union, TYPE_CHECKING
import json
import logging
import os

from openai import OpenAI

if TYPE_CHECKING:
    from causalrag.observability import CausalTelemetry


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
        telemetry: Optional["CausalTelemetry"] = None,
    ):
        self.model = model
        self.provider = provider.lower()
        self.telemetry = telemetry
        self.last_usage: Dict[str, int] = {}
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
        self.last_usage = {}
        try:
            # Some long-standing tests and downstream integrations construct a
            # lightweight LLMInterface with ``__new__`` and inject only the
            # provider/client fields. Observability must remain optional on that
            # path as well, so an absent telemetry attribute is a no-op rather
            # than a compatibility break.
            telemetry = getattr(self, "telemetry", None)
            if telemetry is None:
                return self._dispatch_generate(prompt, temperature, max_tokens, stream, json_mode)

            attributes: Dict[str, Any] = {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": self.model,
                "gen_ai.provider.name": self.provider,
                "causalrag.llm.json_mode": bool(json_mode),
                "causalrag.llm.stream": bool(stream),
                "causalrag.llm.input_characters": len(prompt),
                "causalrag.llm.max_output_tokens": int(max_tokens),
            }
            if telemetry.capture_content:
                attributes["gen_ai.system_instructions"] = self.system_message
                attributes["causalrag.llm.prompt"] = prompt

            with telemetry.span(f"chat {self.model}", attributes) as span:
                result = self._dispatch_generate(prompt, temperature, max_tokens, stream, json_mode)
                input_tokens = self.last_usage.get("input_tokens")
                output_tokens = self.last_usage.get("output_tokens")
                if input_tokens is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
                if output_tokens is not None:
                    span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
                if telemetry.capture_content:
                    span.set_attribute("causalrag.llm.output", result)
                return result
        except Exception as exc:
            logging.error("Error generating completion: %s", exc)
            return f"Error generating response: {exc}"

    def _dispatch_generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
        json_mode: bool,
    ) -> str:
        if self.provider == "openai":
            return self._generate_openai(prompt, temperature, max_tokens, stream, json_mode)
        if self.provider == "anthropic":
            return self._generate_anthropic(prompt, temperature, max_tokens, stream)
        if self.provider == "local":
            return self._generate_local(prompt, temperature, max_tokens, stream)
        raise ValueError(f"Unsupported provider: {self.provider}")

    @staticmethod
    def _read_usage(usage: Any, *, input_names, output_names) -> Dict[str, int]:
        if usage is None:
            return {}
        result: Dict[str, int] = {}
        for name in input_names:
            value = getattr(usage, name, None)
            if value is not None:
                result["input_tokens"] = int(value)
                break
        for name in output_names:
            value = getattr(usage, name, None)
            if value is not None:
                result["output_tokens"] = int(value)
                break
        return result

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
            self.last_usage = self._read_usage(
                getattr(response, "usage", None),
                input_names=("input_tokens", "prompt_tokens"),
                output_names=("output_tokens", "completion_tokens"),
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
        self.last_usage = self._read_usage(
            getattr(response, "usage", None),
            input_names=("prompt_tokens", "input_tokens"),
            output_names=("completion_tokens", "output_tokens"),
        )
        return response.choices[0].message.content or ""

    def _generate_anthropic(
        self, prompt: str, temperature: float, max_tokens: int, stream: bool
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.messages.create(
            model=self.model,
            system=self.system_message,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.last_usage = self._read_usage(
            getattr(response, "usage", None),
            input_names=("input_tokens",),
            output_names=("output_tokens",),
        )
        return response.content[0].text

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
        self.last_usage = self._read_usage(
            getattr(response, "usage", None),
            input_names=("prompt_tokens", "input_tokens"),
            output_names=("completion_tokens", "output_tokens"),
        )
        return response.choices[0].message.content or ""
