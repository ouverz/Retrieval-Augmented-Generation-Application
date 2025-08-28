# services/llm_factory.py
import logging
from typing import Type, Optional
from pydantic import BaseModel

from config.settings import get_settings
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory that returns a configured pydantic-ai Agent.
    Reads all provider/model defaults from config.settings.
    """

    def __init__(self, provider: Optional[str] = None):
        # 1) Load settings FIRST
        self.settings = get_settings()

        # 2) Resolve provider (prefer arg; else try settings; else 'openai')
        resolved_provider = provider
        if resolved_provider is None:
            # Try common fields; fall back to 'openai'
            default_from_settings = getattr(self.settings, "default_factory", "openai")
            if default_from_settings:
                resolved_provider = default_from_settings
            else:
                # If the 'openai' section exists, use it; otherwise just default to 'openai'
                resolved_provider = "openai"

        self.provider = resolved_provider

        # 3) Build the provider model once
        self._model = self._initialize_model()

    def _initialize_model(self):
        """
        Build a pydantic-ai model wrapper for the selected provider using provider-specific settings.
        """
        if self.provider == "openai":
            prov = self.settings.openai  # OpenAISettings
            return OpenAIModel(
                model_name=prov.default_model
                # api_key=prov.api_key,
                # timeout=getattr(prov, "timeout_s", None),
            )

        if self.provider == "anthropic":
            if not hasattr(self.settings, "anthropic"):
                raise ValueError("Anthropic settings not found in config.settings")
            prov = (
                self.settings.anthropic
            )  # AnthropicSettings (must exist in your Settings)
            return AnthropicModel(
                model_name=prov.default_model,
                #  api_key=prov.api_key,
                timeout=getattr(prov, "timeout_s", None),
            )

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def build_agent(
        self,
        system_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Agent:
        """
        Return a configured Agent that will always parse the LLM output into `result_type`.
        """
        prov_settings = getattr(self.settings, self.provider)

        model = self._model
        # Apply runtime defaults if the wrapper supports it
        if hasattr(model, "with_options"):
            model = model.with_options(
                temperature=temperature
                if temperature is not None
                else getattr(prov_settings, "temperature", None),
                max_output_tokens=max_output_tokens
                if max_output_tokens is not None
                else getattr(prov_settings, "max_tokens", None),
            )

        return Agent(model=model, system_prompt=system_prompt)

    async def run_structured(
        self,
        *,
        result_type: Type[BaseModel],
        system_prompt: str,
        user_message: str,
    ) -> BaseModel:
        """
        Convenience wrapper: build an Agent and run once with retries from settings.
        """
        agent = self.build_agent(system_prompt)
        # Get raw response from agent
        response = await agent.run(user_message)
        
        # Extract data from AgentRunResult
        if hasattr(response, 'output'):
            raw_data = response.output
        elif hasattr(response, 'data'):
            raw_data = response.data
        elif hasattr(response, 'content'):
            raw_data = response.content
        elif hasattr(response, 'message'):
            raw_data = response.message
        else:
            # Try to get string representation
            raw_data = str(response)
        
        # If raw_data is already the correct type, return it
        if isinstance(raw_data, result_type):
            return raw_data
        
        # Try to parse raw response as JSON into the Pydantic model
        if isinstance(raw_data, str):
            import json
            import re
            
            try:
                # Clean up the string - remove any markdown formatting
                clean_data = raw_data.strip()
                if clean_data.startswith('```json'):
                    clean_data = clean_data.replace('```json', '').replace('```', '').strip()
                
                # Try to extract JSON from mixed content
                json_match = re.search(r'\{.*\}', clean_data, re.DOTALL)
                if json_match:
                    clean_data = json_match.group(0)
                
                parsed_data = json.loads(clean_data)
                validated_response = result_type(**parsed_data)
                return validated_response
                
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM response as structured JSON: {e}")
                logger.debug(f"Raw response: {raw_data[:200]}...")
                # Force create a valid response rather than returning raw string
                return result_type(
                    thought_process=["LLM returned non-structured response"],
                    answer=str(raw_data)[:500] + ("..." if len(str(raw_data)) > 500 else ""),
                    enough_context=False,
                    confidence=0.1,
                    citations=[],
                    precision=0.0,
                    evidence_precision="low"
                )
        
        # Handle dict response
        if isinstance(raw_data, dict):
            try:
                return result_type(**raw_data)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to validate dict response: {e}")
                # Force create valid response from dict
                return result_type(
                    thought_process=["LLM returned invalid structured response"],
                    answer=str(raw_data.get('answer', 'Invalid response format')),
                    enough_context=False,
                    confidence=0.1,
                    citations=[],
                    precision=0.0,
                    evidence_precision="low"
                )
        
        # Last resort: force structure on any other type
        logger.warning(f"LLM returned unexpected type {type(raw_data)}: {raw_data}")
        return result_type(
            thought_process=["LLM returned completely unexpected response format"],
            answer=f"System error: Received {type(raw_data).__name__} instead of structured response",
            enough_context=False,
            confidence=0.0,
            citations=[],
            precision=0.0,
            evidence_precision="low"
        )
