"""
Model Manager - Manages LLM model access and orchestration
"""

from typing import Dict, Any, Optional, List
import os
from loguru import logger


class ModelManager:
    """
    Manages access to multiple LLM models
    """

    def __init__(self):
        """Initialize model manager"""
        self.available_models = self._detect_available_models()
        logger.info(
            f"Model manager initialized with {len(self.available_models)} models"
        )

    def _detect_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Detect available models based on API keys"""
        models = {}

        # Check OpenAI
        if os.getenv("OPENAI_API_KEY"):
            models["openai"] = {
                "provider": "openai",
                "models": ["gpt-4", "gpt-3.5-turbo"],
                "available": True,
            }

        # Check Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            models["anthropic"] = {
                "provider": "anthropic",
                "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
                "available": True,
            }

        # Check Gemini
        if os.getenv("GEMINI_API_KEY"):
            models["gemini"] = {
                "provider": "gemini",
                "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
                "available": True,
            }

        # Check Moonshot
        if os.getenv("MOONSHOT_API_KEY"):
            models["moonshot"] = {
                "provider": "moonshot",
                "models": ["moonshot-v1-8k", "moonshot-v1-32k"],
                "available": True,
            }

        return models

    def get_available_models(self) -> List[str]:
        """Get list of available model providers"""
        return list(self.available_models.keys())

    def get_model_info(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get information about a model provider"""
        return self.available_models.get(provider)

    def is_available(self, provider: str) -> bool:
        """Check if a model provider is available"""
        model_info = self.available_models.get(provider)
        return model_info is not None and model_info.get("available", False)
