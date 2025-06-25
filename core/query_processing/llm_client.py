"""
LLM Client: OpenAI Integration for Query Preprocessing

This module provides a simplified OpenAI client interface for query preprocessing
in the RAG diagnosis system. It prioritizes configuration loading from settings.yaml
with comprehensive fallback mechanisms and parameter override capabilities for
flexible LLM integration with optimal default settings.

Key components:
- Priority-based configuration loading from external YAML settings
- OpenAI GPT model integration with configurable parameters
- Robust error handling and fallback configuration mechanisms
- Parameter override system for runtime customization
- Comprehensive logging for debugging and monitoring

Dependencies: openai, yaml, dotenv, typing
Usage: Import LLMClient for OpenAI-based query preprocessing with automatic configuration
"""

import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _load_llm_config() -> Dict[str, Any]:
    """
    Load LLM configuration from settings.yaml with priority handling.
    
    Attempts to load LLM configuration from external YAML file first,
    then falls back to sensible defaults if file is missing or invalid.
    
    Returns:
        Dict[str, Any]: LLM configuration with model, tokens, and temperature settings
    """
    default_config = {
        "model": "gpt-4o-mini",  # Fallback if settings.yaml absent
        "max_tokens": 1000,
        "temperature": 0.1
    }
    
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        
        llm_cfg = settings.get("query_processing", {}).get("llm", {})
        
        # Merge with defaults (priority to settings.yaml)
        config = {
            "model": llm_cfg.get("model", default_config["model"]),
            "max_tokens": llm_cfg.get("max_tokens", default_config["max_tokens"]),
            "temperature": llm_cfg.get("temperature", default_config["temperature"])
        }
        
        print(f"LLM configuration loaded from settings.yaml:")
        print(f"   Model: {config['model']}")
        print(f"   Temperature: {config['temperature']}")
        print(f"   Max tokens: {config['max_tokens']}")
        
        return config
        
    except FileNotFoundError:
        print("Warning: settings.yaml not found, using default configuration")
        return default_config
    except Exception as e:
        print(f"Warning: Error reading settings.yaml: {e}, using default configuration")
        return default_config


class LLMClient:
    """Simplified OpenAI client with priority configuration loading from settings.yaml"""
    
    def __init__(self, model: Optional[str] = None, max_tokens: Optional[int] = None, 
                 temperature: Optional[float] = None):
        """
        Initialize LLM client with priority configuration loading.
        
        Loads configuration from settings.yaml first, then applies parameter
        overrides if provided. Ensures consistent configuration management
        with flexible runtime customization capabilities.
        
        Args:
            model (Optional[str]): Model override (supersedes settings.yaml if provided)
            max_tokens (Optional[int]): Token limit override (supersedes settings.yaml if provided)
            temperature (Optional[float]): Temperature override (supersedes settings.yaml if provided)
        """
        # Priority loading from settings.yaml
        config = _load_llm_config()
        
        # Apply parameters (priority: explicit parameters > settings.yaml > defaults)
        self.model = model if model is not None else config["model"]
        self.max_tokens = max_tokens if max_tokens is not None else config["max_tokens"]
        self.temperature = temperature if temperature is not None else config["temperature"]
        
        print(f"LLMClient initialized:")
        print(f"   Final model: {self.model}")
        print(f"   Final temperature: {self.temperature}")
        print(f"   Final max tokens: {self.max_tokens}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate response for given prompt using configured LLM.
        
        Sends prompt to OpenAI API with configured parameters and returns
        the generated response with comprehensive error handling.
        
        Args:
            prompt (str): Input prompt to send to the LLM
            
        Returns:
            str: Generated response from the LLM
            
        Raises:
            Exception: If API call fails or response is invalid
        """
        try:
            print(f"LLM call to {self.model} with {len(prompt)} character prompt")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            llm_response = response.choices[0].message.content.strip()
            print(f"LLM response received: {len(llm_response)} characters")
            
            return llm_response
            
        except Exception as e:
            print(f"LLM error: {e}")
            raise Exception(f"LLM error: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return current client configuration.
        
        Provides access to current LLM configuration for debugging,
        monitoring, and configuration validation purposes.
        
        Returns:
            Dict[str, Any]: Current configuration with model, tokens, and temperature
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


def create_llm_client(model: Optional[str] = None) -> LLMClient:
    """
    Create LLM client with priority configuration from settings.yaml.
    
    Factory function for convenient LLM client creation with automatic
    configuration loading and optional model override.
    
    Args:
        model (Optional[str]): Override model from settings.yaml if provided
        
    Returns:
        LLMClient: Configured LLM client instance
    """
    return LLMClient(model=model)


if __name__ == "__main__":
    # Simple test with configuration display
    print("LLMClient test with settings.yaml")
    
    client = create_llm_client()
    
    test_prompt = "Explain what ACAL-006 error means in one sentence."
    try:
        response = client.generate(test_prompt)
        print(f"Test successful: {response}")
    except Exception as e:
        print(f"Test failed: {e}")