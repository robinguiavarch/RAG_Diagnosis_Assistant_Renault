"""
test_llm_client
===============

Unit tests for the LLMClient class and the create_llm_client factory.

These tests verify correct initialization and output generation behavior,
with and without custom parameters.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------------- #
# Project path setup                                                          #
# --------------------------------------------------------------------------- #

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from core.query_processing.llm_client import LLMClient, create_llm_client


# --------------------------------------------------------------------------- #
# Unit test suite                                                             #
# --------------------------------------------------------------------------- #

class TestLLMClient:
    """Unit tests for the LLMClient object."""

    def test_client_creation(self):
        """Verify that the client is correctly initialized with default parameters."""
        client = LLMClient()
        assert client.model == "gpt-4o-mini"
        assert client.max_tokens == 1000
        assert client.temperature == 0.1

    def test_client_with_custom_params(self):
        """Check client creation with custom parameter values."""
        client = LLMClient(model="gpt-4o", max_tokens=500, temperature=0.2)
        assert client.model == "gpt-4o"
        assert client.max_tokens == 500
        assert client.temperature == 0.2

    @patch('core.query_processing.llm_client.client')
    def test_generate_success(self, mock_openai_client):
        """Simulate a successful API call and check output format."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message={"content": "response"})]
        mock_openai_client.chat.completions.create.return_value = mock_response

        llm = LLMClient()
        result = llm.generate(prompt="Test prompt", system="System prompt")

        assert isinstance(result.content, str)
        assert result.content == "response"

    @patch('core.query_processing.llm_client.client')
    def test_generate_failure(self, mock_openai_client):
        """Simulate an exception from the OpenAI API and ensure fallback behavior."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API error")

        llm = LLMClient()
        result = llm.generate(prompt="Test", system="System")

        assert result.content == ""
        assert "API error" in result.logs["error"]

    def test_factory_creation(self):
        """Ensure the factory function returns an instance of LLMClient."""
        client = create_llm_client()
        assert isinstance(client, LLMClient)
