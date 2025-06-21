"""
Tests simples pour le LLM Client
"""

import pytest
import os
from unittest.mock import patch, MagicMock
import sys

# Ajout du path pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from core.query_processing.llm_client import LLMClient, create_llm_client


class TestLLMClient:
    
    def test_client_creation(self):
        """Test création basique du client"""
        client = LLMClient()
        assert client.model == "gpt-4o-mini"
        assert client.max_tokens == 1000
        assert client.temperature == 0.1
    
    def test_client_with_custom_params(self):
        """Test avec paramètres personnalisés"""
        client = LLMClient(model="gpt-4o", max_tokens=500, temperature=0.2)
        assert client.model == "gpt-4o"
        assert client.max_tokens == 500
        assert client.temperature == 0.2
    
    @patch('core.query_processing.llm_client.client')
    def test_generate_success(self, mock_openai_client):
        """Test génération réussie"""
        # Mock de la réponse OpenAI
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response"
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        client = LLMClient()
        result = client.generate("Test prompt")
        
        assert result == "Test response"
        mock_openai_client.chat.completions.create.assert_called_once()
    
    @patch('core.query_processing.llm_client.client')
    def test_generate_error(self, mock_openai_client):
        """Test gestion d'erreur"""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        client = LLMClient()
        
        with pytest.raises(Exception) as exc_info:
            client.generate("Test prompt")
        
        assert "Erreur LLM" in str(exc_info.value)
    
    def test_get_config(self):
        """Test récupération config"""
        client = LLMClient(model="test-model", max_tokens=123, temperature=0.5)
        config = client.get_config()
        
        assert config["model"] == "test-model"
        assert config["max_tokens"] == 123
        assert config["temperature"] == 0.5
    
    def test_create_llm_client_utility(self):
        """Test fonction utilitaire"""
        client = create_llm_client("gpt-4o")
        assert isinstance(client, LLMClient)
        assert client.model == "gpt-4o"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])