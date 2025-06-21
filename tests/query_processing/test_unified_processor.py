"""
Tests simples pour le Unified Query Processor
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Ajout du path pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from core.query_processing.unified_query_processor import (
    UnifiedQueryProcessor, create_query_processor, process_single_query
)
from core.query_processing.response_parser import ProcessedQuery, TechnicalTerms, EquipmentInfo


class TestUnifiedQueryProcessor:
    
    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_processor_creation(self, mock_llm_client):
        """Test création du processeur"""
        processor = UnifiedQueryProcessor()
        
        assert processor.llm_client is not None
        assert processor.parser is not None
        assert processor.prompt_template is not None
    
    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_process_empty_query(self, mock_llm_client):
        """Test avec requête vide"""
        processor = UnifiedQueryProcessor()
        
        with pytest.raises(ValueError) as exc_info:
            processor.process_user_query("")
        
        assert "ne peut pas être vide" in str(exc_info.value)
    
    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_process_valid_query(self, mock_llm_client):
        """Test traitement requête valide"""
        # Mock du client LLM
        mock_client = MagicMock()
        mock_client.generate.return_value = '''
        {
            "technical_terms": {"error_codes": ["ACAL-006"], "components": [], "equipment_models": [], "technical_keywords": []},
            "equipment_info": {"primary_equipment": "FANUC", "equipment_type": "robot", "manufacturer": "FANUC"},
            "filtered_query": "ACAL-006 error",
            "query_variants": []
        }
        '''
        
        processor = UnifiedQueryProcessor(llm_client=mock_client)
        result = processor.process_user_query("I have ACAL-006 error")
        
        assert isinstance(result, ProcessedQuery)
        assert result.raw_query == "I have ACAL-006 error"
        mock_client.generate.assert_called_once()
    
    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_process_llm_error_fallback(self, mock_llm_client):
        """Test fallback en cas d'erreur LLM"""
        # Mock qui génère une erreur
        mock_client = MagicMock()
        mock_client.generate.side_effect = Exception("LLM Error")
        
        processor = UnifiedQueryProcessor(llm_client=mock_client)
        result = processor.process_user_query("test query")
        
        # Doit retourner un fallback
        assert isinstance(result, ProcessedQuery)
        assert result.raw_query == "test query"
        assert result.filtered_query == "test query"  # Fallback utilise requête originale
    
    def test_get_config(self):
        """Test récupération config"""
        with patch('core.query_processing.unified_query_processor.LLMClient') as mock_llm:
            mock_client = MagicMock()
            mock_client.get_config.return_value = {"model": "test"}
            
            processor = UnifiedQueryProcessor(llm_client=mock_client)
            config = processor.get_config()
            
            assert "llm_config" in config
            assert "prompt_loaded" in config


class TestUtilityFunctions:
    
    @patch('core.query_processing.unified_query_processor.UnifiedQueryProcessor')
    def test_create_query_processor(self, mock_processor_class):
        """Test fonction create_query_processor"""
        mock_instance = MagicMock()
        mock_processor_class.return_value = mock_instance
        
        result = create_query_processor()
        
        mock_processor_class.assert_called_once()
        assert result == mock_instance
    
    @patch('core.query_processing.unified_query_processor.create_query_processor')
    def test_process_single_query(self, mock_create_processor):
        """Test fonction process_single_query"""
        # Mock du processeur
        mock_processor = MagicMock()
        mock_result = MagicMock()
        mock_processor.process_user_query.return_value = mock_result
        mock_create_processor.return_value = mock_processor
        
        result = process_single_query("test query")
        
        mock_processor.process_user_query.assert_called_once_with("test query")
        assert result == mock_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])