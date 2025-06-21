"""
Tests d'intégration simples pour le module query_processing
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Ajout du path pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from core.query_processing import (
    quick_setup, process_single_query, create_enhanced_retrieval_engine
)


class TestIntegration:
    
    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_quick_setup(self, mock_llm_client):
        """Test setup rapide du module"""
        processor = quick_setup()
        
        # Doit retourner un processeur fonctionnel
        assert processor is not None
        assert hasattr(processor, 'process_user_query')
    
    @patch('core.query_processing.unified_query_processor.create_query_processor')
    def test_process_single_query_integration(self, mock_create_processor):
        """Test fonction process_single_query"""
        # Mock du processeur
        mock_processor = MagicMock()
        mock_result = MagicMock()
        mock_processor.process_user_query.return_value = mock_result
        mock_create_processor.return_value = mock_processor
        
        result = process_single_query("test query")
        
        mock_processor.process_user_query.assert_called_once_with("test query")
        assert result == mock_result
    
    def test_create_enhanced_retrieval_integration(self):
        """Test création du moteur de retrieval amélioré"""
        mock_bm25 = MagicMock()
        mock_faiss = MagicMock()
        mock_reranker = MagicMock()
        
        engine = create_enhanced_retrieval_engine(mock_bm25, mock_faiss, mock_reranker)
        
        assert engine is not None
        assert hasattr(engine, 'search_with_variants')
        assert engine.bm25 == mock_bm25
        assert engine.faiss == mock_faiss
        assert engine.reranker == mock_reranker
    
    @patch('core.query_processing.llm_client.client')
    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_full_pipeline_mock(self, mock_llm_class, mock_openai_client):
        """Test pipeline complet avec mocks"""
        # Mock réponse OpenAI
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '''
        {
            "technical_terms": {
                "error_codes": ["ACAL-006"],
                "components": ["TPE"],
                "equipment_models": ["FANUC-30iB"],
                "technical_keywords": ["operation error"]
            },
            "equipment_info": {
                "primary_equipment": "FANUC R-30iB",
                "equipment_type": "industrial_robot",
                "manufacturer": "FANUC"
            },
            "filtered_query": "ACAL-006 TPE operation error FANUC R-30iB",
            "query_variants": ["ACAL-006 teach pendant error"],
            "confidence_score": 0.9
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Test du pipeline
        result = process_single_query("I got ACAL-006 TPE error on FANUC machine")
        
        # Vérifications
        assert result.raw_query == "I got ACAL-006 TPE error on FANUC machine"
        assert "ACAL-006" in result.technical_terms.error_codes
        assert result.equipment_info.manufacturer == "FANUC"
        assert len(result.query_variants) > 0


class TestModuleImports:
    
    def test_all_imports_work(self):
        """Test que tous les imports fonctionnent"""
        try:
            from core.query_processing import (
                LLMClient, ProcessedQuery, UnifiedQueryProcessor, 
                EnhancedRetrievalEngine, create_query_processor
            )
            
            # Si on arrive ici, tous les imports ont fonctionné
            assert True
            
        except ImportError as e:
            pytest.fail(f"Import error: {e}")
    
    def test_module_exports(self):
        """Test que le module exporte les bonnes classes"""
        import core.query_processing as qp
        
        required_exports = [
            'LLMClient', 'ProcessedQuery', 'UnifiedQueryProcessor',
            'EnhancedRetrievalEngine', 'create_query_processor'
        ]
        
        for export in required_exports:
            assert hasattr(qp, export), f"Missing export: {export}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])