"""
Tests simples pour l'Enhanced Retrieval Engine
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Ajout du path pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from core.query_processing.enhanced_retrieval_engine import (
    EnhancedRetrievalEngine, RetrievalResult, create_enhanced_retrieval_engine
)
from core.query_processing.response_parser import ProcessedQuery, TechnicalTerms, EquipmentInfo


class TestEnhancedRetrievalEngine:
    
    def test_engine_creation(self):
        """Test création du moteur"""
        mock_bm25 = MagicMock()
        mock_faiss = MagicMock()
        mock_reranker = MagicMock()
        
        engine = EnhancedRetrievalEngine(
            bm25_retriever=mock_bm25,
            faiss_retriever=mock_faiss,
            reranker=mock_reranker
        )
        
        assert engine.bm25 == mock_bm25
        assert engine.faiss == mock_faiss
        assert engine.reranker == mock_reranker
        assert engine.pool_size == 15  # Valeur par défaut
        assert engine.final_top_k == 5
    
    def test_deduplicate_chunks(self):
        """Test déduplication des chunks"""
        mock_bm25 = MagicMock()
        mock_faiss = MagicMock()
        mock_reranker = MagicMock()
        
        engine = EnhancedRetrievalEngine(mock_bm25, mock_faiss, mock_reranker)
        
        # Chunks avec doublons
        chunks = [
            {"text": "chunk 1", "source": "BM25"},
            {"text": "chunk 2", "source": "FAISS"},
            {"text": "chunk 1", "source": "BM25"},  # Doublon
            {"text": "chunk 3", "source": "FAISS"}
        ]
        
        result = engine._deduplicate_chunks(chunks)
        
        assert len(result) == 3  # Doublon supprimé
        texts = [chunk["text"] for chunk in result]
        assert "chunk 1" in texts
        assert "chunk 2" in texts
        assert "chunk 3" in texts
    
    def test_gather_chunk_candidates(self):
        """Test collecte des candidats"""
        # Mock retrievers
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [{"text": "bm25 result", "score": 0.8}]
        
        mock_faiss = MagicMock()
        mock_faiss.search.return_value = [{"text": "faiss result", "score": 0.7}]
        
        mock_reranker = MagicMock()
        
        engine = EnhancedRetrievalEngine(mock_bm25, mock_faiss, mock_reranker)
        
        # Mock ProcessedQuery
        processed_query = MagicMock()
        processed_query.get_all_queries.return_value = ["query 1", "query 2"]
        
        result = engine._gather_chunk_candidates(processed_query)
        
        # Doit avoir appelé search pour chaque query
        assert mock_bm25.search.call_count == 2
        assert mock_faiss.search.call_count == 2
        assert len(result) >= 2  # Au moins 2 résultats
    
    @patch('core.query_processing.enhanced_retrieval_engine.get_structured_context')
    def test_search_with_variants(self, mock_kg_context):
        """Test recherche complète avec variantes"""
        # Mock du contexte KG
        mock_kg_context.return_value = '[]'  # JSON vide
        
        # Mock retrievers
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [{"text": "result", "score": 0.8}]
        
        mock_faiss = MagicMock()
        mock_faiss.search.return_value = [{"text": "result2", "score": 0.7}]
        
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"text": "reranked result", "score": 0.9, "cross_encoder_score": 0.9}
        ]
        
        engine = EnhancedRetrievalEngine(mock_bm25, mock_faiss, mock_reranker)
        
        # Mock ProcessedQuery
        processed_query = MagicMock()
        processed_query.get_all_queries.return_value = ["main query"]
        processed_query.get_primary_query.return_value = "main query"
        processed_query.query_variants = ["variant1"]
        
        result = engine.search_with_variants(processed_query)
        
        assert isinstance(result, RetrievalResult)
        assert result.variants_used == 1
        assert result.processing_time > 0
        assert len(result.chunks) >= 0
    
    def test_utility_function(self):
        """Test fonction utilitaire"""
        mock_bm25 = MagicMock()
        mock_faiss = MagicMock()
        mock_reranker = MagicMock()
        
        engine = create_enhanced_retrieval_engine(mock_bm25, mock_faiss, mock_reranker)
        
        assert isinstance(engine, EnhancedRetrievalEngine)
        assert engine.bm25 == mock_bm25


class TestRetrievalResult:
    
    def test_retrieval_result_creation(self):
        """Test création RetrievalResult"""
        result = RetrievalResult(
            chunks=[{"text": "chunk1"}],
            triplets=[{"symptom": "s1"}],
            processing_time=1.5,
            variants_used=2
        )
        
        assert len(result.chunks) == 1
        assert len(result.triplets) == 1
        assert result.processing_time == 1.5
        assert result.variants_used == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])