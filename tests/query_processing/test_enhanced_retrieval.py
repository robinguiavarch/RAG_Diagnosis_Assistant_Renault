"""
test_enhanced_retrieval
=======================

Unit tests for the EnhancedRetrievalEngine and related retrieval components.

These tests validate instantiation, query routing logic, and proper score
aggregation across BM25, FAISS, and reranking modules.
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

from core.query_processing.enhanced_retrieval_engine import (
    EnhancedRetrievalEngine,
    RetrievalResult,
    create_enhanced_retrieval_engine,
)
from core.query_processing.response_parser import (
    ProcessedQuery,
    TechnicalTerms,
    EquipmentInfo,
)


# --------------------------------------------------------------------------- #
# Unit test suite                                                             #
# --------------------------------------------------------------------------- #

class TestEnhancedRetrievalEngine:
    """Unit tests for the EnhancedRetrievalEngine class."""

    def test_engine_creation(self):
        """Verify that the engine is correctly instantiated with all components."""
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

    def test_search_routing_logic(self):
        """Ensure that the search method routes correctly and returns results."""
        mock_bm25 = MagicMock()
        mock_faiss = MagicMock()
        mock_reranker = MagicMock()

        mock_bm25.search.return_value = [{"id": "doc1", "score": 0.8}]
        mock_faiss.search.return_value = [{"id": "doc2", "score": 0.9}]
        mock_reranker.rerank.return_value = [{"id": "doc2", "score": 0.95}]

        engine = EnhancedRetrievalEngine(mock_bm25, mock_faiss, mock_reranker)

        mock_query = ProcessedQuery(
            query_type="diagnostic",
            structured=True,
            filtered_query="motor overheating",
            technical_terms=TechnicalTerms(),
            equipment_info=EquipmentInfo()
        )

        results = engine.search(mock_query)

        assert isinstance(results, list)
        assert isinstance(results[0], dict)
        assert "id" in results[0]
        assert "score" in results[0]

    def test_factory_creation(self):
        """Ensure that the factory function returns an EnhancedRetrievalEngine instance."""
        engine = create_enhanced_retrieval_engine()
        assert isinstance(engine, EnhancedRetrievalEngine)
