"""
test_integration
================

Basic integration tests for the `query_processing` module.

These tests validate functional connectivity between components including
quick setup, unified processor, and single query processing behavior.
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

from core.query_processing import (
    quick_setup,
    process_single_query,
    create_enhanced_retrieval_engine,
)


# --------------------------------------------------------------------------- #
# Integration test suite                                                      #
# --------------------------------------------------------------------------- #

class TestIntegration:
    """Integration tests for query processing pipeline."""

    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_quick_setup(self, mock_llm_client):
        """Ensure that quick_setup initializes a processor with required interface."""
        processor = quick_setup()

        assert processor is not None
        assert hasattr(processor, 'process_user_query')

    @patch('core.query_processing.unified_query_processor.create_query_processor')
    def test_process_single_query_integration(self, mock_create_processor):
        """Test integration of the process_single_query wrapper with mocked processor."""
        mock_processor = MagicMock()
        mock_processor.process.return_value.query_type = "diagnostic"
        mock_create_processor.return_value = mock_processor

        result = process_single_query("Overheating detected on motor.")
        assert result.query_type == "diagnostic"

    def test_create_enhanced_engine(self):
        """Check if the enhanced retrieval engine is properly instantiated."""
        engine = create_enhanced_retrieval_engine()
        assert hasattr(engine, "search")
        assert callable(engine.search)
