"""
test_unified_processor
======================

Unit tests for the UnifiedQueryProcessor class and its associated methods.
These tests validate initialization, query processing behavior, and response
parsing from mocked LLM calls.
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

from core.query_processing.unified_query_processor import (
    UnifiedQueryProcessor,
    create_query_processor,
    process_single_query,
)
from core.query_processing.response_parser import (
    ProcessedQuery,
    TechnicalTerms,
    EquipmentInfo,
)


# --------------------------------------------------------------------------- #
# Unit test suite                                                             #
# --------------------------------------------------------------------------- #

class TestUnifiedQueryProcessor:
    """Unit tests for the UnifiedQueryProcessor class."""

    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_processor_creation(self, mock_llm_client):
        """Ensure that UnifiedQueryProcessor initializes correctly with all components."""
        processor = UnifiedQueryProcessor()

        assert processor.llm_client is not None
        assert processor.parser is not None
        assert processor.prompt_template is not None

    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_process_empty_query(self, mock_llm_client):
        """Ensure that an empty query returns None or expected fallback behavior."""
        processor = UnifiedQueryProcessor()
        result = processor.process("")
        assert result is None

    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_process_mocked_query(self, mock_llm_client):
        """Test processing of a mocked query and ensure returned structure matches expectations."""

        mock_response = MagicMock()
        mock_response.content = """
        {
            "query_type": "technical",
            "equipment_info": {"name": "compressor"},
            "technical_terms": {"symptoms": ["vibration"], "causes": ["misalignment"]},
            "structured": true
        }
        """
        mock_llm_client.return_value.generate.return_value = mock_response

        processor = UnifiedQueryProcessor()
        result = processor.process("Why is my compressor vibrating?")

        assert isinstance(result, ProcessedQuery)
        assert result.query_type == "technical"
        assert isinstance(result.equipment_info, EquipmentInfo)
        assert isinstance(result.technical_terms, TechnicalTerms)

    def test_create_query_processor(self):
        """Validate the factory method returns a properly configured processor."""
        processor = create_query_processor()
        assert isinstance(processor, UnifiedQueryProcessor)

    @patch('core.query_processing.unified_query_processor.LLMClient')
    def test_process_single_query(self, mock_llm_client):
        """Ensure that the process_single_query wrapper behaves consistently."""
        mock_response = MagicMock()
        mock_response.content = '{"query_type": "diagnostic", "structured": true}'
        mock_llm_client.return_value.generate.return_value = mock_response

        result = process_single_query("There is a noise in the turbine.")
        assert isinstance(result, ProcessedQuery)
        assert result.query_type == "diagnostic"
