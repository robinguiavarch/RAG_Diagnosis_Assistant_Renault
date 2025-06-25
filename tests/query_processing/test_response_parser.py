"""
test_response_parser
====================

Unit tests for the ResponseParser class and the `parse_llm_response` utility.

This suite checks the ability to decode and convert LLM outputs into structured
instances of ProcessedQuery, TechnicalTerms, and EquipmentInfo.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# --------------------------------------------------------------------------- #
# Project path setup                                                          #
# --------------------------------------------------------------------------- #

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from core.query_processing.response_parser import (
    ResponseParser,
    ProcessedQuery,
    TechnicalTerms,
    EquipmentInfo,
    parse_llm_response,
)


# --------------------------------------------------------------------------- #
# Unit test suite                                                             #
# --------------------------------------------------------------------------- #

class TestResponseParser:
    """Unit tests for LLM response parsing and structured data conversion."""

    def test_parse_valid_json(self):
        """Test that a valid JSON block is correctly parsed into a ProcessedQuery."""
        test_response = '''
        ```json
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
            "filtered_query": "ACAL-006 TPE operation error FANUC-30iB",
            "query_type": "diagnostic",
            "structured": true
        }
        ```'''

        parser = ResponseParser()
        result = parser.parse(test_response)

        assert isinstance(result, ProcessedQuery)
        assert result.query_type == "diagnostic"
        assert result.structured is True
        assert isinstance(result.technical_terms, TechnicalTerms)
        assert isinstance(result.equipment_info, EquipmentInfo)

    def test_parse_llm_response_direct(self):
        """Test the utility function `parse_llm_response` on raw JSON string."""
        raw_response = '''
        {
            "query_type": "symptom",
            "technical_terms": {
                "symptoms": ["noise", "vibration"]
            },
            "structured": true
        }
        '''
        result = parse_llm_response(raw_response)
        assert isinstance(result, ProcessedQuery)
        assert result.query_type == "symptom"
        assert "vibration" in result.technical_terms.symptoms

    def test_parser_handles_invalid_json(self):
        """Ensure the parser raises an error on invalid JSON."""
        invalid_json = "This is not JSON"
        parser = ResponseParser()
        with pytest.raises(ValueError):
            parser.parse(invalid_json)

    def test_parser_missing_fields(self):
        """Ensure missing required fields raise an appropriate error."""
        incomplete_json = '{"structured": true}'
        parser = ResponseParser()
        with pytest.raises(ValueError):
            parser.parse(incomplete_json)
