"""
Tests simples pour le Response Parser
"""

import pytest
import json
import sys
import os

# Ajout du path pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from core.query_processing.response_parser import (
    ResponseParser, ProcessedQuery, TechnicalTerms, EquipmentInfo, parse_llm_response
)


class TestResponseParser:
    
    def test_parse_valid_json(self):
        """Test parsing JSON valide"""
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
            "filtered_query": "ACAL-006 TPE operation error FANUC",
            "query_variants": ["ACAL-006 teach pendant error"],
            "confidence_score": 0.9
        }
        ```
        '''
        
        parser = ResponseParser()
        result = parser.parse_llm_response(test_response, "test query")
        
        assert isinstance(result, ProcessedQuery)
        assert result.technical_terms.error_codes == ["ACAL-006"]
        assert result.equipment_info.manufacturer == "FANUC"
        assert result.filtered_query == "ACAL-006 TPE operation error FANUC"
    
    def test_parse_invalid_json(self):
        """Test fallback avec JSON invalide"""
        parser = ResponseParser()
        result = parser.parse_llm_response("invalid json", "test query")
        
        assert isinstance(result, ProcessedQuery)
        assert result.raw_query == "test query"
        assert result.filtered_query == "test query"  # Fallback
    
    def test_processed_query_methods(self):
        """Test méthodes de ProcessedQuery"""
        terms = TechnicalTerms(["CODE-1"], ["motor"], ["FANUC"], ["error"])
        equipment = EquipmentInfo("FANUC R-30iB", "robot", "FANUC")
        
        query = ProcessedQuery(
            raw_query="original",
            technical_terms=terms,
            equipment_info=equipment,
            filtered_query="filtered",
            query_variants=["variant1", "variant2"]
        )
        
        assert query.get_primary_query() == "filtered"
        all_queries = query.get_all_queries()
        assert "filtered" in all_queries
        assert "variant1" in all_queries
        assert len(all_queries) == 3  # Déduplication
    
    def test_utility_function(self):
        """Test fonction utilitaire"""
        test_json = '{"technical_terms": {"error_codes": [], "components": [], "equipment_models": [], "technical_keywords": []}, "equipment_info": {"primary_equipment": "test", "equipment_type": "test", "manufacturer": "test"}, "filtered_query": "test", "query_variants": []}'
        
        result = parse_llm_response(test_json, "test query")
        assert isinstance(result, ProcessedQuery)


class TestDataStructures:
    
    def test_technical_terms(self):
        """Test structure TechnicalTerms"""
        terms = TechnicalTerms(
            error_codes=["ACAL-006", "SYST-001"],
            components=["TPE", "motor"],
            equipment_models=["FANUC-30iB"],
            technical_keywords=["error", "operation"]
        )
        
        assert len(terms.error_codes) == 2
        assert "TPE" in terms.components
    
    def test_equipment_info(self):
        """Test structure EquipmentInfo"""
        equipment = EquipmentInfo(
            primary_equipment="FANUC R-30iB",
            equipment_type="industrial_robot",
            manufacturer="FANUC",
            series="R-30iB"
        )
        
        assert equipment.manufacturer == "FANUC"
        assert equipment.equipment_type == "industrial_robot"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])