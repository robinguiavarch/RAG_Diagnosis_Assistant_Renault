"""
Response Parser: Structured LLM Response Processing and Validation

This module provides comprehensive parsing capabilities for LLM preprocessing responses
in the RAG diagnosis system. It extracts and validates technical information from JSON
responses, creating structured data objects for downstream processing with robust error
handling and fallback mechanisms.

Key components:
- JSON extraction from LLM responses with multiple pattern matching strategies
- Structured data validation and object construction for technical analysis
- Comprehensive fallback mechanisms for parsing failures
- Technical term categorization and equipment information extraction
- Query variant processing and deduplication capabilities

Dependencies: json, re, dataclasses, typing
Usage: Import ResponseParser for structured LLM response processing and validation
"""

import json
import re
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class TechnicalTerms:
    """Container for extracted technical terminology and identifiers"""
    error_codes: List[str]
    components: List[str]
    equipment_models: List[str]
    technical_keywords: List[str]


@dataclass
class EquipmentInfo:
    """Structured equipment information with hierarchical classification"""
    primary_equipment: str
    equipment_type: str
    manufacturer: str
    series: str = "unknown"


@dataclass
class ProcessedQuery:
    """Complete processed query structure with technical analysis"""
    raw_query: str
    technical_terms: TechnicalTerms
    equipment_info: EquipmentInfo
    filtered_query: str
    query_variants: List[str]
    confidence_score: float = 0.0
    
    def get_primary_query(self) -> str:
        """
        Return primary query for processing.
        
        Provides the filtered query if available, otherwise returns the raw query
        for consistent downstream processing.
        
        Returns:
            str: Primary query text (filtered or raw)
        """
        return self.filtered_query if self.filtered_query else self.raw_query
    
    def get_all_queries(self) -> List[str]:
        """
        Return all query variants with deduplication.
        
        Combines the primary query with all variants, removing duplicates
        to provide comprehensive query coverage for retrieval.
        
        Returns:
            List[str]: Deduplicated list of all query variants
        """
        queries = [self.get_primary_query()]
        queries.extend(self.query_variants)
        return list(set(queries))  # Deduplication


class ResponseParser:
    """Simple parser for LLM preprocessing responses with robust error handling"""
    
    def parse_llm_response(self, llm_response: str, raw_query: str) -> ProcessedQuery:
        """
        Parse LLM response into structured ProcessedQuery object.
        
        Extracts JSON data from LLM response and constructs a structured query
        object with technical analysis, equipment information, and query variants.
        
        Args:
            llm_response (str): Raw LLM response containing JSON data
            raw_query (str): Original user query for fallback reference
            
        Returns:
            ProcessedQuery: Structured query analysis object
            
        Raises:
            Exception: If parsing fails, returns fallback result instead of raising
        """
        try:
            # JSON extraction
            parsed_json = self._extract_json(llm_response)
            
            # Object construction
            return self._build_processed_query(parsed_json, raw_query)
            
        except Exception as e:
            # Simple fallback on failure
            return self._create_fallback_result(raw_query, str(e))
    
    def _extract_json(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON data from LLM response using multiple strategies.
        
        Attempts to locate and parse JSON content using code block patterns
        and direct JSON detection with comprehensive error handling.
        
        Args:
            response (str): Raw LLM response text
            
        Returns:
            Dict[str, Any]: Parsed JSON data
            
        Raises:
            Exception: If no valid JSON can be extracted
        """
        # Search for JSON block
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: search for direct JSON
            json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                raise Exception("No JSON found in response")
        
        return json.loads(json_str)
    
    def _build_processed_query(self, data: Dict[str, Any], raw_query: str) -> ProcessedQuery:
        """
        Construct ProcessedQuery from parsed JSON data.
        
        Builds structured query object with technical analysis, handling missing
        fields gracefully with appropriate defaults.
        
        Args:
            data (Dict[str, Any]): Parsed JSON data from LLM response
            raw_query (str): Original user query
            
        Returns:
            ProcessedQuery: Complete structured query object
        """
        # Default values for missing fields
        tech_terms_data = data.get("technical_terms", {})
        equipment_data = data.get("equipment_info", {})
        
        technical_terms = TechnicalTerms(
            error_codes=tech_terms_data.get("error_codes", []),
            components=tech_terms_data.get("components", []),
            equipment_models=tech_terms_data.get("equipment_models", []),
            technical_keywords=tech_terms_data.get("technical_keywords", [])
        )
        
        equipment_info = EquipmentInfo(
            primary_equipment=equipment_data.get("primary_equipment", "unknown"),
            equipment_type=equipment_data.get("equipment_type", "unknown"),
            manufacturer=equipment_data.get("manufacturer", "unknown"),
            series=equipment_data.get("series", "unknown")
        )
        
        return ProcessedQuery(
            raw_query=raw_query,
            technical_terms=technical_terms,
            equipment_info=equipment_info,
            filtered_query=data.get("filtered_query", raw_query),
            query_variants=data.get("query_variants", []),
            confidence_score=data.get("confidence_score", 0.5)
        )
    
    def _create_fallback_result(self, raw_query: str, error_msg: str) -> ProcessedQuery:
        """
        Create minimal fallback result for parsing failures.
        
        Generates a basic ProcessedQuery structure when parsing fails,
        ensuring system continues to function with original query.
        
        Args:
            raw_query (str): Original user query
            error_msg (str): Error message for logging
            
        Returns:
            ProcessedQuery: Minimal fallback query structure
        """
        return ProcessedQuery(
            raw_query=raw_query,
            technical_terms=TechnicalTerms([], [], [], []),
            equipment_info=EquipmentInfo("unknown", "unknown", "unknown"),
            filtered_query=raw_query,
            query_variants=[],
            confidence_score=0.0
        )


def parse_llm_response(llm_response: str, raw_query: str) -> ProcessedQuery:
    """
    Utility function for parsing LLM response.
    
    Provides convenient interface for LLM response parsing without requiring
    manual parser instantiation.
    
    Args:
        llm_response (str): Raw LLM response to parse
        raw_query (str): Original user query
        
    Returns:
        ProcessedQuery: Structured query analysis result
    """
    parser = ResponseParser()
    return parser.parse_llm_response(llm_response, raw_query)


if __name__ == "__main__":
    # Simple test
    test_response = '''
    ```json
    {
        "technical_terms": {
            "error_codes": ["ACAL-006"],
            "components": ["TPE", "teach pendant"],
            "equipment_models": ["FANUC-30iB"],
            "technical_keywords": ["operation error"]
        },
        "equipment_info": {
            "primary_equipment": "FANUC R-30iB",
            "equipment_type": "industrial_robot",
            "manufacturer": "FANUC",
            "series": "R-30iB series"
        },
        "filtered_query": "ACAL-006 TPE operation error FANUC R-30iB",
        "query_variants": [
            "ACAL-006 teach pendant error FANUC",
            "TPE operation failure FANUC robot"
        ],
        "confidence_score": 0.95
    }
    ```
    '''
    
    test_query = "I got ACAL-006 TPE operation error on FANUC machine. I'm lost."
    
    try:
        parsed = parse_llm_response(test_response, test_query)
        print(f"Test successful:")
        print(f"   Filtered query: {parsed.filtered_query}")
        print(f"   Error codes: {parsed.technical_terms.error_codes}")
        print(f"   Equipment: {parsed.equipment_info.primary_equipment}")
    except Exception as e:
        print(f"Test failed: {e}")