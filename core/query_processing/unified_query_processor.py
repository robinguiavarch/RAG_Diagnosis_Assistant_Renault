"""
Unified Query Processor: Comprehensive User Query Analysis and Enhancement

This module provides unified processing capabilities for user queries in the RAG diagnosis
system. It integrates LLM-based query analysis with structured parsing to extract technical
terms, equipment information, and generate query variants for enhanced retrieval performance.

Key components:
- LLM-based query processing with external prompt template loading
- Structured parsing of technical terms and equipment information
- Query filtering and variant generation for improved search coverage
- Robust error handling with fallback mechanisms
- Detailed logging for debugging and performance monitoring

Dependencies: yaml, pathlib, llm_client, response_parser
Usage: Import UnifiedQueryProcessor for comprehensive query analysis and enhancement
"""

import os
import yaml
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient
from .response_parser import ResponseParser, ProcessedQuery


class UnifiedQueryProcessor:
    """Unified processor for comprehensive user query analysis and enhancement"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize unified query processor with LLM client and parser.
        
        Sets up the complete query processing pipeline with LLM integration,
        structured parsing, and external prompt template loading.
        
        Args:
            llm_client (Optional[LLMClient]): LLM client instance, creates default if None
        """
        print("Initializing UnifiedQueryProcessor...")
        
        # LLM client
        self.llm_client = llm_client or LLMClient()
        print(f"LLM Client created: {type(self.llm_client)}")
        
        # Parser
        self.parser = ResponseParser()
        print(f"Parser created: {type(self.parser)}")
        
        # Prompt template loading
        self.prompt_template = self._load_prompt_template()
        print(f"Prompt template loaded: {len(self.prompt_template)} characters")
    
    def _load_prompt_template(self) -> str:
        """
        Load external prompt template with fallback mechanism.
        
        Attempts to load the unified query prompt from external file with
        comprehensive error handling and fallback to minimal prompt.
        
        Returns:
            str: Prompt template with query placeholder
        """
        script_dir = Path(__file__).parent.parent.parent
        prompt_path = script_dir / "config" / "prompts" / "unified_query_prompt.txt"
        
        print(f"Attempting to load prompt: {prompt_path}")
        print(f"Absolute path: {prompt_path.absolute()}")
        print(f"File exists: {prompt_path.exists()}")
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"Prompt loaded successfully: {len(content)} characters")
                # Display first 100 characters for content verification
                print(f"Prompt preview: {content[:100]}...")
                return content
        except FileNotFoundError as e:
            print(f"Prompt file not found: {e}")
            print("Using fallback prompt")
            # Minimal default prompt
            return """
You are an expert in industrial equipment troubleshooting. Analyze this query and extract technical information.

USER QUERY: "{raw_query}"

Respond with JSON containing technical_terms, equipment_info, filtered_query, and query_variants.
            """.strip()
        except Exception as e:
            print(f"Error reading prompt: {e}")
            return "Minimal fallback prompt"
    
    def process_user_query(self, raw_query: str) -> ProcessedQuery:
        """
        Process user query with comprehensive LLM analysis.
        
        Performs complete query processing including technical term extraction,
        equipment identification, query filtering, and variant generation using
        LLM-based analysis with structured parsing.
        
        Args:
            raw_query (str): Raw user input query to process
            
        Returns:
            ProcessedQuery: Structured query analysis with technical information
            
        Raises:
            ValueError: If query is empty or invalid
        """
        print(f"\nStarting process_user_query")
        print(f"Query received: '{raw_query[:50]}...'")
        
        if not raw_query or not raw_query.strip():
            print("Empty query detected")
            raise ValueError("Query cannot be empty")
        
        raw_query = raw_query.strip()
        print(f"Query cleaned: {len(raw_query)} characters")
        
        try:
            print("STEP 1: Prompt construction")
            # Using replace() instead of format() to avoid JSON conflicts
            prompt = self.prompt_template.replace("{raw_query}", raw_query)
            print(f"Prompt constructed: {len(prompt)} characters")
            print(f"Prompt preview: {prompt[:200]}...")
            
            print("STEP 2: LLM call")
            # LLM call
            llm_response = self.llm_client.generate(prompt)
            print(f"LLM response received: {len(llm_response)} characters")
            print(f"Response preview: {llm_response[:200]}...")
            
            print("STEP 3: Response parsing")
            # Response parsing
            processed_query = self.parser.parse_llm_response(llm_response, raw_query)
            print(f"Parsing successful")
            print(f"Filtered query result: '{processed_query.filtered_query}'")
            print(f"Number of variants: {len(processed_query.query_variants)}")
            print(f"Equipment detected: {processed_query.equipment_info.primary_equipment}")
            
            return processed_query
            
        except Exception as e:
            print(f"ERROR in process_user_query: {e}")
            print(f"Error type: {type(e)}")
            print("Activating fallback")
            
            # Simple fallback: return original query
            return self._create_fallback_result(raw_query, str(e))
    
    def _create_fallback_result(self, query: str, error_msg: str) -> ProcessedQuery:
        """
        Create fallback result in case of processing errors.
        
        Generates a minimal ProcessedQuery structure when LLM processing fails,
        ensuring system continues to function with basic query information.
        
        Args:
            query (str): Original user query
            error_msg (str): Error message for logging
            
        Returns:
            ProcessedQuery: Minimal fallback query structure
        """
        print(f"FALLBACK activated: {error_msg}")
        
        from .response_parser import TechnicalTerms, EquipmentInfo
        
        fallback_result = ProcessedQuery(
            raw_query=query,
            technical_terms=TechnicalTerms([], [], [], []),
            equipment_info=EquipmentInfo("unknown", "unknown", "unknown"),
            filtered_query=query,
            query_variants=[],
            confidence_score=0.0
        )
        
        print(f"Fallback result created with identical query")
        return fallback_result
    
    def get_config(self):
        """
        Return current processor configuration.
        
        Provides access to current configuration state including LLM settings
        and prompt template status for debugging and monitoring.
        
        Returns:
            dict: Configuration dictionary with LLM config and prompt status
        """
        return {
            "llm_config": self.llm_client.get_config(),
            "prompt_loaded": bool(self.prompt_template)
        }


def create_query_processor() -> UnifiedQueryProcessor:
    """
    Create query processor with default configuration.
    
    Factory function for convenient instantiation of unified query processor
    with standard settings and automatic component initialization.
    
    Returns:
        UnifiedQueryProcessor: Fully configured query processor instance
    """
    print("Creating query processor...")
    return UnifiedQueryProcessor()


def process_single_query(query: str) -> ProcessedQuery:
    """
    Utility function for processing a single query quickly.
    
    Provides streamlined interface for single query processing without
    requiring manual processor instantiation.
    
    Args:
        query (str): User query to process
        
    Returns:
        ProcessedQuery: Processed query with technical analysis
    """
    processor = create_query_processor()
    return processor.process_user_query(query)


if __name__ == "__main__":
    # Simple test
    test_queries = [
        "I got the error ACAL-006 TPE operation error on the FANUC-30iB machine teach pendant. I don't understand why.",
        "motor overheating on robot arm",
        "SYST-001 brake failure"
    ]
    
    try:
        processor = create_query_processor()
        
        print("UnifiedQueryProcessor Test")
        
        # Test one query
        result = processor.process_user_query(test_queries[0])
        
        print(f"Result:")
        print(f"   Filtered query: {result.filtered_query}")
        print(f"   Error codes: {result.technical_terms.error_codes}")
        print(f"   Equipment: {result.equipment_info.primary_equipment}")
        print(f"   Variants: {len(result.query_variants)}")
        
    except Exception as e:
        print(f"Test failed: {e}")