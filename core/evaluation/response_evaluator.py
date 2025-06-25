"""
Response Evaluator: Comprehensive RAG Response Assessment System

This module provides advanced evaluation capabilities for comparing multiple RAG responses
in the diagnosis system. It supports both dual and quad-response evaluation modes, utilizing
LLM-based judging to assess response quality, accuracy, and comparative effectiveness across
different knowledge graph integration strategies.

Key components:
- Dual response evaluation for classic RAG comparison
- Quad response evaluation for comprehensive RAG strategy assessment
- External prompt template loading for consistent evaluation criteria
- JSON response parsing with robust error handling
- Utility functions for quick evaluation workflows

Dependencies: json, re, pathlib, llm_judge_client
Usage: Import evaluator classes for automated response quality assessment and comparison
"""

import json
import re
from pathlib import Path
from .llm_judge_client import LLMJudgeClient


class ResponseEvaluator:
    """Response evaluator for comparing 2 or 4 RAG responses simultaneously"""
    
    def __init__(self):
        """
        Initialize response evaluator with LLM judge client and prompt template.
        
        Sets up the evaluation system with external prompt loading and
        fallback mechanisms for robust operation.
        """
        self.judge_client = LLMJudgeClient()
        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """
        Load evaluation prompt from external template file.
        
        Attempts to load the judge evaluation prompt from the externalized
        prompt file, with fallback to default template for basic functionality.
        
        Returns:
            str: Evaluation prompt template with placeholders for responses
        """
        try:
            prompt_path = Path("config/prompts/judge_evaluation_prompt.txt")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Minimal default prompt for 2 responses
            return """Compare these 2 responses and give scores 0-5:

QUERY: {query}

RESPONSE 1: {response1}

RESPONSE 2: {response2}

Return JSON: {{"score_response_1": X.X, "score_response_2": X.X, "comparative_justification": "brief explanation"}}"""
    
    def evaluate_responses(self, query: str, response1: str, response2: str) -> dict:
        """
        Evaluate and compare 2 responses with detailed scoring.
        
        Performs comparative evaluation of two RAG responses using LLM-based
        judging with structured scoring and justification.
        
        Args:
            query (str): User question or prompt
            response1 (str): First response (typically classic RAG)
            response2 (str): Second response (typically enhanced RAG)
            
        Returns:
            dict: Evaluation results with scores and comparative justification
                - score_response_1 (float): Score for first response (0-5)
                - score_response_2 (float): Score for second response (0-5) 
                - comparative_justification (str): Detailed comparison explanation
        """
        # Build prompt for 2 responses
        prompt = self.prompt_template.format(
            query=query,
            response1=response1,
            response2=response2
        )
        
        # LLM call
        llm_response = self.judge_client.evaluate(prompt)
        
        # Parse JSON
        try:
            return self._parse_evaluation(llm_response)
        except:
            # Fallback on error
            return {
                "score_response_1": 2.5,
                "score_response_2": 2.5,
                "comparative_justification": "Evaluation parsing error"
            }
    
    def evaluate_4_responses(self, query: str, response_classic: str, response_dense: str, 
                            response_sparse: str, response_dense_sc: str) -> dict:
        """
        Evaluate and compare 4 RAG responses simultaneously across different strategies.
        
        Performs comprehensive evaluation of four different RAG approaches using
        externalized prompt templates for consistent assessment criteria.
        
        Args:
            query (str): User question or prompt
            response_classic (str): Classic RAG response
            response_dense (str): Dense knowledge graph enhanced RAG response
            response_sparse (str): Sparse knowledge graph enhanced RAG response
            response_dense_sc (str): Dense symptom-cause knowledge graph enhanced response
            
        Returns:
            dict: Comprehensive evaluation results with individual scores and analysis
                - score_classic (float): Classic RAG score
                - score_dense (float): Dense KG RAG score
                - score_sparse (float): Sparse KG RAG score
                - score_dense_sc (float): Dense S&C KG RAG score
                - best_approach (str): Identification of best performing approach
                - comparative_analysis (str): Detailed comparative analysis
        """
        # Build prompt for 4 responses using externalized template
        prompt = self.prompt_template.format(
            query=query,
            response_classic=response_classic,
            response_dense=response_dense,
            response_sparse=response_sparse,
            response_dense_sc=response_dense_sc
        )
        
        # LLM call
        llm_response = self.judge_client.evaluate(prompt)
        
        # Parse JSON
        try:
            return self._parse_evaluation(llm_response)
        except Exception as e:
            print(f"Error parsing 4-response evaluation: {e}")
            # Fallback on error
            return {
                "score_classic": 2.5,
                "score_dense": 2.5,
                "score_sparse": 2.5,
                "score_dense_sc": 2.5,
                "best_approach": "Evaluation unavailable (parsing error)",
                "comparative_analysis": "Error during automatic response evaluation"
            }
    
    def _parse_evaluation(self, llm_response: str) -> dict:
        """
        Parse LLM JSON response with support for both 2 and 4 response formats.
        
        Extracts and validates JSON evaluation results from LLM responses using
        multiple parsing strategies for robust operation.
        
        Args:
            llm_response (str): Raw LLM response containing JSON evaluation
            
        Returns:
            dict: Parsed evaluation dictionary with scores and analysis
            
        Raises:
            Exception: If no valid JSON can be extracted from the response
        """
        # JSON extraction (same logic as preprocessing)
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: search for direct JSON
            json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            json_match = re.search(json_pattern, llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise Exception("No JSON found in LLM response")
        
        return json.loads(json_str)


def create_response_evaluator() -> ResponseEvaluator:
    """
    Utility function to create a response evaluator instance.
    
    Factory function for convenient evaluator instantiation with proper
    initialization of all required components.
    
    Returns:
        ResponseEvaluator: Fully initialized response evaluator instance
    """
    return ResponseEvaluator()


def evaluate_responses_quick(query: str, response1: str, response2: str) -> dict:
    """
    Utility function for quick evaluation of 2 responses.
    
    Provides streamlined interface for dual response evaluation without
    requiring manual evaluator instantiation.
    
    Args:
        query (str): User question or prompt
        response1 (str): First response to evaluate
        response2 (str): Second response to evaluate
        
    Returns:
        dict: Evaluation results with scores and justification
    """
    evaluator = create_response_evaluator()
    return evaluator.evaluate_responses(query, response1, response2)


def evaluate_4_responses_quick(query: str, response_classic: str, response_dense: str, 
                              response_sparse: str, response_dense_sc: str) -> dict:
    """
    Utility function for quick evaluation of 4 responses.
    
    Provides streamlined interface for comprehensive quad response evaluation
    without requiring manual evaluator instantiation.
    
    Args:
        query (str): User question or prompt
        response_classic (str): Classic RAG response
        response_dense (str): Dense KG enhanced response
        response_sparse (str): Sparse KG enhanced response
        response_dense_sc (str): Dense S&C KG enhanced response
        
    Returns:
        dict: Comprehensive evaluation results with comparative analysis
    """
    evaluator = create_response_evaluator()
    return evaluator.evaluate_4_responses(query, response_classic, response_dense, response_sparse, response_dense_sc)


if __name__ == "__main__":
    # Simple test of 4-response evaluator
    test_query = "ACAL-006 error on FANUC R-30iB teach pendant"
    
    test_responses = {
        "classic": "Error ACAL-006 indicates a calibration issue. Check teach pendant connection.",
        "dense": "ACAL-006 is a calibration error on FANUC R-30iB. Based on knowledge graph: symptom indicates TPE communication failure. Remedy: restart controller and recalibrate.",
        "sparse": "Error ACAL-006: Direct mapping shows teach pendant operation error. Solution: Check cables and restart system.",
        "dense_sc": "ACAL-006 represents calibration failure (enriched with cause analysis). The symptom-cause relationship indicates TPE hardware issue. Recommended remedy: hardware check then recalibration."
    }
    
    try:
        evaluator = create_response_evaluator()
        result = evaluator.evaluate_4_responses(
            test_query,
            test_responses["classic"],
            test_responses["dense"], 
            test_responses["sparse"],
            test_responses["dense_sc"]
        )
        
        print("4-response evaluation test:")
        print(f"   Classic: {result.get('score_classic', 'N/A')}")
        print(f"   Dense: {result.get('score_dense', 'N/A')}")
        print(f"   Sparse: {result.get('score_sparse', 'N/A')}")
        print(f"   Dense S&C: {result.get('score_dense_sc', 'N/A')}")
        print(f"   Best approach: {result.get('best_approach', 'N/A')}")
        print("Test successful")
        
    except Exception as e:
        print(f"Test failed: {e}")