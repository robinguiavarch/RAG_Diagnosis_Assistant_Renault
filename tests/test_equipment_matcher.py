"""
Equipment Matching Test Suite

This module provides comprehensive test coverage for the equipment matching functionality
used in the RAG diagnosis system. The tests validate exact matching, partial matching,
semantic similarity, and scoring mechanisms for equipment identification and mapping
between LLM-extracted equipment names and Knowledge Graph equipment entities.

Key components:
- Exact match validation: Tests precise equipment name matching
- Partial match testing: Validates substring and fuzzy matching capabilities
- Semantic match verification: Tests similarity-based equipment identification
- Scoring system validation: Verifies match confidence scores and ranking
- Utility function testing: Tests standalone equipment matching functions

Dependencies: sys, os, equipment_matcher module
Usage: Execute as standalone script or integrate into larger test suite
Path: tests/test_equipment_matching.py
"""

import sys
import os
sys.path.append('..')

from core.retrieval_graph.equipment_matcher import (
    create_equipment_matcher,
    match_single_equipment
)

def test_exact_match():
    """
    Test exact equipment name matching
    
    Validates that the equipment matcher correctly identifies exact matches
    between LLM-extracted equipment names and Knowledge Graph equipment entries.
    """
    matcher = create_equipment_matcher()
    
    llm_equipment = "FANUC R-30iB"
    kg_equipments = ["FANUC R-30iB", "ABB IRC5", "KUKA KR C4"]
    
    result = matcher.match_equipment(llm_equipment, kg_equipments)
    assert result == "FANUC R-30iB"
    print("Exact match test passed")

def test_partial_match():
    """
    Test partial equipment name matching
    
    Validates that the equipment matcher can identify equipment through partial
    name matching when exact matches are not available but substring matches exist.
    """
    matcher = create_equipment_matcher()
    
    llm_equipment = "FANUC"
    kg_equipments = ["FANUC R-30iB Controller", "ABB IRC5", "KUKA KR C4"]
    
    result = matcher.match_equipment(llm_equipment, kg_equipments)
    assert result == "FANUC R-30iB Controller"
    print("Partial match test passed")

def test_no_match():
    """
    Test handling of unmatched equipment names
    
    Validates that the equipment matcher correctly returns None when no suitable
    match can be found for an equipment name in the Knowledge Graph equipment list.
    """
    matcher = create_equipment_matcher()
    
    llm_equipment = "Unknown Robot"
    kg_equipments = ["FANUC R-30iB", "ABB IRC5", "KUKA KR C4"]
    
    result = matcher.match_equipment(llm_equipment, kg_equipments)
    assert result is None
    print("No match test passed")

def test_semantic_match():
    """
    Test semantic equipment name matching
    
    Validates that the equipment matcher can identify equipment through semantic
    similarity when exact or partial matches are insufficient but contextual
    similarity exists between equipment descriptions.
    """
    matcher = create_equipment_matcher()
    
    llm_equipment = "FANUC robot"
    kg_equipments = ["FANUC R-30iB Robot Controller", "ABB IRC5", "KUKA KR C4"]
    
    result = matcher.match_equipment(llm_equipment, kg_equipments)
    # Should match FANUC even if not exact
    assert result is not None
    assert "FANUC" in result
    print("Semantic match test passed")

def test_top_matches():
    """
    Test top matches retrieval with confidence scores
    
    Validates that the equipment matcher can return multiple candidate matches
    with associated confidence scores, enabling evaluation of match quality
    and selection of best alternatives when multiple options exist.
    """
    matcher = create_equipment_matcher()
    
    llm_equipment = "FANUC"
    kg_equipments = ["FANUC R-30iB", "FANUC R-30iA", "ABB IRC5", "KUKA KR C4"]
    
    matches = matcher.get_best_matches(llm_equipment, kg_equipments, top_k=2)
    
    assert len(matches) >= 1
    assert matches[0][1] > 0.8  # High score for FANUC
    print(f"Top matches test passed: {matches}")

def test_utility_function():
    """
    Test standalone utility function for equipment matching
    
    Validates that the utility function provides correct equipment matching
    functionality without requiring explicit matcher instantiation, enabling
    simple one-off equipment matching operations.
    """
    kg_equipments = ["FANUC R-30iB", "ABB IRC5", "KUKA KR C4"]
    
    result = match_single_equipment("FANUC R-30iB", kg_equipments)
    assert result == "FANUC R-30iB"
    print("Utility function test passed")

def run_all_tests():
    """
    Execute complete equipment matching test suite
    
    Orchestrates execution of all equipment matching tests, providing comprehensive
    validation of the equipment matching system functionality. Reports test results
    and handles any test failures with appropriate error reporting.
    
    Raises:
        Exception: When any test fails, with details about the specific failure
    """
    print("Equipment Matching Test Suite")
    print("-" * 40)
    
    try:
        test_exact_match()
        test_partial_match()
        test_no_match()
        test_semantic_match()
        test_top_matches()
        test_utility_function()
        
        print("-" * 40)
        print("All tests passed successfully")
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()