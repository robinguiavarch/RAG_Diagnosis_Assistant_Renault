"""
Tests pour Equipment Matching
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
    """Test match exact"""
    matcher = create_equipment_matcher()
    
    llm_equipment = "FANUC R-30iB"
    kg_equipments = ["FANUC R-30iB", "ABB IRC5", "KUKA KR C4"]
    
    result = matcher.match_equipment(llm_equipment, kg_equipments)
    assert result == "FANUC R-30iB"
    print("✅ Test exact match")

def test_partial_match():
    """Test match partiel"""
    matcher = create_equipment_matcher()
    
    llm_equipment = "FANUC"
    kg_equipments = ["FANUC R-30iB Controller", "ABB IRC5", "KUKA KR C4"]
    
    result = matcher.match_equipment(llm_equipment, kg_equipments)
    assert result == "FANUC R-30iB Controller"
    print("✅ Test partial match")

def test_no_match():
    """Test pas de match"""
    matcher = create_equipment_matcher()
    
    llm_equipment = "Unknown Robot"
    kg_equipments = ["FANUC R-30iB", "ABB IRC5", "KUKA KR C4"]
    
    result = matcher.match_equipment(llm_equipment, kg_equipments)
    assert result is None
    print("✅ Test no match")

def test_semantic_match():
    """Test match sémantique"""
    matcher = create_equipment_matcher()
    
    llm_equipment = "FANUC robot"
    kg_equipments = ["FANUC R-30iB Robot Controller", "ABB IRC5", "KUKA KR C4"]
    
    result = matcher.match_equipment(llm_equipment, kg_equipments)
    # Devrait matcher FANUC même si pas exact
    assert result is not None
    assert "FANUC" in result
    print("✅ Test semantic match")

def test_top_matches():
    """Test top matches avec scores"""
    matcher = create_equipment_matcher()
    
    llm_equipment = "FANUC"
    kg_equipments = ["FANUC R-30iB", "FANUC R-30iA", "ABB IRC5", "KUKA KR C4"]
    
    matches = matcher.get_best_matches(llm_equipment, kg_equipments, top_k=2)
    
    assert len(matches) >= 1
    assert matches[0][1] > 0.8  # Score élevé pour FANUC
    print(f"✅ Test top matches: {matches}")

def test_utility_function():
    """Test fonction utilitaire"""
    kg_equipments = ["FANUC R-30iB", "ABB IRC5", "KUKA KR C4"]
    
    result = match_single_equipment("FANUC R-30iB", kg_equipments)
    assert result == "FANUC R-30iB"
    print("✅ Test utility function")

def run_all_tests():
    """Lance tous les tests"""
    print("🧪 Tests Equipment Matching")
    print("-" * 40)
    
    try:
        test_exact_match()
        test_partial_match()
        test_no_match()
        test_semantic_match()
        test_top_matches()
        test_utility_function()
        
        print("-" * 40)
        print("✅ Tous les tests passés !")
        
    except Exception as e:
        print(f"❌ Test échoué: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()