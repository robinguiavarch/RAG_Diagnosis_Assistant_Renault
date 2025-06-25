"""
Equipment Matcher: LLM Equipment to Knowledge Graph Equipment Mapping

This module provides sophisticated equipment matching capabilities for mapping
LLM-extracted equipment names to standardized equipment identifiers in knowledge
graphs. It implements multi-stage matching strategies including exact matching,
partial string matching, and semantic similarity using cosine similarity calculations.

Key components:
- Multi-stage equipment matching with exact, partial, and semantic approaches
- Configurable similarity thresholds with YAML configuration integration
- Cosine similarity-based semantic matching using SentenceTransformer embeddings
- Neo4j knowledge graph equipment extraction utilities
- Comprehensive testing framework with multiple test scenarios

Dependencies: sentence-transformers, sklearn, numpy, yaml, neo4j
Usage: Import EquipmentMatcher for mapping LLM equipment names to KG equipment identifiers
"""

import os
import yaml
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EquipmentMatcher:
    """
    Simple matcher for LLM equipment to knowledge graph equipment mapping.
    
    Implements hierarchical matching strategy starting with exact matches,
    progressing through partial string matching, and concluding with
    semantic similarity matching for robust equipment identification.
    """
    
    def __init__(self):
        """
        Initialize equipment matcher with lazy model loading and threshold configuration.
        
        Sets up the matcher with deferred SentenceTransformer model loading
        and configurable similarity threshold from YAML settings.
        """
        self.model = None
        self.threshold = self._load_threshold()
    
    def _load_threshold(self) -> float:
        """
        Load similarity threshold from configuration file.
        
        Reads the equipment matching similarity threshold from YAML configuration
        with fallback to default value for robust operation.
        
        Returns:
            float: Similarity threshold for equipment matching (default: 0.9)
        """
        try:
            with open("config/settings.yaml", 'r') as f:
                config = yaml.safe_load(f)
            return config.get("equipment_matching", {}).get("similarity_threshold", 0.9)
        except:
            return 0.9
    
    def _get_model(self):
        """
        Lazy loading of SentenceTransformer model for embedding generation.
        
        Initializes the embedding model on first use with configuration-based
        model selection and fallback to default model for reliability.
        
        Returns:
            SentenceTransformer: Embedding model for semantic similarity calculation
        """
        if self.model is None:
            try:
                with open("config/settings.yaml", 'r') as f:
                    config = yaml.safe_load(f)
                model_name = config["models"]["embedding_model"]
                self.model = SentenceTransformer(model_name)
            except:
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.model
    
    def match_equipment(self, llm_equipment: str, kg_equipments: List[str]) -> Optional[str]:
        """
        Match LLM equipment with knowledge graph equipment using multi-stage approach.
        
        Implements hierarchical matching strategy: exact match, partial string match,
        and semantic similarity match to find the best equipment correspondence
        with configurable threshold validation.
        
        Args:
            llm_equipment (str): Equipment name extracted by LLM
            kg_equipments (List[str]): List of available equipment in knowledge graph
            
        Returns:
            Optional[str]: Best matching KG equipment if similarity exceeds threshold, None otherwise
        """
        if not llm_equipment or not kg_equipments:
            return None
        
        # Cleaning
        llm_clean = llm_equipment.strip().lower()
        kg_clean = [eq.strip().lower() for eq in kg_equipments if eq]
        
        if not kg_clean:
            return None
        
        # Exact match first
        if llm_clean in kg_clean:
            idx = kg_clean.index(llm_clean)
            return kg_equipments[idx]
        
        # Simple partial match
        for i, kg_eq in enumerate(kg_clean):
            if llm_clean in kg_eq or kg_eq in llm_clean:
                return kg_equipments[i]
        
        # Semantic match
        try:
            model = self._get_model()
            
            # Embeddings
            llm_embedding = model.encode([llm_equipment])
            kg_embeddings = model.encode(kg_equipments)
            
            # Cosine similarity
            similarities = cosine_similarity(llm_embedding, kg_embeddings)[0]
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]
            
            if max_similarity >= self.threshold:
                return kg_equipments[max_idx]
            
        except Exception as e:
            print(f"Warning: Equipment matching error: {e}")
        
        return None
    
    def get_best_matches(self, llm_equipment: str, kg_equipments: List[str], 
                        top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Return top-k equipment matches with similarity scores.
        
        Computes semantic similarity scores for all equipment candidates
        and returns the highest scoring matches for analysis and validation.
        
        Args:
            llm_equipment (str): Equipment name from LLM extraction
            kg_equipments (List[str]): Available equipment in knowledge graph
            top_k (int): Number of top matches to return
            
        Returns:
            List[Tuple[str, float]]: Equipment matches with scores, sorted by descending score
        """
        if not llm_equipment or not kg_equipments:
            return []
        
        try:
            model = self._get_model()
            
            llm_embedding = model.encode([llm_equipment])
            kg_embeddings = model.encode(kg_equipments)
            
            similarities = cosine_similarity(llm_embedding, kg_embeddings)[0]
            
            # Sort by descending score
            sorted_indices = np.argsort(similarities)[::-1]
            
            results = []
            for idx in sorted_indices[:top_k]:
                if similarities[idx] >= 0.5:  # Minimum threshold for display
                    results.append((kg_equipments[idx], float(similarities[idx])))
            
            return results
            
        except Exception:
            return []


def create_equipment_matcher() -> EquipmentMatcher:
    """
    Create equipment matcher instance for equipment mapping operations.
    
    Factory function for creating configured equipment matcher instances
    with proper initialization and configuration loading.
    
    Returns:
        EquipmentMatcher: Configured equipment matcher instance
    """
    return EquipmentMatcher()

def match_single_equipment(llm_equipment: str, kg_equipments: List[str]) -> Optional[str]:
    """
    Utility function for quick single equipment matching.
    
    Convenience function for one-off equipment matching operations
    without requiring explicit matcher instance management.
    
    Args:
        llm_equipment (str): Equipment name to match
        kg_equipments (List[str]): Available equipment options
        
    Returns:
        Optional[str]: Best matching equipment or None
    """
    matcher = create_equipment_matcher()
    return matcher.match_equipment(llm_equipment, kg_equipments)

def extract_kg_equipments_from_neo4j(driver, kg_type: str = "dense") -> List[str]:
    """
    Extract all unique equipment from Neo4j knowledge graph.
    
    Queries the knowledge graph database to retrieve all unique equipment
    identifiers for use in equipment matching operations with error handling.
    
    Args:
        driver: Neo4j database driver instance
        kg_type (str): Type of knowledge graph ("dense", "sparse", "dense_sc")
        
    Returns:
        List[str]: Unique equipment identifiers from knowledge graph
    """
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.equipment IS NOT NULL
                RETURN DISTINCT n.equipment AS equipment
                ORDER BY n.equipment
            """)
            
            equipments = [record["equipment"] for record in result if record["equipment"]]
            return equipments
            
    except Exception as e:
        print(f"Warning: Equipment extraction error for {kg_type}: {e}")
        return []

def test_equipment_matcher():
    """
    Simple test function for equipment matcher validation.
    
    Provides comprehensive testing of equipment matching functionality
    with various test cases including exact matches, partial matches,
    and semantic similarity scenarios.
    """
    print("Testing Equipment Matcher")
    
    matcher = create_equipment_matcher()
    
    # Test cases
    test_cases = [
        ("FANUC R-30iB", ["FANUC R-30iB", "ABB IRC5", "KUKA KR C4"]),
        ("fanuc robot", ["FANUC R-30iB", "ABB IRC5", "KUKA KR C4"]),
        ("unknown equipment", ["FANUC R-30iB", "ABB IRC5", "KUKA KR C4"])
    ]
    
    for llm_eq, kg_eqs in test_cases:
        result = matcher.match_equipment(llm_eq, kg_eqs)
        print(f"LLM: '{llm_eq}' → KG: '{result}'")
        
        # Top matches
        matches = matcher.get_best_matches(llm_eq, kg_eqs)
        print(f"  Top matches: {matches}")
        print()

if __name__ == "__main__":
    test_equipment_matcher()