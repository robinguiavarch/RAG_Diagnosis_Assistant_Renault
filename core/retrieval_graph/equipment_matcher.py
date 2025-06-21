"""
Equipment Matcher - Version Ultra-Simple
Matching LLM equipment avec equipments des KGs (cosine similarity)
Path: core/retrieval_graph/equipment_matcher.py
"""

import os
import yaml
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EquipmentMatcher:
    """Matcher simple pour equipment LLM → KG"""
    
    def __init__(self):
        self.model = None
        self.threshold = self._load_threshold()
    
    def _load_threshold(self) -> float:
        """Charge le seuil depuis config"""
        try:
            with open("config/settings.yaml", 'r') as f:
                config = yaml.safe_load(f)
            return config.get("equipment_matching", {}).get("similarity_threshold", 0.9)
        except:
            return 0.9
    
    def _get_model(self):
        """Lazy loading du modèle"""
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
        Matche un equipment LLM avec les equipments du KG
        
        Args:
            llm_equipment: Equipment extrait par LLM
            kg_equipments: Liste equipments disponibles dans le KG
            
        Returns:
            str: Equipment KG le plus proche si match > threshold, sinon None
        """
        if not llm_equipment or not kg_equipments:
            return None
        
        # Nettoyage
        llm_clean = llm_equipment.strip().lower()
        kg_clean = [eq.strip().lower() for eq in kg_equipments if eq]
        
        if not kg_clean:
            return None
        
        # Match exact d'abord
        if llm_clean in kg_clean:
            idx = kg_clean.index(llm_clean)
            return kg_equipments[idx]
        
        # Match partiel simple
        for i, kg_eq in enumerate(kg_clean):
            if llm_clean in kg_eq or kg_eq in llm_clean:
                return kg_equipments[i]
        
        # Match sémantique
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
            print(f"⚠️ Erreur equipment matching: {e}")
        
        return None
    
    def get_best_matches(self, llm_equipment: str, kg_equipments: List[str], 
                        top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retourne les top-k matches avec scores
        
        Returns:
            List[Tuple[str, float]]: (equipment, score) triés par score décroissant
        """
        if not llm_equipment or not kg_equipments:
            return []
        
        try:
            model = self._get_model()
            
            llm_embedding = model.encode([llm_equipment])
            kg_embeddings = model.encode(kg_equipments)
            
            similarities = cosine_similarity(llm_embedding, kg_embeddings)[0]
            
            # Tri par score décroissant
            sorted_indices = np.argsort(similarities)[::-1]
            
            results = []
            for idx in sorted_indices[:top_k]:
                if similarities[idx] >= 0.5:  # Seuil minimum pour affichage
                    results.append((kg_equipments[idx], float(similarities[idx])))
            
            return results
            
        except Exception:
            return []


# === FONCTIONS UTILITAIRES ===

def create_equipment_matcher() -> EquipmentMatcher:
    """Crée un matcher equipment"""
    return EquipmentMatcher()

def match_single_equipment(llm_equipment: str, kg_equipments: List[str]) -> Optional[str]:
    """Fonction utilitaire pour match rapide"""
    matcher = create_equipment_matcher()
    return matcher.match_equipment(llm_equipment, kg_equipments)

def extract_kg_equipments_from_neo4j(driver, kg_type: str = "dense") -> List[str]:
    """
    Extrait tous les equipments uniques d'un KG Neo4j
    
    Args:
        driver: Driver Neo4j
        kg_type: Type de KG ("dense", "sparse", "dense_sc")
        
    Returns:
        List[str]: Equipments uniques du KG
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
        print(f"⚠️ Erreur extraction equipments {kg_type}: {e}")
        return []

def test_equipment_matcher():
    """Test simple du matcher"""
    print("🧪 Test Equipment Matcher")
    
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