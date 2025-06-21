"""
Parser simple pour les réponses JSON du LLM de préprocessing
Focus sur l'essentiel - extraction et validation minimale
"""

import json
import re
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class TechnicalTerms:
    """Termes techniques extraits"""
    error_codes: List[str]
    components: List[str]
    equipment_models: List[str]
    technical_keywords: List[str]


@dataclass
class EquipmentInfo:
    """Informations d'équipement"""
    primary_equipment: str
    equipment_type: str
    manufacturer: str
    series: str = "unknown"


@dataclass
class ProcessedQuery:
    """Requête traitée complète"""
    raw_query: str
    technical_terms: TechnicalTerms
    equipment_info: EquipmentInfo
    filtered_query: str
    query_variants: List[str]
    confidence_score: float = 0.0
    
    def get_primary_query(self) -> str:
        """Retourne la requête principale (filtrée si disponible, sinon brute)"""
        return self.filtered_query if self.filtered_query else self.raw_query
    
    def get_all_queries(self) -> List[str]:
        """Retourne toutes les variantes de requête"""
        queries = [self.get_primary_query()]
        queries.extend(self.query_variants)
        return list(set(queries))  # Déduplication


class ResponseParser:
    """Parser simple pour les réponses LLM"""
    
    def parse_llm_response(self, llm_response: str, raw_query: str) -> ProcessedQuery:
        """
        Parse la réponse LLM en ProcessedQuery
        
        Args:
            llm_response: Réponse brute du LLM
            raw_query: Requête utilisateur originale
            
        Returns:
            ProcessedQuery: Objet structuré
        """
        try:
            # Extraction du JSON
            parsed_json = self._extract_json(llm_response)
            
            # Construction de l'objet
            return self._build_processed_query(parsed_json, raw_query)
            
        except Exception as e:
            # Fallback simple en cas d'échec
            return self._create_fallback_result(raw_query, str(e))
    
    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extrait le JSON de la réponse LLM"""
        # Recherche du bloc JSON
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: chercher un JSON direct
            json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                raise Exception("Aucun JSON trouvé dans la réponse")
        
        return json.loads(json_str)
    
    def _build_processed_query(self, data: Dict[str, Any], raw_query: str) -> ProcessedQuery:
        """Construit ProcessedQuery à partir des données JSON"""
        # Valeurs par défaut si champs manquants
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
        """Crée un résultat de fallback minimal"""
        return ProcessedQuery(
            raw_query=raw_query,
            technical_terms=TechnicalTerms([], [], [], []),
            equipment_info=EquipmentInfo("unknown", "unknown", "unknown"),
            filtered_query=raw_query,
            query_variants=[],
            confidence_score=0.0
        )


# Fonction utilitaire
def parse_llm_response(llm_response: str, raw_query: str) -> ProcessedQuery:
    """Fonction utilitaire pour parser une réponse LLM"""
    parser = ResponseParser()
    return parser.parse_llm_response(llm_response, raw_query)


if __name__ == "__main__":
    # Test simple
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
        print(f"✅ Test réussi:")
        print(f"   Requête filtrée: {parsed.filtered_query}")
        print(f"   Codes d'erreur: {parsed.technical_terms.error_codes}")
        print(f"   Équipement: {parsed.equipment_info.primary_equipment}")
    except Exception as e:
        print(f"❌ Test échoué: {e}")