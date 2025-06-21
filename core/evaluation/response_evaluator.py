"""
Évaluateur de réponses - Version complète avec support 4 réponses
Path: core/evaluation/response_evaluator.py
"""

import json
import re
from pathlib import Path
from .llm_judge_client import LLMJudgeClient


class ResponseEvaluator:
    """Évaluateur pour comparer 2 ou 4 réponses RAG simultanément"""
    
    def __init__(self):
        self.judge_client = LLMJudgeClient()
        self.prompt_template = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Charge le prompt d'évaluation depuis le fichier externalisé"""
        try:
            prompt_path = Path("config/prompts/judge_evaluation_prompt.txt")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Prompt minimal par défaut pour 2 réponses
            return """Compare these 2 responses and give scores 0-5:

QUERY: {query}

RESPONSE 1: {response1}

RESPONSE 2: {response2}

Return JSON: {{"score_response_1": X.X, "score_response_2": X.X, "comparative_justification": "brief explanation"}}"""
    
    def evaluate_responses(self, query: str, response1: str, response2: str) -> dict:
        """
        Évalue et compare 2 réponses (fonction originale conservée)
        
        Args:
            query: Question utilisateur
            response1: Première réponse (RAG classique)
            response2: Deuxième réponse (RAG enrichi)
            
        Returns:
            dict: {"score_response_1": float, "score_response_2": float, "comparative_justification": str}
        """
        # Construction du prompt pour 2 réponses
        prompt = self.prompt_template.format(
            query=query,
            response1=response1,
            response2=response2
        )
        
        # Appel LLM
        llm_response = self.judge_client.evaluate(prompt)
        
        # Parse JSON
        try:
            return self._parse_evaluation(llm_response)
        except:
            # Fallback en cas d'erreur
            return {
                "score_response_1": 2.5,
                "score_response_2": 2.5,
                "comparative_justification": "Erreur de parsing de l'évaluation"
            }
    
    def evaluate_4_responses(self, query: str, response_classic: str, response_dense: str, 
                            response_sparse: str, response_dense_sc: str) -> dict:
        """
        🆕 NOUVELLE FONCTION: Évalue et compare 4 réponses RAG simultanément
        Utilise le prompt externalisé judge_evaluation_prompt.txt
        
        Args:
            query: Question utilisateur
            response_classic: Réponse RAG Classique
            response_dense: Réponse RAG + KG Dense
            response_sparse: Réponse RAG + KG Sparse
            response_dense_sc: Réponse RAG + KG Dense S&C
            
        Returns:
            dict: Scores et analyse comparative des 4 approches
        """
        # Construction du prompt 4 réponses avec le template externalisé
        prompt = self.prompt_template.format(
            query=query,
            response_classic=response_classic,
            response_dense=response_dense,
            response_sparse=response_sparse,
            response_dense_sc=response_dense_sc
        )
        
        # Appel LLM
        llm_response = self.judge_client.evaluate(prompt)
        
        # Parse JSON
        try:
            return self._parse_evaluation(llm_response)
        except Exception as e:
            print(f"❌ Erreur parsing évaluation 4 réponses: {e}")
            # Fallback en cas d'erreur
            return {
                "score_classic": 2.5,
                "score_dense": 2.5,
                "score_sparse": 2.5,
                "score_dense_sc": 2.5,
                "best_approach": "Évaluation indisponible (erreur de parsing)",
                "comparative_analysis": "Erreur lors de l'évaluation automatique des réponses"
            }
    
    def _parse_evaluation(self, llm_response: str) -> dict:
        """Parse la réponse JSON du LLM (compatible 2 et 4 réponses)"""
        # Extraction du JSON (même logique que preprocessing)
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_match = re.search(json_pattern, llm_response, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: chercher JSON direct
            json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
            json_match = re.search(json_pattern, llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                raise Exception("Pas de JSON trouvé dans la réponse LLM")
        
        return json.loads(json_str)


def create_response_evaluator() -> ResponseEvaluator:
    """Fonction utilitaire pour créer un évaluateur"""
    return ResponseEvaluator()


# === FONCTION UTILITAIRE POUR TEST RAPIDE ===
def evaluate_responses_quick(query: str, response1: str, response2: str) -> dict:
    """Fonction utilitaire pour évaluation rapide de 2 réponses"""
    evaluator = create_response_evaluator()
    return evaluator.evaluate_responses(query, response1, response2)


def evaluate_4_responses_quick(query: str, response_classic: str, response_dense: str, 
                              response_sparse: str, response_dense_sc: str) -> dict:
    """Fonction utilitaire pour évaluation rapide de 4 réponses"""
    evaluator = create_response_evaluator()
    return evaluator.evaluate_4_responses(query, response_classic, response_dense, response_sparse, response_dense_sc)


if __name__ == "__main__":
    # Test simple de l'évaluateur 4 réponses
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
        
        print("🧪 Test évaluation 4 réponses:")
        print(f"   Classic: {result.get('score_classic', 'N/A')}")
        print(f"   Dense: {result.get('score_dense', 'N/A')}")
        print(f"   Sparse: {result.get('score_sparse', 'N/A')}")
        print(f"   Dense S&C: {result.get('score_dense_sc', 'N/A')}")
        print(f"   Meilleure approche: {result.get('best_approach', 'N/A')}")
        print("✅ Test réussi")
        
    except Exception as e:
        print(f"❌ Test échoué: {e}")