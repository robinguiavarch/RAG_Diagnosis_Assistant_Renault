"""
Client LLM amélioré pour l'évaluation des réponses
Version avec vérification de cohérence et paramètres optimisés
"""

import os
import yaml
import json
from dotenv import load_dotenv
from openai import OpenAI
from difflib import SequenceMatcher
from typing import List

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class LLMJudgeClient:
    """Client LLM juge avec vérification de cohérence"""
    
    def __init__(self):
        # Chargement config depuis settings.yaml
        try:
            with open("config/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            
            judge_cfg = settings.get("evaluation", {}).get("llm_judge", {})
            self.model = judge_cfg.get("model", "gpt-4o")  # 🔧 UPGRADE par défaut
            self.temperature = judge_cfg.get("temperature", 0.0)  # 🔧 FIX déterminisme
            self.max_tokens = judge_cfg.get("max_tokens", 500)  # 🔧 INCREASE
            
            # 🆕 NOUVEAUX PARAMÈTRES pour cohérence
            self.seed = judge_cfg.get("seed", 42)
            self.retry_on_inconsistency = judge_cfg.get("retry_on_inconsistency", True)
            self.max_retries = judge_cfg.get("max_retries", 2)
            self.similarity_threshold = judge_cfg.get("similarity_threshold", 0.9)
            self.max_score_difference = judge_cfg.get("max_score_difference", 0.3)
            
            print(f"🎯 LLMJudgeClient initialisé:")
            print(f"   🤖 Modèle: {self.model}")
            print(f"   🌡️ Temperature: {self.temperature}")
            print(f"   📏 Max tokens: {self.max_tokens}")
            print(f"   🔄 Retry incohérence: {self.retry_on_inconsistency}")
            
        except FileNotFoundError:
            # Config par défaut améliorée
            print("⚠️ settings.yaml non trouvé, utilisation config par défaut améliorée")
            self.model = "gpt-4o"
            self.temperature = 0.0
            self.max_tokens = 500
            self.seed = 42
            self.retry_on_inconsistency = True
            self.max_retries = 2
            self.similarity_threshold = 0.9
            self.max_score_difference = 0.3
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité entre deux textes"""
        return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()
    
    def extract_responses_from_prompt(self, prompt: str) -> List[str]:
        """Extrait les 4 réponses depuis le prompt pour vérification similarité"""
        try:
            responses = []
            lines = prompt.split('\n')
            current_response = ""
            capturing = False
            
            for line in lines:
                if "RESPONSE" in line and any(keyword in line for keyword in ["RAG", "CLASSIQUE", "DENSE", "SPARSE"]):
                    if current_response and capturing:
                        responses.append(current_response.strip())
                    current_response = ""
                    capturing = True
                    continue
                elif capturing and (line.startswith("RESPONSE") or "Provide JSON" in line or "EVALUATION CRITERIA" in line):
                    if current_response:
                        responses.append(current_response.strip())
                    if "Provide JSON" in line:
                        break
                    current_response = ""
                    capturing = True
                elif capturing and line.strip():
                    current_response += line.strip() + " "
            
            # Ajouter la dernière réponse si nécessaire
            if current_response and capturing:
                responses.append(current_response.strip())
            
            return responses[:4]  # Max 4 réponses
            
        except Exception as e:
            print(f"⚠️ Erreur extraction réponses: {e}")
            return []
    
    def check_response_consistency(self, responses: List[str], scores: List[float]) -> bool:
        """Vérifie la cohérence entre réponses similaires et leurs scores"""
        if len(responses) != 4 or len(scores) != 4:
            return True  # Skip si pas 4 réponses
            
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity_score = self.similarity(responses[i], responses[j])
                
                if similarity_score >= self.similarity_threshold:
                    score_diff = abs(scores[i] - scores[j])
                    if score_diff > self.max_score_difference:
                        print(f"⚠️ Incohérence détectée:")
                        print(f"   Similarité réponses {i+1}-{j+1}: {similarity_score:.2f}")
                        print(f"   Différence scores: {score_diff:.2f} > seuil {self.max_score_difference}")
                        print(f"   Scores: {scores[i]:.1f} vs {scores[j]:.1f}")
                        return False
        return True
    
    def parse_evaluation_for_check(self, llm_response: str) -> dict:
        """Parse rapide pour vérification cohérence"""
        try:
            import re
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, llm_response, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
                json_match = re.search(json_pattern, llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    return {}
            
            return json.loads(json_str)
        except Exception:
            return {}
    
    def evaluate(self, prompt: str) -> str:
        """
        Évalue avec vérification de cohérence et retry si nécessaire
        
        Args:
            prompt: Prompt avec les réponses à évaluer
            
        Returns:
            str: Réponse JSON du LLM
        """
        for attempt in range(self.max_retries + 1):
            try:
                # Appel LLM avec paramètres déterministes optimisés
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,  # 🆕 Déterminisme
                    top_p=1.0,      # 🆕 Pas de nucleus sampling
                    frequency_penalty=0.0,  # 🆕 Pas de pénalité
                    presence_penalty=0.0    # 🆕 Pas de pénalité
                )
                
                llm_response = response.choices[0].message.content.strip()
                
                # 🆕 VÉRIFICATION DE COHÉRENCE si activée
                if self.retry_on_inconsistency and attempt < self.max_retries:
                    try:
                        # Parse l'évaluation
                        evaluation = self.parse_evaluation_for_check(llm_response)
                        
                        # Extraction des scores
                        scores = [
                            evaluation.get("score_classic", 0),
                            evaluation.get("score_dense", 0),
                            evaluation.get("score_sparse", 0),
                            evaluation.get("score_dense_sc", 0)
                        ]
                        
                        # Extraction des réponses depuis le prompt
                        responses = self.extract_responses_from_prompt(prompt)
                        
                        # Vérification cohérence
                        if len(responses) == 4 and len(scores) == 4:
                            if self.check_response_consistency(responses, scores):
                                print(f"✅ Évaluation cohérente (tentative {attempt + 1})")
                                return llm_response
                            else:
                                print(f"🔄 Retry {attempt + 1}/{self.max_retries} - Incohérence détectée")
                                continue
                        else:
                            # Si on ne peut pas vérifier, on accepte
                            return llm_response
                            
                    except Exception as e:
                        print(f"⚠️ Erreur vérification cohérence: {e}")
                        return llm_response
                else:
                    # Dernière tentative ou vérification désactivée
                    return llm_response
                
            except Exception as e:
                if attempt == self.max_retries:
                    error_response = json.dumps({
                        "error": f"Erreur LLM juge après {self.max_retries + 1} tentatives: {str(e)}",
                        "score_classic": 2.5,
                        "score_dense": 2.5,
                        "score_sparse": 2.5,
                        "score_dense_sc": 2.5,
                        "best_approach": "Évaluation indisponible (erreur technique)",
                        "comparative_analysis": "Erreur lors de l'évaluation automatique"
                    })
                    return error_response
                
                print(f"❌ Tentative {attempt + 1} échouée: {e}")
                continue
        
        # Fallback (ne devrait jamais arriver)
        return '{"error": "Erreur inattendue dans l\'évaluateur"}'


# Fonction utilitaire (rétrocompatibilité)
def create_judge_client() -> LLMJudgeClient:
    """Crée un client juge amélioré"""
    return LLMJudgeClient()