"""
Client LLM simple pour le préprocessing des requêtes
Focus OpenAI uniquement - simplicité maximale
🔧 CORRECTION: Chargement settings.yaml AVANT initialisation par défaut
"""

import os
import yaml
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _load_llm_config() -> Dict[str, Any]:
    """
    🆕 Charge la configuration LLM depuis settings.yaml EN PREMIER
    Retourne les paramètres par défaut si fichier absent
    """
    default_config = {
        "model": "gpt-4o-mini",  # Fallback si settings.yaml absent
        "max_tokens": 1000,
        "temperature": 0.1
    }
    
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        
        llm_cfg = settings.get("query_processing", {}).get("llm", {})
        
        # Merge avec défauts (priorité aux settings.yaml)
        config = {
            "model": llm_cfg.get("model", default_config["model"]),
            "max_tokens": llm_cfg.get("max_tokens", default_config["max_tokens"]),
            "temperature": llm_cfg.get("temperature", default_config["temperature"])
        }
        
        print(f"🔧 Configuration LLM chargée depuis settings.yaml:")
        print(f"   📝 Modèle: {config['model']}")
        print(f"   🎛️ Temperature: {config['temperature']}")
        print(f"   📊 Max tokens: {config['max_tokens']}")
        
        return config
        
    except FileNotFoundError:
        print("⚠️ settings.yaml non trouvé, utilisation configuration par défaut")
        return default_config
    except Exception as e:
        print(f"⚠️ Erreur lecture settings.yaml: {e}, utilisation configuration par défaut")
        return default_config


class LLMClient:
    """Client simplifié pour OpenAI avec chargement prioritaire settings.yaml"""
    
    def __init__(self, model: Optional[str] = None, max_tokens: Optional[int] = None, 
                 temperature: Optional[float] = None):
        """
        🔧 CORRECTION: Charge settings.yaml AVANT d'appliquer les paramètres
        
        Args:
            model: Modèle à utiliser (override settings.yaml si fourni)
            max_tokens: Tokens max (override settings.yaml si fourni)
            temperature: Température (override settings.yaml si fourni)
        """
        # 🆕 CHARGEMENT PRIORITAIRE depuis settings.yaml
        config = _load_llm_config()
        
        # Application des paramètres (priorité: paramètres explicites > settings.yaml > défauts)
        self.model = model if model is not None else config["model"]
        self.max_tokens = max_tokens if max_tokens is not None else config["max_tokens"]
        self.temperature = temperature if temperature is not None else config["temperature"]
        
        print(f"✅ LLMClient initialisé:")
        print(f"   🤖 Modèle final: {self.model}")
        print(f"   🌡️ Température finale: {self.temperature}")
        print(f"   📏 Max tokens final: {self.max_tokens}")
    
    def generate(self, prompt: str) -> str:
        """
        Génère une réponse pour le prompt donné
        
        Args:
            prompt: Le prompt à envoyer au LLM
            
        Returns:
            str: Réponse du LLM
        """
        try:
            print(f"🧠 Appel LLM {self.model} avec {len(prompt)} caractères de prompt")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            llm_response = response.choices[0].message.content.strip()
            print(f"✅ Réponse LLM reçue: {len(llm_response)} caractères")
            
            return llm_response
            
        except Exception as e:
            print(f"❌ Erreur LLM: {e}")
            raise Exception(f"Erreur LLM: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle"""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


# Fonction utilitaire pour usage rapide
def create_llm_client(model: Optional[str] = None) -> LLMClient:
    """
    Crée un client LLM avec configuration prioritaire settings.yaml
    
    Args:
        model: Override le modèle depuis settings.yaml si fourni
    """
    return LLMClient(model=model)


if __name__ == "__main__":
    # Test simple avec affichage de la config
    print("🧪 Test LLMClient avec settings.yaml")
    
    client = create_llm_client()
    
    test_prompt = "Explain what ACAL-006 error means in one sentence."
    try:
        response = client.generate(test_prompt)
        print(f"✅ Test réussi: {response}")
    except Exception as e:
        print(f"❌ Test échoué: {e}")