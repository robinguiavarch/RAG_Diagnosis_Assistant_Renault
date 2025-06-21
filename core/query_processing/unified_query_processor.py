"""
Processeur unifié simple pour les requêtes utilisateur
🔧 VERSION DEBUG avec logs détaillés pour identifier le problème
"""

import os
import yaml
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient
from .response_parser import ResponseParser, ProcessedQuery


class UnifiedQueryProcessor:
    """Processeur unifié simple pour traiter les requêtes utilisateur"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        print("🔧 Initialisation UnifiedQueryProcessor...")
        
        # Client LLM
        self.llm_client = llm_client or LLMClient()
        print(f"✅ LLM Client créé: {type(self.llm_client)}")
        
        # Parser
        self.parser = ResponseParser()
        print(f"✅ Parser créé: {type(self.parser)}")
        
        # Chargement du prompt
        self.prompt_template = self._load_prompt_template()
        print(f"✅ Prompt template chargé: {len(self.prompt_template)} caractères")
    
    def _load_prompt_template(self) -> str:
        """Charge le template de prompt"""
        script_dir = Path(__file__).parent.parent.parent
        prompt_path = script_dir / "config" / "prompts" / "unified_query_prompt.txt"
        
        print(f"🔍 Tentative chargement prompt: {prompt_path}")
        print(f"🔍 Chemin absolu: {prompt_path.absolute()}")
        print(f"🔍 Fichier existe: {prompt_path.exists()}")
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"✅ Prompt chargé avec succès: {len(content)} caractères")
                # Afficher les 100 premiers caractères pour vérifier le contenu
                print(f"🔍 Début du prompt: {content[:100]}...")
                return content
        except FileNotFoundError as e:
            print(f"❌ Fichier prompt non trouvé: {e}")
            print("🔄 Utilisation du prompt fallback")
            # Prompt minimal par défaut
            return """
You are an expert in industrial equipment troubleshooting. Analyze this query and extract technical information.

USER QUERY: "{raw_query}"

Respond with JSON containing technical_terms, equipment_info, filtered_query, and query_variants.
            """.strip()
        except Exception as e:
            print(f"❌ Erreur lecture prompt: {e}")
            return "Prompt fallback minimal"
    
    def process_user_query(self, raw_query: str) -> ProcessedQuery:
        """
        Traite une requête utilisateur avec le LLM
        🔧 CORRECTION: Utilisation de replace() au lieu de format()
        """
        print(f"\n🚀 DÉBUT process_user_query")
        print(f"📝 Query reçue: '{raw_query[:50]}...'")
        
        if not raw_query or not raw_query.strip():
            print("❌ Query vide détectée")
            raise ValueError("La requête ne peut pas être vide")
        
        raw_query = raw_query.strip()
        print(f"📝 Query nettoyée: {len(raw_query)} caractères")
        
        try:
            print("🔧 ÉTAPE 1: Construction du prompt")
            # 🔧 CORRECTION: replace() au lieu de format() pour éviter les conflits JSON
            prompt = self.prompt_template.replace("{raw_query}", raw_query)
            print(f"✅ Prompt construit: {len(prompt)} caractères")
            print(f"🔍 Début du prompt final: {prompt[:200]}...")
            
            print("🔧 ÉTAPE 2: Appel LLM")
            # Appel LLM
            llm_response = self.llm_client.generate(prompt)
            print(f"✅ Réponse LLM reçue: {len(llm_response)} caractères")
            print(f"🔍 Début de la réponse: {llm_response[:200]}...")
            
            print("🔧 ÉTAPE 3: Parsing de la réponse")
            # Parsing de la réponse
            processed_query = self.parser.parse_llm_response(llm_response, raw_query)
            print(f"✅ Parsing réussi")
            print(f"🔍 Query filtrée résultante: '{processed_query.filtered_query}'")
            print(f"🔍 Nombre de variantes: {len(processed_query.query_variants)}")
            print(f"🔍 Equipment détecté: {processed_query.equipment_info.primary_equipment}")
            
            return processed_query
            
        except Exception as e:
            print(f"❌ ERREUR dans process_user_query: {e}")
            print(f"❌ Type d'erreur: {type(e)}")
            print("🔄 Activation du fallback")
            
            # Fallback simple : retourner la requête originale
            return self._create_fallback_result(raw_query, str(e))
    
    def _create_fallback_result(self, query: str, error_msg: str) -> ProcessedQuery:
        """Crée un résultat de fallback en cas d'erreur"""
        print(f"🚨 FALLBACK activé: {error_msg}")
        
        from .response_parser import TechnicalTerms, EquipmentInfo
        
        fallback_result = ProcessedQuery(
            raw_query=query,
            technical_terms=TechnicalTerms([], [], [], []),
            equipment_info=EquipmentInfo("unknown", "unknown", "unknown"),
            filtered_query=query,  # ← ICI LE PROBLÈME ! Query identique !
            query_variants=[],
            confidence_score=0.0
        )
        
        print(f"🔄 Fallback result créé avec query identique")
        return fallback_result
    
    def get_config(self):
        """Retourne la configuration actuelle"""
        return {
            "llm_config": self.llm_client.get_config(),
            "prompt_loaded": bool(self.prompt_template)
        }


# Fonction utilitaire pour usage rapide
def create_query_processor() -> UnifiedQueryProcessor:
    """Crée un processeur avec configuration par défaut"""
    print("🏭 Création du query processor...")
    return UnifiedQueryProcessor()


def process_single_query(query: str) -> ProcessedQuery:
    """Fonction utilitaire pour traiter une seule requête rapidement"""
    processor = create_query_processor()
    return processor.process_user_query(query)


if __name__ == "__main__":
    # Test simple
    test_queries = [
        "I got the error ACAL-006 TPE operation error on the FANUC-30iB machine teach pendant. I don't understand why.",
        "motor overheating on robot arm",
        "SYST-001 brake failure"
    ]
    
    try:
        processor = create_query_processor()
        
        print("🧪 Test du UnifiedQueryProcessor")
        
        # Test d'une requête
        result = processor.process_user_query(test_queries[0])
        
        print(f"✅ Résultat:")
        print(f"   Requête filtrée: {result.filtered_query}")
        print(f"   Codes d'erreur: {result.technical_terms.error_codes}")
        print(f"   Équipement: {result.equipment_info.primary_equipment}")
        print(f"   Variantes: {len(result.query_variants)}")
        
    except Exception as e:
        print(f"❌ Test échoué: {e}")