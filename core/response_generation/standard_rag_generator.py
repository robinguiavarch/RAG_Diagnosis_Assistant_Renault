import os
import yaml
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAIGenerator:
    def __init__(self, model: str = "gpt-4o", context_token_limit: int = 6000, max_tokens: int = 2000):
        self.model = model
        self.context_token_limit = context_token_limit
        self.max_tokens = max_tokens
        
        # 🆕 CHARGEMENT max_context_chunks depuis settings.yaml
        try:
            with open("config/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            gen_cfg = settings.get("generation", {})
            self.max_context_chunks = gen_cfg.get("max_context_chunks", 3)
            print(f"📊 OpenAIGenerator (RAG Classique) config:")
            print(f"   🔢 Max chunks: {self.max_context_chunks}")
            print(f"   🎯 Token limit: {self.context_token_limit}")
        except Exception as e:
            print(f"⚠️ Erreur chargement settings.yaml: {e}, utilisation valeur par défaut")
            self.max_context_chunks = 3
        
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Charge le template de prompt depuis le fichier"""
        try:
            prompt_path = Path("config/prompts/standard_rag_prompt.txt")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback vers le prompt original si fichier absent
            return """Here are excerpts from technical documents:

{context}

Question: {query}

Answer clearly and precisely, strictly based on the provided context. 
If the answer is not in the context, explicitly state "Information not available in the provided context".
Answer in English only.
Answer:"""

    def _estimate_tokens(self, text: str) -> int:
        """Estimation simple du nombre de tokens"""
        return int(len(text.split()) * 0.75)

    def select_passages_with_limits(self, passages: List[str]) -> tuple:
        """
        🆕 SÉLECTION STRICTE avec respect de max_context_chunks
        
        Args:
            passages: Liste des passages à sélectionner
            
        Returns:
            tuple: (selected_passages, total_tokens)
        """
        # 🎯 LIMITE 1 : Nombre maximum de chunks (PRIORITAIRE)
        max_chunks = self.max_context_chunks
        selected_passages = passages[:max_chunks]
        
        # 🎯 LIMITE 2 : Vérification tokens (sécurité)
        total_tokens = 0
        final_passages = []
        
        for i, passage in enumerate(selected_passages):
            token_estimate = self._estimate_tokens(passage)
            
            # Vérification que ce passage ne dépasse pas la limite
            if total_tokens + token_estimate > self.context_token_limit:
                print(f"⚠️ Passage {i+1} ignoré: dépassement limite tokens ({total_tokens + token_estimate} > {self.context_token_limit})")
                break
                
            final_passages.append(passage)
            total_tokens += token_estimate

        print(f"✅ RAG Classique - Passages sélectionnés: {len(final_passages)}/{len(passages)} (tokens: {total_tokens})")
        return final_passages, total_tokens

    def generate_answer(self, query: str, passages: List[str]) -> str:
        """
        🆕 Génère une réponse avec respect strict de max_context_chunks
        
        Args:
            query: Question utilisateur
            passages: Liste des passages récupérés
            
        Returns:
            str: Réponse générée
        """
        if not passages:
            return "Information not available in the provided context."
        
        # 🆕 SÉLECTION AVEC LIMITES STRICTES
        selected_passages, total_tokens = self.select_passages_with_limits(passages)
        
        if not selected_passages:
            return "Information not available in the provided context."

        context = "\n\n".join(selected_passages)

        # Utilisation du template externalisé
        prompt = self.prompt_template.format(
            context=context,
            query=query
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=self.max_tokens,
                stop=["\n\n", "\nQuestion"]
            )
            
            generated_answer = response.choices[0].message.content.strip()
            
            # 🆕 LOGGING DE DIAGNOSTIC
            print(f"📝 Réponse RAG Classique générée: {len(generated_answer)} caractères")
            print(f"🎯 Contexte utilisé: {len(selected_passages)} passages, {total_tokens} tokens")
            
            return generated_answer

        except Exception as e:
            error_msg = f"❌ Erreur API OpenAI RAG Classique: {str(e)}"
            print(error_msg)
            return error_msg

    def get_generation_stats(self) -> dict:
        """Retourne les statistiques du générateur"""
        return {
            "model": self.model,
            "max_context_chunks": self.max_context_chunks,
            "context_token_limit": self.context_token_limit,
            "max_tokens": self.max_tokens,
            "generator_type": "rag_classique",
            "prompt_loaded": bool(self.prompt_template)
        }