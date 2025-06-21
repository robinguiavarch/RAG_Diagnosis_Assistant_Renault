"""
RAG + KG Dense S&C Generator - Version avec Multi-Query Fusion + max_context_chunks
Générateur spécialisé pour Knowledge Graph Dense S&C (Symptôme + Cause)
🆕 NOUVEAU: Support Multi-Query avec processed_query + Equipment Matching + Limite chunks stricte
"""

import os
import math
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import yaml
import sys

# Ajoute le dossier racine du projet à sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 🆕 IMPORT DES NOUVELLES FONCTIONS MULTI-QUERY DENSE S&C
from core.retrieval_graph.dense_sc_kg_querier import (
    get_structured_context_dense_sc_with_multi_query,
    get_structured_context_dense_sc_with_equipment_filter,
    get_structured_context_dense_sc
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize_cross_encoder_score(raw_score: float) -> float:
    """Normalise les scores CrossEncoder entre 0 et 1 via sigmoid"""
    try:
        normalized = 1.0 / (1.0 + math.exp(-raw_score))
        return float(normalized)
    except (OverflowError, ZeroDivisionError):
        if raw_score > 0:
            return 1.0
        else:
            return 0.0

def evaluate_document_relevance(reranked_docs: List[Dict[str, Any]], threshold: float = 0.7) -> Dict[str, Any]:
    """Évalue la pertinence des documents (identique aux autres générateurs)"""
    if not reranked_docs:
        return {
            "is_relevant": False,
            "relevant_count": 0,
            "total_count": 0,
            "max_score": 0.0,
            "avg_score": 0.0
        }
    
    normalized_scores = []
    relevant_count = 0
    
    for doc in reranked_docs:
        raw_score = doc.get('cross_encoder_score', 0.0)
        normalized_score = normalize_cross_encoder_score(raw_score)
        normalized_scores.append(normalized_score)
        doc['cross_encoder_score_normalized'] = normalized_score
        
        if normalized_score >= threshold:
            relevant_count += 1
    
    max_score = max(normalized_scores) if normalized_scores else 0.0
    avg_score = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0
    is_relevant = relevant_count >= 1
    
    stats = {
        "is_relevant": is_relevant,
        "relevant_count": relevant_count,
        "total_count": len(reranked_docs),
        "max_score": max_score,
        "avg_score": avg_score,
        "threshold_used": threshold
    }
    
    print(f"📊 Évaluation pertinence documents Dense S&C:")
    print(f"   🎯 Seuil utilisé: {threshold}")
    print(f"   ✅ Documents pertinents: {relevant_count}/{len(reranked_docs)}")
    print(f"   📈 Score max (normalisé): {max_score:.3f}")
    print(f"   🏆 Verdict: {'PERTINENT' if is_relevant else 'NON PERTINENT'}")
    
    return stats

class OpenAIGeneratorKGDenseSC:
    """Générateur RAG + Knowledge Graph Dense S&C avec Multi-Query + max_context_chunks"""
    
    def __init__(self, model: str = "gpt-4o", context_token_limit: int = 6000):
        self.model = model
        self.context_token_limit = context_token_limit

        # Chargement des paramètres YAML
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)

        gen_cfg = settings.get("generation", {})
        self.importance_context_rerank = gen_cfg.get("importance_context_rerank", 50)
        self.importance_context_graph = gen_cfg.get("importance_context_graph", 50)
        self.max_tokens = gen_cfg.get("max_new_tokens", 2000)
        
        # 🆕 CHARGEMENT max_context_chunks depuis settings.yaml
        self.max_context_chunks = gen_cfg.get("max_context_chunks", 3)
        
        self.max_triplets = gen_cfg.get("top_k_triplets", 3)
        self.seuil_pertinence = gen_cfg.get("seuil_pertinence", 0.7)
        
        # 🆕 CHARGEMENT DES PROMPTS POUR DENSE S&C
        self.prompts = self._load_prompt_templates()
        
        print(f"🎯 OpenAIGeneratorKGDenseSC initialisé:")
        print(f"   🔢 Max chunks: {self.max_context_chunks}")
        print(f"   🎯 Token limit: {self.context_token_limit}")
        print(f"   📊 Limitation triplets: {self.max_triplets}")
        print(f"   🎯 Seuil pertinence: {self.seuil_pertinence}")
        print(f"   📋 Prompts chargés: {list(self.prompts.keys())}")

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Charge les templates de prompts pour Dense S&C"""
        prompt_path = Path("config/prompts/rag_with_kg_dense_s&c_prompt.txt")
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Fichier prompt manquant: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse les prompts selon le format des fichiers fournis
        prompts = {}
        sections = content.split("# === PROMPT ")
        
        for section in sections[1:]:
            lines = section.split('\n')
            # Format: "KG_DENSE_SC_ONLY ===", "DOC_ONLY ===", "BOTH ==="
            prompt_header = lines[0].split(' ===')[0]
            prompt_content = '\n'.join(lines[1:]).strip()
            
            # Mapping vers les clés utilisées par le générateur
            if prompt_header == "KG_DENSE_SC_ONLY":
                prompts["kg_only"] = prompt_content
            elif prompt_header == "DOC_ONLY":
                prompts["doc_only"] = prompt_content
            elif prompt_header == "BOTH":
                prompts["both"] = prompt_content
        
        if len(prompts) != 3:
            raise ValueError(f"Prompts manquants dans {prompt_path}. Trouvés: {list(prompts.keys())}")
        
        return prompts

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

        print(f"✅ RAG Dense S&C - Passages sélectionnés: {len(final_passages)}/{len(passages)} (tokens: {total_tokens})")
        return final_passages, total_tokens

    def generate_answer(self, query: str, passages: List[str], 
                       reranked_metadata: Optional[List[Dict[str, Any]]] = None,
                       equipment_info: Optional[Dict] = None,
                       processed_query: Optional[Any] = None) -> str:  # 🆕 NOUVEAU PARAMÈTRE
        """
        🆕 Génère une réponse avec Multi-Query Dense S&C KG si processed_query fourni
        
        Args:
            query: Question de l'utilisateur (utilisée si pas de processed_query)
            passages: Textes des passages sélectionnés
            reranked_metadata: Métadonnées des documents re-rankés
            equipment_info: Informations equipment 
            processed_query: 🆕 Données complètes du LLM preprocessing
        """
        # 🆕 SÉLECTION AVEC LIMITES STRICTES
        selected_passages, total_tokens = self.select_passages_with_limits(passages)
        context_rerank = "\n\n".join(selected_passages)

        # Évaluation de la pertinence documentaire
        if reranked_metadata:
            doc_relevance_stats = evaluate_document_relevance(
                reranked_metadata, 
                threshold=self.seuil_pertinence
            )
            doc_has_content = doc_relevance_stats["is_relevant"]
        else:
            doc_has_content = len(selected_passages) > 0 and any(len(p.strip()) > 20 for p in selected_passages)
            doc_relevance_stats = {"is_relevant": doc_has_content, "max_score": 0.0}

        # 🆕 RÉCUPÉRATION DU CONTEXTE KG DENSE S&C AVEC MULTI-QUERY OU FALLBACK
        try:
            # 🆕 MULTI-QUERY si processed_query disponible
            if processed_query and hasattr(processed_query, 'query_variants'):
                print(f"🧠 Utilisation Multi-Query Dense S&C KG avec LLM preprocessing")
                context_graph = get_structured_context_dense_sc_with_multi_query(
                    processed_query.filtered_query,
                    processed_query.query_variants,
                    equipment_info or {},
                    format_type="compact", 
                    max_triplets=self.max_triplets
                )
            elif equipment_info:
                print(f"🏭 Utilisation Single-Query Dense S&C KG avec equipment matching")
                context_graph = get_structured_context_dense_sc_with_equipment_filter(
                    query,
                    equipment_info,
                    format_type="compact", 
                    max_triplets=self.max_triplets
                )
            else:
                print(f"📄 Utilisation Single-Query Dense S&C KG classique")
                context_graph = get_structured_context_dense_sc(
                    query, 
                    format_type="compact", 
                    max_triplets=self.max_triplets
                )
            
            if not context_graph or context_graph.startswith("No relevant"):
                context_graph = "[No relevant information found in Dense S&C Knowledge Graph]"
                kg_has_content = False
                triplet_count = 0
            else:
                triplet_count = len([line for line in context_graph.split('\n') if '→' in line])
                print(f"✅ Contexte Dense S&C KG récupéré : {triplet_count} triplets")
                kg_has_content = triplet_count > 0
                
        except Exception as e:
            context_graph = f"[⚠️ Unable to retrieve Dense S&C KG context: {str(e)}]"
            print(f"❌ Erreur lors de la récupération du contexte Dense S&C KG : {e}")
            kg_has_content = False
            triplet_count = 0

        # Stratégie de réponse (identique aux autres générateurs)
        if not doc_has_content and not kg_has_content:
            print("🚫 Stratégie: AUCUN_CONTEXTE")
            return "Information not available in the provided context."
        
        elif not doc_has_content and kg_has_content:
            prompt_type = "kg_only"
            mode_str = "Multi-Query" if processed_query and hasattr(processed_query, 'query_variants') else "Single-Query"
            print(f"🧠 Stratégie: KG_DENSE_SC_SEULEMENT ({mode_str}) - {triplet_count} triplets enrichis")
        
        elif doc_has_content and not kg_has_content:
            prompt_type = "doc_only"
            print(f"📄 Stratégie: DOC_SEULEMENT")
        
        else:
            prompt_type = "both"
            mode_str = "Multi-Query" if processed_query and hasattr(processed_query, 'query_variants') else "Single-Query"
            print(f"🔄 Stratégie: HYBRIDE ({mode_str}) - docs + {triplet_count} triplets Dense S&C")

        # Génération du prompt
        prompt = self._generate_adaptive_prompt(
            prompt_type, query, context_rerank, context_graph, doc_relevance_stats
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
            
            # 🆕 LOGGING DE DIAGNOSTIC COMPLET AVEC MODE
            multi_query_info = ""
            if processed_query and hasattr(processed_query, 'query_variants'):
                multi_query_info = f" [Multi-Query: {len(processed_query.query_variants)} variantes]"
            
            print(f"📝 Réponse Dense S&C générée: {len(generated_answer)} caractères")
            print(f"🎯 Stratégie utilisée: {prompt_type.upper()}{multi_query_info}")
            print(f"📄 Contexte doc: {'✅' if doc_has_content else '❌'} (score max: {doc_relevance_stats.get('max_score', 0):.3f})")
            print(f"🧠 Contexte Dense S&C KG: {'✅' if kg_has_content else '❌'} ({triplet_count} triplets)")
            print(f"🎯 Passages utilisés: {len(selected_passages)}, tokens: {total_tokens}")
            
            return generated_answer

        except Exception as e:
            error_msg = f"❌ OpenAI API error with Dense S&C KG context: {str(e)}"
            print(error_msg)
            return error_msg

    def _generate_adaptive_prompt(self, prompt_type: str, query: str, 
                                 context_rerank: str, context_graph: str,
                                 doc_stats: Dict[str, Any]) -> str:
        """Génère le prompt adaptatif pour Dense S&C"""
        
        if prompt_type not in self.prompts:
            raise ValueError(f"Type de prompt inconnu: {prompt_type}. Disponibles: {list(self.prompts.keys())}")
        
        template = self.prompts[prompt_type]
        
        if prompt_type == "kg_only":
            return template.format(
                context_graph=context_graph,
                query=query
            )
        elif prompt_type == "doc_only":
            return template.format(
                context_rerank=context_rerank,
                query=query
            )
        else:  # both
            return template.format(
                importance_context_rerank=self.importance_context_rerank,
                max_score=doc_stats.get('max_score', 0),
                context_rerank=context_rerank,
                importance_context_graph=self.importance_context_graph,
                max_triplets=self.max_triplets,
                context_graph=context_graph,
                query=query
            )

    def get_generation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du générateur Dense S&C"""
        return {
            "model": self.model,
            "max_context_chunks": self.max_context_chunks,
            "context_token_limit": self.context_token_limit,
            "max_tokens": self.max_tokens,
            "max_triplets": self.max_triplets,
            "seuil_pertinence": self.seuil_pertinence,
            "kg_type": "dense_s&c",
            "structure": "symptom+cause enriched with semantic propagation",
            "prompts_loaded": len(self.prompts),
            "multi_query_support": True  # 🆕 Indicateur Multi-Query
        }