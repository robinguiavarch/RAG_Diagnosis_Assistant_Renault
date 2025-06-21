"""
Dense S&C KG Querier - Version COMPLÈTE avec Multi-Query Fusion + Equipment Matching CORRIGÉE
Recherche dans le Knowledge Graph Dense S&C (Symptôme + Cause)
Même logique que Dense standard mais avec texte enrichi
🆕 NOUVEAU: Multi-Query Fusion (filtered_query + variants) avec stratégie MAX Score
Pour tester:
docker run --rm \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/core:/app/core \
  --network host \
  -e PYTHONPATH=/app \
  diagnosis-app \
  poetry run python core/retrieval_graph/dense_sc_kg_querier.py "motor overheating FANUC R-30iB error ACAL-006"
"""

import os
import pickle
import faiss
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import yaml
from typing import List, Dict, Optional
import sys

load_dotenv()

# === CONFIGURATION SIMPLE ===
def load_settings():
    """Charge la configuration depuis settings.yaml"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", "settings.yaml")
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config = load_settings()
model_name = config["models"]["embedding_model"]
threshold = config["graph_retrieval"]["symptom_similarity_threshold"]
symptom_top_k = config["graph_retrieval"]["symptom_top_k"]
triplets_limit = config["generation"]["top_k_triplets"]

# Chemins pour Dense S&C
script_dir = os.path.dirname(os.path.abspath(__file__))
embedding_dir = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "symptom_embedding_dense_s&c")

# 🆕 IMPORT DU MATCHER HYBRIDE (même que Dense)
try:
    from core.retrieval_graph.hybrid_symptom_matcher import create_hybrid_symptom_matcher
    HYBRID_MATCHER_AVAILABLE = True
except ImportError:
    print("⚠️ Hybrid matcher non disponible, fallback vers FAISS")
    HYBRID_MATCHER_AVAILABLE = False

# === FONCTIONS ESSENTIELLES ===

def get_model():
    """Charge le modèle d'embedding"""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(model_name, device=device)
    except Exception as e:
        print(f"❌ Erreur modèle : {e}")
        raise

def get_dense_sc_driver():
    """🔧 CORRECTION: Connexion Dense S&C DIRECTE avec logique cloud/local"""
    load_dotenv()
    
    # Priorité absolue au Cloud si activé
    cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
    
    if cloud_enabled:
        print("🌐 MODE CLOUD DENSE S&C (connexion directe)")
        uri = os.getenv("NEO4J_DENSE_SC_CLOUD_URI")
        password = os.getenv("NEO4J_DENSE_SC_CLOUD_PASS")
        
        if uri and password:
            print(f"🔌 Connexion Cloud: {uri}")
            try:
                driver = GraphDatabase.driver(uri, auth=("neo4j", password))
                # Test rapide de connexion
                with driver.session() as session:
                    session.run("RETURN 1")
                print("✅ Connexion Cloud Dense S&C réussie")
                return driver
            except Exception as e:
                print(f"❌ Échec connexion Cloud Dense S&C: {e}")
                print("🔄 Fallback vers local...")
        else:
            print("❌ Credentials cloud manquants")
            print("🔄 Fallback vers local...")
    
    # Fallback Local
    print("🏠 MODE LOCAL DENSE S&C")
    db_uri = os.getenv("NEO4J_URI_DENSE_SC", "bolt://host.docker.internal:7690")
    db_user = os.getenv("NEO4J_USER_DENSE_SC", "neo4j")
    db_pass = os.getenv("NEO4J_PASS_DENSE_SC", "password")
    print(f"🔌 Connexion Local: {db_uri}")
    return GraphDatabase.driver(db_uri, auth=(db_user, db_pass))

def load_symptom_index_dense_sc():
    """Charge l'index FAISS des symptômes Dense S&C"""
    index_path = os.path.join(embedding_dir, "index.faiss")
    metadata_path = os.path.join(embedding_dir, "symptom_embedding_dense_s&c.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ Index Dense S&C manquant dans {embedding_dir}")
    
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return index, metadata

def get_similar_symptoms_dense_sc(query: str) -> List[tuple]:
    """
    🆕 Trouve les symptômes similaires via recherche HYBRIDE ou FAISS dans Dense S&C
    Recherche sur les textes combinés symptôme + cause
    """
    # Pour l'instant, utilisation FAISS directe (hybride sera ajouté plus tard)
    print("🔍 Utilisation de la recherche FAISS Dense S&C (texte combiné)")
    try:
        model = get_model()
        index, metadata = load_symptom_index_dense_sc()
        
        # 🆕 Métadonnées enrichies S&C
        symptom_names = metadata['symptom_names']
        combined_texts = metadata['combined_texts']
        
        # Recherche vectorielle sur texte combiné
        query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(query_vec, symptom_top_k * 2)
        
        # Filtrage par seuil
        results = []
        for i, score in zip(indices[0], scores[0]):
            if score >= threshold and i < len(symptom_names):
                results.append((symptom_names[i], float(score)))
                if len(results) >= symptom_top_k:
                    break
        
        print(f"🔍 Dense S&C: {len(results)} symptômes trouvés")
        return results
        
    except Exception as e:
        print(f"❌ Erreur recherche symptômes Dense S&C: {e}")
        return []

def query_neo4j_triplets_dense_sc(symptom: str) -> List[Dict]:
    """Récupère les triplets pour un symptôme dans Dense S&C"""
    driver = get_dense_sc_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Symptom {name: $symptom})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, 
                       s.equipment AS equipment, s.combined_text AS combined_text
            """, symptom=symptom)
            
            triplets = [record.data() for record in result]
            return triplets
    except Exception as e:
        print(f"❌ Erreur Neo4j Dense S&C: {e}")
        return []
    finally:
        driver.close()

# === 🆕 FONCTION MULTI-QUERY FUSION DENSE S&C ===

def get_symptoms_with_variants_dense_sc(filtered_query: str, query_variants: List[str]) -> List[tuple]:
    """
    🆕 MULTI-QUERY FUSION DENSE S&C - Recherche symptoms avec query filtrée + variantes
    Stratégie MAX Score adaptée pour texte combiné (symptôme + cause)
    
    Args:
        filtered_query: Query optimisée par LLM
        query_variants: Liste des variantes générées par LLM
        
    Returns:
        List[tuple]: [(symptom_name, max_score), ...] top symptoms
    """
    print(f"🔍 Multi-Query Fusion Dense S&C KG:")
    print(f"🎯 Query filtrée: '{filtered_query}'")
    print(f"🔄 Variantes: {query_variants}")
    
    # Poids par source (query filtrée prioritaire)
    weights = {
        "filtered": 1.0,      # Query LLM optimisée = poids max
        "variant": 0.8        # Variantes = poids réduit
    }
    
    symptom_scores = {}
    
    # 1. Recherche avec query filtrée (poids principal)
    print(f"🎯 Recherche avec query filtrée...")
    filtered_symptoms = get_similar_symptoms_dense_sc(filtered_query)
    for symptom, score in filtered_symptoms:
        weighted_score = score * weights["filtered"]
        symptom_scores[symptom] = weighted_score
        print(f"   ✅ Filtered: {symptom} → {weighted_score:.3f}")
    
    # 2. Recherche avec variantes (poids réduit)
    for i, variant in enumerate(query_variants[:2]):  # Max 2 variantes pour performance
        if not variant or variant == filtered_query:  # Skip si vide ou identique
            continue
            
        print(f"🔄 Recherche avec variante {i+1}: '{variant}'")
        variant_symptoms = get_similar_symptoms_dense_sc(variant)
        
        for symptom, score in variant_symptoms:
            weighted_score = score * weights["variant"]
            
            # STRATÉGIE MAX Score - garde le meilleur score pour ce symptom
            if symptom in symptom_scores:
                old_score = symptom_scores[symptom]
                new_score = max(old_score, weighted_score)
                symptom_scores[symptom] = new_score
                print(f"   🔄 Variant{i+1}: {symptom} → MAX({old_score:.3f}, {weighted_score:.3f}) = {new_score:.3f}")
            else:
                symptom_scores[symptom] = weighted_score
                print(f"   🆕 Variant{i+1}: {symptom} → {weighted_score:.3f}")
    
    # 3. Tri et limitation par score final
    sorted_symptoms = sorted(symptom_scores.items(), key=lambda x: x[1], reverse=True)
    final_symptoms = sorted_symptoms[:symptom_top_k]
    
    print(f"✅ Multi-Query Dense S&C: {len(final_symptoms)} symptoms sélectionnés (top scores MAX)")
    for i, (symptom, score) in enumerate(final_symptoms, 1):
        print(f"   {i}. {symptom} → {score:.3f}")
    
    return final_symptoms

# === 🆕 FONCTIONS EQUIPMENT MATCHING (CONSERVÉES) ===

def _extract_kg_equipments_dense_sc() -> List[str]:
    """Extrait tous les equipments uniques du KG Dense S&C"""
    try:
        driver = get_dense_sc_driver()
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.equipment IS NOT NULL
                RETURN DISTINCT n.equipment AS equipment
                ORDER BY n.equipment
            """)
            
            equipments = [record["equipment"] for record in result if record["equipment"]]
            print(f"📊 {len(equipments)} equipments trouvés dans KG Dense S&C")
            return equipments
            
    except Exception as e:
        print(f"⚠️ Erreur extraction equipments Dense S&C: {e}")
        return []
    finally:
        if 'driver' in locals():
            driver.close()

def _query_neo4j_triplets_dense_sc_with_equipment_filter(symptom: str, matched_equipment: Optional[str]) -> List[Dict]:
    """Récupère les triplets Dense S&C pour un symptôme avec filtrage equipment optionnel"""
    driver = get_dense_sc_driver()
    try:
        with driver.session() as session:
            if matched_equipment:
                # Requête filtrée par equipment
                result = session.run("""
                    MATCH (s:Symptom {name: $symptom})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                    WHERE s.equipment = $equipment
                    RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, 
                           s.equipment AS equipment, s.combined_text AS combined_text
                """, symptom=symptom, equipment=matched_equipment)
            else:
                # Requête globale (comportement actuel)
                result = session.run("""
                    MATCH (s:Symptom {name: $symptom})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                    RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, 
                           s.equipment AS equipment, s.combined_text AS combined_text
                """, symptom=symptom)
            
            triplets = [record.data() for record in result]
            return triplets
    except Exception as e:
        print(f"❌ Erreur Neo4j Dense S&C avec equipment: {e}")
        return []
    finally:
        driver.close()

# === 🆕 FONCTION PRINCIPALE MULTI-QUERY + EQUIPMENT ===

def get_structured_context_with_variants_and_equipment_dense_sc(
    filtered_query: str,
    query_variants: List[str],
    equipment_info: Dict,
    format_type: str = "detailed", 
    max_triplets: Optional[int] = None
) -> str:
    """
    🆕 FONCTION PRINCIPALE - Multi-Query Fusion + Equipment Matching pour Dense S&C KG
    
    Args:
        filtered_query: Query optimisée par LLM  
        query_variants: Variantes générées par LLM
        equipment_info: Infos equipment pour matching
        format_type: Format de sortie ("detailed", "compact", "json")
        max_triplets: Limite triplets finaux
        
    Returns:
        str: Contexte KG Dense S&C formaté avec Multi-Query (texte combiné)
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"🔍 Dense S&C KG avec Multi-Query Fusion + Equipment Matching")
        
        # === EQUIPMENT MATCHING (logique existante conservée) ===
        matched_equipment = None
        if equipment_info and equipment_info.get('primary_equipment') != 'unknown':
            try:
                from core.retrieval_graph.equipment_matcher import create_equipment_matcher
                matcher = create_equipment_matcher()
                
                # Extraction des equipments disponibles dans le KG Dense S&C
                kg_equipments = _extract_kg_equipments_dense_sc()
                
                if kg_equipments:
                    # Matching LLM equipment → KG equipment
                    matched_equipment = matcher.match_equipment(
                        equipment_info['primary_equipment'], 
                        kg_equipments
                    )
                    
                    if matched_equipment:
                        print(f"🏭 Equipment match trouvé: '{matched_equipment}' (score > 0.9)")
                    else:
                        print(f"🔍 Pas de match equipment (< 0.9), recherche globale")
                else:
                    print("⚠️ Aucun equipment trouvé dans le KG Dense S&C")
                    
            except Exception as e:
                print(f"⚠️ Equipment matching échoué: {e}, fallback global")
                matched_equipment = None
        
        # === 🆕 RECHERCHE MULTI-QUERY DENSE S&C ===
        similar_symptoms = get_symptoms_with_variants_dense_sc(filtered_query, query_variants)
        
        if not similar_symptoms:
            return "No relevant structured information found with multi-query approach."
        
        # === RÉCUPÉRATION TRIPLETS (avec equipment filter et propagation sémantique) ===
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score in similar_symptoms:
            triplets = _query_neo4j_triplets_dense_sc_with_equipment_filter(symptom_name, matched_equipment)
            
            for triplet in triplets:
                triplet_key = (triplet['symptom'], triplet['cause'], triplet['remedy'])
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # === LIMITATION ET TRI ===
        if len(all_triplets) > max_triplets:
            all_triplets.sort(key=lambda x: x['similarity_score'], reverse=True)
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        equipment_info_str = f" (equipment: {matched_equipment})" if matched_equipment else " (global search)"
        print(f"✅ {len(selected)} triplets Dense S&C sélectionnés avec Multi-Query{equipment_info_str}")
        
        # === FORMATAGE ===
        if format_type == "json":
            import json
            return json.dumps(selected, indent=2, ensure_ascii=False)
        
        elif format_type == "compact":
            lines = [f"{t['symptom']} → {t['cause']} → {t['remedy']}" for t in selected]
            return "\n".join(lines)
        
        else:  # detailed
            lines = []
            for i, t in enumerate(selected, 1):
                lines.append(
                    f"Triplet {i} (multi-query score: {t['similarity_score']:.3f}):\n"
                    f"  Symptom: {t['symptom']}\n"
                    f"  Cause: {t['cause']}\n"
                    f"  Remedy: {t['remedy']}\n"
                    f"  Equipment: {t.get('equipment', 'N/A')}\n"
                    f"  Combined: {t.get('combined_text', 'N/A')}\n"
                )
            
            if lines:
                header = f"=== MULTI-QUERY DENSE S&C KNOWLEDGE GRAPH CONTEXT ===\n"
                header += f"Filtered Query: '{filtered_query}'\n"
                header += f"Variants: {query_variants}\n"
                header += f"Equipment filter: {matched_equipment or 'None (global search)'}\n"
                header += f"Method: Multi-Query Fusion (MAX Score) + Equipment Matching + Symptom+Cause combined\n"
                header += f"Triplets: {len(selected)}/{len(all_triplets)}\n\n"
                return header + "\n".join(lines)
            else:
                return "No structured context available with multi-query approach."
        
    except Exception as e:
        print(f"❌ Erreur Multi-Query Dense S&C KG: {e}")
        # Fallback vers fonction single-query
        print("🔄 Fallback vers recherche single-query...")
        return get_structured_context_dense_sc_with_equipment(filtered_query, equipment_info, format_type, max_triplets)

# === FONCTIONS SINGLE-QUERY (CONSERVÉES POUR RÉTROCOMPATIBILITÉ) ===

def get_structured_context_dense_sc_with_equipment(query: str, equipment_info: Dict, 
                                                  format_type: str = "detailed", 
                                                  max_triplets: Optional[int] = None) -> str:
    """
    🎯 FONCTION SINGLE-QUERY avec Equipment Matching (logique existante conservée)
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"🔍 Dense S&C KG avec Single-Query + Equipment Matching")
        
        # === EQUIPMENT MATCHING ===
        matched_equipment = None
        if equipment_info and equipment_info.get('primary_equipment') != 'unknown':
            try:
                from core.retrieval_graph.equipment_matcher import create_equipment_matcher
                matcher = create_equipment_matcher()
                
                kg_equipments = _extract_kg_equipments_dense_sc()
                
                if kg_equipments:
                    matched_equipment = matcher.match_equipment(
                        equipment_info['primary_equipment'], 
                        kg_equipments
                    )
                    
                    if matched_equipment:
                        print(f"🏭 Equipment match trouvé: '{matched_equipment}' (score > 0.9)")
                    else:
                        print(f"🔍 Pas de match equipment (< 0.9), recherche globale")
                        
            except Exception as e:
                print(f"⚠️ Equipment matching échoué: {e}, fallback global")
                matched_equipment = None
        
        # === RECHERCHE SINGLE-QUERY ===
        similar_symptoms = get_similar_symptoms_dense_sc(query)
        if not similar_symptoms:
            return "No relevant structured information found in Dense S&C Knowledge Base."
        
        # === RÉCUPÉRATION TRIPLETS ===
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score in similar_symptoms:
            triplets = _query_neo4j_triplets_dense_sc_with_equipment_filter(symptom_name, matched_equipment)
            
            for triplet in triplets:
                triplet_key = (triplet['symptom'], triplet['cause'], triplet['remedy'])
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # === LIMITATION ET TRI ===
        if len(all_triplets) > max_triplets:
            all_triplets.sort(key=lambda x: x['similarity_score'], reverse=True)
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        equipment_info_str = f" (equipment: {matched_equipment})" if matched_equipment else " (global search)"
        print(f"✅ {len(selected)} triplets Dense S&C sélectionnés{equipment_info_str}")
        
        # === FORMATAGE ===
        if format_type == "json":
            import json
            return json.dumps(selected, indent=2, ensure_ascii=False)
        
        elif format_type == "compact":
            lines = [f"{t['symptom']} → {t['cause']} → {t['remedy']}" for t in selected]
            return "\n".join(lines)
        
        else:  # detailed
            lines = []
            for i, t in enumerate(selected, 1):
                lines.append(
                    f"Triplet {i} (score: {t['similarity_score']:.3f}):\n"
                    f"  Symptom: {t['symptom']}\n"
                    f"  Cause: {t['cause']}\n"
                    f"  Remedy: {t['remedy']}\n"
                    f"  Equipment: {t.get('equipment', 'N/A')}\n"
                    f"  Combined: {t.get('combined_text', 'N/A')}\n"
                )
            
            if lines:
                header = f"=== DENSE S&C KNOWLEDGE GRAPH CONTEXT ===\n"
                header += f"Query: '{query}'\n"
                header += f"Equipment filter: {matched_equipment or 'None (global search)'}\n"
                header += f"Method: Single-Query + Equipment Matching + Symptom+Cause combined\n"
                header += f"Triplets: {len(selected)}/{len(all_triplets)}\n\n"
                return header + "\n".join(lines)
            else:
                return "No structured context available in Dense S&C KB."
    
    except Exception as e:
        print(f"❌ Erreur Dense S&C KG avec equipment: {e}")
        # Fallback vers fonction originale
        print("🔄 Fallback vers recherche Dense S&C globale...")
        return get_structured_context_dense_sc_original(query, format_type, max_triplets)

def get_structured_context_dense_sc_original(query: str, format_type: str = "detailed", 
                                            max_triplets: Optional[int] = None) -> str:
    """
    🎯 Fonction originale INCHANGÉE pour rétrocompatibilité
    Recherche enrichie par symptôme + cause combinés
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"🔍 Recherche dans KG DENSE S&C (symptôme + cause enrichi)")
        
        # 1. Recherche symptômes similaires (texte combiné)
        similar_symptoms = get_similar_symptoms_dense_sc(query)
        if not similar_symptoms:
            return "No relevant structured information found in Dense S&C Knowledge Base."
        
        # 2. Récupération triplets (avec propagation sémantique)
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score in similar_symptoms:
            triplets = query_neo4j_triplets_dense_sc(symptom_name)
            
            for triplet in triplets:
                triplet_key = (triplet['symptom'], triplet['cause'], triplet['remedy'])
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # 3. Limitation et tri
        if len(all_triplets) > max_triplets:
            all_triplets.sort(key=lambda x: x['similarity_score'], reverse=True)
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        print(f"✅ {len(selected)} triplets Dense S&C sélectionnés")
        
        # 4. Formatage
        if format_type == "json":
            import json
            return json.dumps(selected, indent=2, ensure_ascii=False)
        
        elif format_type == "compact":
            lines = [f"{t['symptom']} → {t['cause']} → {t['remedy']}" for t in selected]
            return "\n".join(lines)
        
        else:  # detailed
            lines = []
            for i, t in enumerate(selected, 1):
                lines.append(
                    f"Triplet {i} (score: {t['similarity_score']:.3f}):\n"
                    f"  Symptom: {t['symptom']}\n"
                    f"  Cause: {t['cause']}\n"
                    f"  Remedy: {t['remedy']}\n"
                    f"  Equipment: {t.get('equipment', 'N/A')}\n"
                    f"  Combined: {t.get('combined_text', 'N/A')}\n"
                )
            
            if lines:
                header = f"=== DENSE S&C KNOWLEDGE GRAPH CONTEXT ===\n"
                header += f"Query: '{query}'\n"
                header += f"Method: Symptom + Cause combined search\n"
                header += f"Triplets: {len(selected)}/{len(all_triplets)}\n\n"
                return header + "\n".join(lines)
            else:
                return "No structured context available in Dense S&C KB."
    
    except Exception as e:
        print(f"❌ Erreur Dense S&C KG: {e}")
        return f"Error retrieving Dense S&C context: {str(e)}"

# === 🎯 INTERFACES PUBLIQUES (3 NIVEAUX) ===

def get_structured_context_dense_sc_with_multi_query(
    filtered_query: str,
    query_variants: List[str],
    equipment_info: Dict,
    format_type: str = "detailed",
    max_triplets: Optional[int] = None
) -> str:
    """
    🎯 INTERFACE MULTI-QUERY - Nouvelle interface principale
    Utilisée par les générateurs RAG quand processed_query disponible
    """
    return get_structured_context_with_variants_and_equipment_dense_sc(
        filtered_query, query_variants, equipment_info, format_type, max_triplets
    )

def get_structured_context_dense_sc_with_equipment_filter(query: str, equipment_info: Dict, 
                                                         format_type: str = "detailed", 
                                                         max_triplets: Optional[int] = None) -> str:
    """
    🎯 INTERFACE SINGLE-QUERY + EQUIPMENT - Rétrocompatibilité
    Utilisée par les générateurs RAG en mode single-query avec equipment
    """
    return get_structured_context_dense_sc_with_equipment(query, equipment_info, format_type, max_triplets)

def get_structured_context_dense_sc(query: str, format_type: str = "detailed", 
                                   max_triplets: Optional[int] = None) -> str:
    """
    🎯 INTERFACE ORIGINALE - Rétrocompatibilité totale
    Utilisée par les anciens appels et mode classique
    """
    return get_structured_context_dense_sc_original(query, format_type, max_triplets)

# === TEST CLI SIMPLE ===
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Test Dense S&C KG Multi-Query: {query}")
        print("-" * 50)
        
        # Test Multi-Query
        filtered_query = f"ACAL-006 TPE operation error FANUC R-30iB"
        variants = ["ACAL-006 teach pendant error FANUC", "TPE operation failure ACAL-006"]
        equipment_info = {"primary_equipment": "FANUC R-30iB"}
        
        result = get_structured_context_dense_sc_with_multi_query(
            filtered_query, variants, equipment_info
        )
        print(result)
    else:
        print("Usage: python dense_sc_kg_querier.py 'votre requête'")
        print("Exemple: python dense_sc_kg_querier.py 'motor overheating FANUC'")