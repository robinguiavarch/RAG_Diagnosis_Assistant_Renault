"""
Sparse KG Querier - Version COMPLÈTE avec Multi-Query Fusion + Equipment Matching CORRIGÉE
Recherche dans le Knowledge Graph Sparse (SANS propagation sémantique)
Structure: 1 Symptôme → 1 Cause → 1 Remède (1:1:1)
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
  poetry run python core/retrieval_graph/sparse_kg_querier.py "motor overheating FANUC R-30iB error ACAL-006"
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

# Chemins pour Sparse
script_dir = os.path.dirname(os.path.abspath(__file__))
embedding_dir = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "symptom_embedding_sparse")

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

def get_sparse_driver():
    """🔧 CORRECTION: Connexion Sparse DIRECTE avec logique cloud/local"""
    load_dotenv()
    
    # Priorité absolue au Cloud si activé
    cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
    
    if cloud_enabled:
        print("🌐 MODE CLOUD SPARSE (connexion directe)")
        uri = os.getenv("NEO4J_SPARSE_CLOUD_URI")
        password = os.getenv("NEO4J_SPARSE_CLOUD_PASS")
        
        if uri and password:
            print(f"🔌 Connexion Cloud: {uri}")
            try:
                driver = GraphDatabase.driver(uri, auth=("neo4j", password))
                # Test rapide de connexion
                with driver.session() as session:
                    session.run("RETURN 1")
                print("✅ Connexion Cloud Sparse réussie")
                return driver
            except Exception as e:
                print(f"❌ Échec connexion Cloud Sparse: {e}")
                print("🔄 Fallback vers local...")
        else:
            print("❌ Credentials cloud manquants")
            print("🔄 Fallback vers local...")
    
    # Fallback Local
    print("🏠 MODE LOCAL SPARSE")
    db_uri = os.getenv("NEO4J_URI_SPARSE", "bolt://host.docker.internal:7689")
    db_user = os.getenv("NEO4J_USER_SPARSE", "neo4j")
    db_pass = os.getenv("NEO4J_PASS_SPARSE", "password")
    print(f"🔌 Connexion Local: {db_uri}")
    return GraphDatabase.driver(db_uri, auth=(db_user, db_pass))

def load_symptom_index_sparse():
    """Charge l'index FAISS des symptômes Sparse"""
    index_path = os.path.join(embedding_dir, "index.faiss")
    metadata_path = os.path.join(embedding_dir, "symptom_embedding_sparse.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ Index Sparse manquant dans {embedding_dir}")
    
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return index, metadata

def get_similar_symptoms_sparse(query: str) -> List[tuple]:
    """Trouve les symptômes similaires via FAISS dans KG Sparse"""
    try:
        model = get_model()
        index, metadata = load_symptom_index_sparse()
        symptom_names = metadata['symptom_names']
        symptoms_data = metadata['symptoms_data']
        
        # Recherche vectorielle
        query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(query_vec, symptom_top_k * 2)
        
        # Filtrage par seuil et formatage avec métadonnées Sparse (avec triplet_id)
        results = []
        for i, score in zip(indices[0], scores[0]):
            if score >= threshold and i < len(symptoms_data):
                symptom_data = symptoms_data[i]
                results.append((
                    symptom_data['name'], 
                    float(score),
                    symptom_data['triplet_id'],  # 🆕 ID unique pour Sparse
                    symptom_data['equipment']
                ))
                if len(results) >= symptom_top_k:
                    break
        
        return results
    except Exception as e:
        print(f"❌ Erreur recherche symptômes Sparse: {e}")
        return []

def query_neo4j_triplets_sparse(symptom: str, triplet_id: int) -> List[Dict]:
    """
    🎯 Récupère LE triplet spécifique pour un symptôme dans KG Sparse
    Dans Sparse: 1 symptôme avec triplet_id → 1 triplet exact (pas de propagation)
    """
    driver = get_sparse_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Symptom {name: $symptom, triplet_id: $triplet_id})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                WHERE c.triplet_id = $triplet_id AND r.triplet_id = $triplet_id
                RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, 
                       s.equipment AS equipment, s.triplet_id AS triplet_id
            """, symptom=symptom, triplet_id=triplet_id)
            
            triplets = [record.data() for record in result]
            return triplets
    except Exception as e:
        print(f"❌ Erreur Neo4j Sparse: {e}")
        return []
    finally:
        driver.close()

# === 🆕 FONCTION MULTI-QUERY FUSION SPARSE ===

def get_symptoms_with_variants_sparse(filtered_query: str, query_variants: List[str]) -> List[tuple]:
    """
    🆕 MULTI-QUERY FUSION SPARSE - Recherche symptoms avec query filtrée + variantes
    Stratégie MAX Score adaptée pour structure 1:1:1
    
    Args:
        filtered_query: Query optimisée par LLM
        query_variants: Liste des variantes générées par LLM
        
    Returns:
        List[tuple]: [(symptom_name, max_score, triplet_id, equipment), ...] top symptoms avec métadonnées Sparse
    """
    print(f"🔍 Multi-Query Fusion Sparse KG:")
    print(f"🎯 Query filtrée: '{filtered_query}'")
    print(f"🔄 Variantes: {query_variants}")
    
    # Poids par source (query filtrée prioritaire)
    weights = {
        "filtered": 1.0,      # Query LLM optimisée = poids max
        "variant": 0.8        # Variantes = poids réduit
    }
    
    # Dict pour stocker les scores par (symptom_name, triplet_id) - clé unique Sparse
    symptom_scores = {}
    
    # 1. Recherche avec query filtrée (poids principal)
    print(f"🎯 Recherche avec query filtrée...")
    filtered_symptoms = get_similar_symptoms_sparse(filtered_query)
    for symptom, score, triplet_id, equipment in filtered_symptoms:
        weighted_score = score * weights["filtered"]
        # Clé unique pour Sparse: (symptom_name, triplet_id)
        key = (symptom, triplet_id)
        symptom_scores[key] = {
            'score': weighted_score,
            'symptom': symptom,
            'triplet_id': triplet_id,
            'equipment': equipment
        }
        print(f"   ✅ Filtered: {symptom} (ID:{triplet_id}) → {weighted_score:.3f}")
    
    # 2. Recherche avec variantes (poids réduit)
    for i, variant in enumerate(query_variants[:2]):  # Max 2 variantes pour performance
        if not variant or variant == filtered_query:  # Skip si vide ou identique
            continue
            
        print(f"🔄 Recherche avec variante {i+1}: '{variant}'")
        variant_symptoms = get_similar_symptoms_sparse(variant)
        
        for symptom, score, triplet_id, equipment in variant_symptoms:
            weighted_score = score * weights["variant"]
            key = (symptom, triplet_id)
            
            # STRATÉGIE MAX Score - garde le meilleur score pour ce symptom + triplet_id
            if key in symptom_scores:
                old_score = symptom_scores[key]['score']
                new_score = max(old_score, weighted_score)
                symptom_scores[key]['score'] = new_score
                print(f"   🔄 Variant{i+1}: {symptom} (ID:{triplet_id}) → MAX({old_score:.3f}, {weighted_score:.3f}) = {new_score:.3f}")
            else:
                symptom_scores[key] = {
                    'score': weighted_score,
                    'symptom': symptom,
                    'triplet_id': triplet_id,
                    'equipment': equipment
                }
                print(f"   🆕 Variant{i+1}: {symptom} (ID:{triplet_id}) → {weighted_score:.3f}")
    
    # 3. Tri et limitation par score final (format compatible avec logique Sparse)
    sorted_symptoms = sorted(symptom_scores.values(), key=lambda x: x['score'], reverse=True)
    final_symptoms = sorted_symptoms[:symptom_top_k]
    
    # Conversion au format attendu: (symptom_name, score, triplet_id, equipment)
    result = [(s['symptom'], s['score'], s['triplet_id'], s['equipment']) for s in final_symptoms]
    
    print(f"✅ Multi-Query Sparse: {len(result)} symptoms sélectionnés (top scores MAX)")
    for i, (symptom, score, triplet_id, equipment) in enumerate(result, 1):
        print(f"   {i}. {symptom} (ID:{triplet_id}) → {score:.3f}")
    
    return result

# === 🆕 FONCTIONS EQUIPMENT MATCHING (CONSERVÉES) ===

def _extract_kg_equipments_sparse() -> List[str]:
    """Extrait tous les equipments uniques du KG Sparse"""
    try:
        driver = get_sparse_driver()
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.equipment IS NOT NULL
                RETURN DISTINCT n.equipment AS equipment
                ORDER BY n.equipment
            """)
            
            equipments = [record["equipment"] for record in result if record["equipment"]]
            print(f"📊 {len(equipments)} equipments trouvés dans KG Sparse")
            return equipments
            
    except Exception as e:
        print(f"⚠️ Erreur extraction equipments Sparse: {e}")
        return []
    finally:
        if 'driver' in locals():
            driver.close()

def _query_neo4j_triplets_sparse_with_equipment_filter(symptom: str, triplet_id: int, 
                                                      matched_equipment: Optional[str]) -> List[Dict]:
    """
    🎯 Récupère LE triplet spécifique Sparse avec filtrage equipment optionnel
    Structure 1:1:1 : 1 symptôme avec triplet_id → 1 triplet exact
    """
    driver = get_sparse_driver()
    try:
        with driver.session() as session:
            if matched_equipment:
                # Requête filtrée par equipment ET triplet_id
                result = session.run("""
                    MATCH (s:Symptom {name: $symptom, triplet_id: $triplet_id})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                    WHERE c.triplet_id = $triplet_id AND r.triplet_id = $triplet_id 
                    AND s.equipment = $equipment AND c.equipment = $equipment AND r.equipment = $equipment
                    RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, 
                           s.equipment AS equipment, s.triplet_id AS triplet_id
                """, symptom=symptom, triplet_id=triplet_id, equipment=matched_equipment)
            else:
                # Requête globale par triplet_id (comportement actuel)
                result = session.run("""
                    MATCH (s:Symptom {name: $symptom, triplet_id: $triplet_id})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                    WHERE c.triplet_id = $triplet_id AND r.triplet_id = $triplet_id
                    RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, 
                           s.equipment AS equipment, s.triplet_id AS triplet_id
                """, symptom=symptom, triplet_id=triplet_id)
            
            triplets = [record.data() for record in result]
            return triplets
    except Exception as e:
        print(f"❌ Erreur Neo4j Sparse avec equipment: {e}")
        return []
    finally:
        driver.close()

# === 🆕 FONCTION PRINCIPALE MULTI-QUERY + EQUIPMENT ===

def get_structured_context_with_variants_and_equipment_sparse(
    filtered_query: str,
    query_variants: List[str],
    equipment_info: Dict,
    format_type: str = "detailed", 
    max_triplets: Optional[int] = None
) -> str:
    """
    🆕 FONCTION PRINCIPALE - Multi-Query Fusion + Equipment Matching pour Sparse KG
    
    Args:
        filtered_query: Query optimisée par LLM  
        query_variants: Variantes générées par LLM
        equipment_info: Infos equipment pour matching
        format_type: Format de sortie ("detailed", "compact", "json")
        max_triplets: Limite triplets finaux
        
    Returns:
        str: Contexte KG Sparse formaté avec Multi-Query (structure 1:1:1)
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"🔍 Sparse KG avec Multi-Query Fusion + Equipment Matching")
        
        # === EQUIPMENT MATCHING (logique existante conservée) ===
        matched_equipment = None
        if equipment_info and equipment_info.get('primary_equipment') != 'unknown':
            try:
                from core.retrieval_graph.equipment_matcher import create_equipment_matcher
                matcher = create_equipment_matcher()
                
                # Extraction des equipments disponibles dans le KG Sparse
                kg_equipments = _extract_kg_equipments_sparse()
                
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
                    print("⚠️ Aucun equipment trouvé dans le KG Sparse")
                    
            except Exception as e:
                print(f"⚠️ Equipment matching échoué: {e}, fallback global")
                matched_equipment = None
        
        # === 🆕 RECHERCHE MULTI-QUERY SPARSE ===
        similar_symptoms = get_symptoms_with_variants_sparse(filtered_query, query_variants)
        
        if not similar_symptoms:
            return "No relevant structured information found with multi-query approach."
        
        # === RÉCUPÉRATION TRIPLETS EXACTS (structure 1:1:1 avec equipment filter) ===
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score, triplet_id, equipment in similar_symptoms:
            # Recherche du triplet exact correspondant (avec equipment filter)
            triplets = _query_neo4j_triplets_sparse_with_equipment_filter(
                symptom_name, triplet_id, matched_equipment
            )
            
            for triplet in triplets:
                # Clé unique basée sur triplet_id (structure Sparse)
                triplet_key = triplet['triplet_id']
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # === LIMITATION SIMPLE (déjà ordonné par pertinence Multi-Query) ===
        if len(all_triplets) > max_triplets:
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        equipment_info_str = f" (equipment: {matched_equipment})" if matched_equipment else " (global search)"
        print(f"✅ {len(selected)} triplets Sparse sélectionnés avec Multi-Query{equipment_info_str}")
        
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
                    f"Triplet {i} (multi-query score: {t['similarity_score']:.3f}, ID: {t['triplet_id']}):\n"
                    f"  Symptom: {t['symptom']}\n"
                    f"  Cause: {t['cause']}\n"
                    f"  Remedy: {t['remedy']}\n"
                    f"  Equipment: {t.get('equipment', 'N/A')}\n"
                )
            
            if lines:
                header = f"=== MULTI-QUERY SPARSE KNOWLEDGE GRAPH CONTEXT ===\n"
                header += f"Filtered Query: '{filtered_query}'\n"
                header += f"Variants: {query_variants}\n"
                header += f"Equipment filter: {matched_equipment or 'None (global search)'}\n"
                header += f"Structure: 1:1:1 (no semantic propagation) + Multi-Query Fusion\n"
                header += f"Triplets: {len(selected)}/{len(all_triplets)}\n\n"
                return header + "\n".join(lines)
            else:
                return "No structured context available with multi-query approach."
        
    except Exception as e:
        print(f"❌ Erreur Multi-Query Sparse KG: {e}")
        # Fallback vers fonction single-query
        print("🔄 Fallback vers recherche single-query...")
        return get_structured_context_sparse_with_equipment(filtered_query, equipment_info, format_type, max_triplets)

# === FONCTIONS SINGLE-QUERY (CONSERVÉES POUR RÉTROCOMPATIBILITÉ) ===

def get_structured_context_sparse_with_equipment(query: str, equipment_info: Dict, 
                                                format_type: str = "detailed", 
                                                max_triplets: Optional[int] = None) -> str:
    """
    🎯 FONCTION SINGLE-QUERY avec Equipment Matching (logique existante conservée)
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"🔍 Sparse KG avec Single-Query + Equipment Matching")
        
        # === EQUIPMENT MATCHING ===
        matched_equipment = None
        if equipment_info and equipment_info.get('primary_equipment') != 'unknown':
            try:
                from core.retrieval_graph.equipment_matcher import create_equipment_matcher
                matcher = create_equipment_matcher()
                
                kg_equipments = _extract_kg_equipments_sparse()
                
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
        similar_symptoms = get_similar_symptoms_sparse(query)
        if not similar_symptoms:
            return "No relevant structured information found in Sparse Knowledge Base."
        
        # === RÉCUPÉRATION TRIPLETS EXACTS ===
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score, triplet_id, equipment in similar_symptoms:
            triplets = _query_neo4j_triplets_sparse_with_equipment_filter(
                symptom_name, triplet_id, matched_equipment
            )
            
            for triplet in triplets:
                triplet_key = triplet['triplet_id']
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # === LIMITATION SIMPLE ===
        if len(all_triplets) > max_triplets:
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        equipment_info_str = f" (equipment: {matched_equipment})" if matched_equipment else " (global search)"
        print(f"✅ {len(selected)} triplets Sparse sélectionnés{equipment_info_str}")
        
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
                    f"Triplet {i} (score: {t['similarity_score']:.3f}, ID: {t['triplet_id']}):\n"
                    f"  Symptom: {t['symptom']}\n"
                    f"  Cause: {t['cause']}\n"
                    f"  Remedy: {t['remedy']}\n"
                    f"  Equipment: {t.get('equipment', 'N/A')}\n"
                )
            
            if lines:
                header = f"=== SPARSE KNOWLEDGE GRAPH CONTEXT ===\n"
                header += f"Query: '{query}'\n"
                header += f"Equipment filter: {matched_equipment or 'None (global search)'}\n"
                header += f"Structure: 1:1:1 (no semantic propagation) + Equipment Matching\n"
                header += f"Triplets: {len(selected)}/{len(all_triplets)}\n\n"
                return header + "\n".join(lines)
            else:
                return "No structured context available in Sparse KB."
    
    except Exception as e:
        print(f"❌ Erreur Sparse KG avec equipment: {e}")
        # Fallback vers fonction originale
        print("🔄 Fallback vers recherche Sparse globale...")
        return get_structured_context_sparse_original(query, format_type, max_triplets)

def get_structured_context_sparse_original(query: str, format_type: str = "detailed", 
                                          max_triplets: Optional[int] = None) -> str:
    """
    🎯 Fonction originale INCHANGÉE pour rétrocompatibilité
    Recherche dans KG Sparse: AUCUNE propagation sémantique, structure 1:1:1 pure
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"🔍 Recherche dans KG SPARSE (structure 1:1:1)")
        
        # 1. Recherche symptômes similaires (avec triplet_id)
        similar_symptoms = get_similar_symptoms_sparse(query)
        if not similar_symptoms:
            return "No relevant structured information found in Sparse Knowledge Base."
        
        # 2. Récupération triplets exacts (1:1 mapping)
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score, triplet_id, equipment in similar_symptoms:
            # Recherche du triplet exact correspondant
            triplets = query_neo4j_triplets_sparse(symptom_name, triplet_id)
            
            for triplet in triplets:
                # Clé unique basée sur triplet_id (pas de déduplication par contenu)
                triplet_key = triplet['triplet_id']
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # 3. Limitation simple (pas de tri complexe - déjà ordonné par pertinence)
        if len(all_triplets) > max_triplets:
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        print(f"✅ {len(selected)} triplets Sparse sélectionnés")
        
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
                    f"Triplet {i} (score: {t['similarity_score']:.3f}, ID: {t['triplet_id']}):\n"
                    f"  Symptom: {t['symptom']}\n"
                    f"  Cause: {t['cause']}\n"
                    f"  Remedy: {t['remedy']}\n"
                    f"  Equipment: {t.get('equipment', 'N/A')}\n"
                )
            
            if lines:
                header = f"=== SPARSE KNOWLEDGE GRAPH CONTEXT ===\n"
                header += f"Query: '{query}'\n"
                header += f"Structure: 1:1:1 (no semantic propagation)\n"
                header += f"Triplets: {len(selected)}/{len(all_triplets)}\n\n"
                return header + "\n".join(lines)
            else:
                return "No structured context available in Sparse KB."
    
    except Exception as e:
        print(f"❌ Erreur Sparse KG: {e}")
        return f"Error retrieving Sparse context: {str(e)}"

# === 🎯 INTERFACES PUBLIQUES (3 NIVEAUX) ===

def get_structured_context_sparse_with_multi_query(
    filtered_query: str,
    query_variants: List[str],
    equipment_info: Dict,
    format_type: str = "detailed",
    max_triplets: Optional[int] = None) -> str:
    """
    🎯 INTERFACE MULTI-QUERY - Nouvelle interface principale
    Utilisée par les générateurs RAG quand processed_query disponible
    """
    return get_structured_context_with_variants_and_equipment_sparse(
        filtered_query, query_variants, equipment_info, format_type, max_triplets
    )

def get_structured_context_sparse_with_equipment_filter(query: str, equipment_info: Dict, 
                                                        format_type: str = "detailed", 
                                                        max_triplets: Optional[int] = None) -> str:
    """
    🎯 INTERFACE SINGLE-QUERY + EQUIPMENT - Rétrocompatibilité
    Utilisée par les générateurs RAG en mode single-query avec equipment
    """
    return get_structured_context_sparse_with_equipment(query, equipment_info, format_type, max_triplets)

def get_structured_context_sparse(query: str, format_type: str = "detailed", 
                                 max_triplets: Optional[int] = None) -> str:
    """
    🎯 INTERFACE ORIGINALE - Rétrocompatibilité totale
    Utilisée par les anciens appels et mode classique
    """
    return get_structured_context_sparse_original(query, format_type, max_triplets)

# === TEST CLI SIMPLE ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Test Sparse KG Multi-Query: {query}")
        print("-" * 50)
        
        # Test Multi-Query
        filtered_query = f"ACAL-006 TPE operation error FANUC R-30iB"
        variants = ["ACAL-006 teach pendant error FANUC", "TPE operation failure ACAL-006"]
        equipment_info = {"primary_equipment": "FANUC R-30iB"}
        
        result = get_structured_context_sparse_with_multi_query(
            filtered_query, variants, equipment_info
        )
        print(result)
    else:
        print("Usage: python sparse_kg_querier.py 'votre requête'")
        print("Exemple: python sparse_kg_querier.py 'motor overheating FANUC'")