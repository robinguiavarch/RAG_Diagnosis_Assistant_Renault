"""
Sparse KG Querier: Multi-Query Fusion Knowledge Graph Retrieval for 1:1:1 Structure

This module provides comprehensive knowledge graph querying capabilities for the Sparse
knowledge base with 1:1:1 structure (one symptom to one cause to one remedy). It implements
multi-query fusion strategies, equipment matching, and direct triplet retrieval without
semantic propagation for precise and efficient sparse knowledge graph operations.

Key components:
- Multi-query fusion with filtered queries and variants using MAX score strategy
- Equipment matching and filtering for targeted sparse knowledge graph searches
- Direct 1:1:1 triplet retrieval without semantic propagation for precise matching
- Neo4j cloud and local connectivity with automatic fallback mechanisms
- Unique triplet ID-based deduplication for sparse structure integrity

Dependencies: neo4j, faiss, sentence-transformers, numpy, yaml, pickle
Usage: Import querying functions for Sparse knowledge graph retrieval operations
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

# Configuration Management
def load_settings():
    """Load configuration from settings.yaml with proper path resolution."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", "settings.yaml")
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config = load_settings()
model_name = config["models"]["embedding_model"]
threshold = config["graph_retrieval"]["symptom_similarity_threshold"]
symptom_top_k = config["graph_retrieval"]["symptom_top_k"]
triplets_limit = config["generation"]["top_k_triplets"]

# Paths for Sparse
script_dir = os.path.dirname(os.path.abspath(__file__))
embedding_dir = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "symptom_embedding_sparse")

# Core Functions

def get_model():
    """Load SentenceTransformer embedding model with device optimization."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return SentenceTransformer(model_name, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def get_sparse_driver():
    """
    Establish Sparse database connection with cloud/local fallback logic.
    
    Implements priority-based connection strategy: cloud first, then local fallback
    with comprehensive error handling and connection validation for Sparse KG.
    
    Returns:
        GraphDatabase.driver: Neo4j driver instance for Sparse database
    """
    load_dotenv()
    
    # Priority to cloud if enabled
    cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
    
    if cloud_enabled:
        print("Cloud mode Sparse (direct connection)")
        uri = os.getenv("NEO4J_SPARSE_CLOUD_URI")
        password = os.getenv("NEO4J_SPARSE_CLOUD_PASS")
        
        if uri and password:
            print(f"Connecting to cloud: {uri}")
            try:
                driver = GraphDatabase.driver(uri, auth=("neo4j", password))
                # Quick connection test
                with driver.session() as session:
                    session.run("RETURN 1")
                print("Cloud Sparse connection successful")
                return driver
            except Exception as e:
                print(f"Cloud Sparse connection failed: {e}")
                print("Falling back to local...")
        else:
            print("Cloud credentials missing")
            print("Falling back to local...")
    
    # Local fallback
    print("Local mode Sparse")
    db_uri = os.getenv("NEO4J_URI_SPARSE", "bolt://host.docker.internal:7689")
    db_user = os.getenv("NEO4J_USER_SPARSE", "neo4j")
    db_pass = os.getenv("NEO4J_PASS_SPARSE", "password")
    print(f"Connecting to local: {db_uri}")
    return GraphDatabase.driver(db_uri, auth=(db_user, db_pass))

def load_symptom_index_sparse():
    """
    Load FAISS index and metadata for Sparse symptom embeddings.
    
    Loads the specialized FAISS index for sparse knowledge graph structure
    with validation for proper index and metadata file existence.
    
    Returns:
        tuple: (FAISS index, metadata dict) for Sparse symptom search
        
    Raises:
        FileNotFoundError: If index or metadata files are missing
    """
    index_path = os.path.join(embedding_dir, "index.faiss")
    metadata_path = os.path.join(embedding_dir, "symptom_embedding_sparse.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Sparse index missing in {embedding_dir}")
    
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return index, metadata

def get_similar_symptoms_sparse(query: str) -> List[tuple]:
    """
    Find similar symptoms using FAISS search in Sparse KG structure.
    
    Performs semantic search specifically for sparse knowledge graph with
    unique triplet ID tracking and equipment metadata for precise 1:1:1 matching.
    
    Args:
        query (str): Search query for symptom matching
        
    Returns:
        List[tuple]: List of (symptom_name, similarity_score, triplet_id, equipment) tuples
    """
    try:
        model = get_model()
        index, metadata = load_symptom_index_sparse()
        symptom_names = metadata['symptom_names']
        symptoms_data = metadata['symptoms_data']
        
        # Vector search
        query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(query_vec, symptom_top_k * 2)
        
        # Threshold filtering and formatting with Sparse metadata (including triplet_id)
        results = []
        for i, score in zip(indices[0], scores[0]):
            if score >= threshold and i < len(symptoms_data):
                symptom_data = symptoms_data[i]
                results.append((
                    symptom_data['name'], 
                    float(score),
                    symptom_data['triplet_id'],  # Unique ID for Sparse
                    symptom_data['equipment']
                ))
                if len(results) >= symptom_top_k:
                    break
        
        return results
    except Exception as e:
        print(f"Error in Sparse symptom search: {e}")
        return []

def query_neo4j_triplets_sparse(symptom: str, triplet_id: int) -> List[Dict]:
    """
    Retrieve the specific triplet for a symptom in Sparse KG.
    
    Executes precise Neo4j query to extract the exact triplet for given symptom
    and triplet ID, maintaining 1:1:1 structure without semantic propagation.
    
    Args:
        symptom (str): Symptom name for triplet retrieval
        triplet_id (int): Unique triplet identifier for exact matching
        
    Returns:
        List[Dict]: Single triplet dictionary with comprehensive metadata
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
        print(f"Neo4j Sparse error: {e}")
        return []
    finally:
        driver.close()

# Multi-Query Fusion Functions

def get_symptoms_with_variants_sparse(filtered_query: str, query_variants: List[str]) -> List[tuple]:
    """
    Multi-query fusion for Sparse symptom retrieval with MAX score strategy.
    
    Implements sophisticated multi-query fusion combining filtered queries and
    variants with weighted scoring and MAX score selection optimized for
    sparse 1:1:1 structure with unique triplet ID tracking.
    
    Args:
        filtered_query (str): LLM-optimized query
        query_variants (List[str]): List of query variants generated by LLM
        
    Returns:
        List[tuple]: Top symptoms with maximum scores and sparse metadata
    """
    print(f"Multi-Query Fusion Sparse KG:")
    print(f"Filtered query: '{filtered_query}'")
    print(f"Variants: {query_variants}")
    
    # Weights by source (filtered query priority)
    weights = {
        "filtered": 1.0,      # LLM optimized query = max weight
        "variant": 0.8        # Variants = reduced weight
    }
    
    # Dict to store scores by (symptom_name, triplet_id) - unique Sparse key
    symptom_scores = {}
    
    # 1. Search with filtered query (main weight)
    print(f"Searching with filtered query...")
    filtered_symptoms = get_similar_symptoms_sparse(filtered_query)
    for symptom, score, triplet_id, equipment in filtered_symptoms:
        weighted_score = score * weights["filtered"]
        # Unique key for Sparse: (symptom_name, triplet_id)
        key = (symptom, triplet_id)
        symptom_scores[key] = {
            'score': weighted_score,
            'symptom': symptom,
            'triplet_id': triplet_id,
            'equipment': equipment
        }
        print(f"   Filtered: {symptom} (ID:{triplet_id}) → {weighted_score:.3f}")
    
    # 2. Search with variants (reduced weight)
    for i, variant in enumerate(query_variants[:2]):  # Max 2 variants for performance
        if not variant or variant == filtered_query:  # Skip if empty or identical
            continue
            
        print(f"Searching with variant {i+1}: '{variant}'")
        variant_symptoms = get_similar_symptoms_sparse(variant)
        
        for symptom, score, triplet_id, equipment in variant_symptoms:
            weighted_score = score * weights["variant"]
            key = (symptom, triplet_id)
            
            # MAX Score strategy - keep best score for this symptom + triplet_id
            if key in symptom_scores:
                old_score = symptom_scores[key]['score']
                new_score = max(old_score, weighted_score)
                symptom_scores[key]['score'] = new_score
                print(f"   Variant{i+1}: {symptom} (ID:{triplet_id}) → MAX({old_score:.3f}, {weighted_score:.3f}) = {new_score:.3f}")
            else:
                symptom_scores[key] = {
                    'score': weighted_score,
                    'symptom': symptom,
                    'triplet_id': triplet_id,
                    'equipment': equipment
                }
                print(f"   New Variant{i+1}: {symptom} (ID:{triplet_id}) → {weighted_score:.3f}")
    
    # 3. Sort and limit by final score (compatible format with Sparse logic)
    sorted_symptoms = sorted(symptom_scores.values(), key=lambda x: x['score'], reverse=True)
    final_symptoms = sorted_symptoms[:symptom_top_k]
    
    # Convert to expected format: (symptom_name, score, triplet_id, equipment)
    result = [(s['symptom'], s['score'], s['triplet_id'], s['equipment']) for s in final_symptoms]
    
    print(f"Multi-Query Sparse: {len(result)} symptoms selected (top MAX scores)")
    for i, (symptom, score, triplet_id, equipment) in enumerate(result, 1):
        print(f"   {i}. {symptom} (ID:{triplet_id}) → {score:.3f}")
    
    return result

# Equipment Matching Functions

def _extract_kg_equipments_sparse() -> List[str]:
    """
    Extract all unique equipment from Sparse knowledge graph.
    
    Queries the Sparse knowledge graph to retrieve all unique equipment
    identifiers for equipment matching operations with error handling.
    
    Returns:
        List[str]: Unique equipment identifiers from Sparse KG
    """
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
            print(f"{len(equipments)} equipment found in Sparse KG")
            return equipments
            
    except Exception as e:
        print(f"Warning: Equipment extraction error for Sparse: {e}")
        return []
    finally:
        if 'driver' in locals():
            driver.close()

def _query_neo4j_triplets_sparse_with_equipment_filter(symptom: str, triplet_id: int, 
                                                      matched_equipment: Optional[str]) -> List[Dict]:
    """
    Retrieve specific Sparse triplet with optional equipment filtering.
    
    Executes targeted Neo4j queries with equipment-based filtering when available,
    maintaining 1:1:1 structure integrity with precise triplet ID matching.
    
    Args:
        symptom (str): Symptom name for triplet retrieval
        triplet_id (int): Unique triplet identifier
        matched_equipment (Optional[str]): Equipment filter for targeted search
        
    Returns:
        List[Dict]: Filtered triplets with equipment-specific relevance
    """
    driver = get_sparse_driver()
    try:
        with driver.session() as session:
            if matched_equipment:
                # Equipment-filtered query with triplet_id
                result = session.run("""
                    MATCH (s:Symptom {name: $symptom, triplet_id: $triplet_id})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                    WHERE c.triplet_id = $triplet_id AND r.triplet_id = $triplet_id 
                    AND s.equipment = $equipment AND c.equipment = $equipment AND r.equipment = $equipment
                    RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, 
                           s.equipment AS equipment, s.triplet_id AS triplet_id
                """, symptom=symptom, triplet_id=triplet_id, equipment=matched_equipment)
            else:
                # Global query by triplet_id (current behavior)
                result = session.run("""
                    MATCH (s:Symptom {name: $symptom, triplet_id: $triplet_id})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                    WHERE c.triplet_id = $triplet_id AND r.triplet_id = $triplet_id
                    RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, 
                           s.equipment AS equipment, s.triplet_id AS triplet_id
                """, symptom=symptom, triplet_id=triplet_id)
            
            triplets = [record.data() for record in result]
            return triplets
    except Exception as e:
        print(f"Neo4j Sparse error with equipment: {e}")
        return []
    finally:
        driver.close()

# Main Multi-Query Function

def get_structured_context_with_variants_and_equipment_sparse(
    filtered_query: str,
    query_variants: List[str],
    equipment_info: Dict,
    format_type: str = "detailed", 
    max_triplets: Optional[int] = None
) -> str:
    """
    Main function for multi-query fusion with equipment matching in Sparse KG.
    
    Implements comprehensive retrieval strategy combining multi-query fusion,
    equipment matching, and direct 1:1:1 triplet retrieval for optimal
    sparse knowledge graph context generation without semantic propagation.
    
    Args:
        filtered_query (str): LLM-optimized query
        query_variants (List[str]): LLM-generated query variants
        equipment_info (Dict): Equipment information for matching
        format_type (str): Output format ("detailed", "compact", "json")
        max_triplets (Optional[int]): Maximum number of triplets to return
        
    Returns:
        str: Formatted Sparse KG context with multi-query results
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"Sparse KG with Multi-Query Fusion + Equipment Matching")
        
        # Equipment matching (existing logic preserved)
        matched_equipment = None
        if equipment_info and equipment_info.get('primary_equipment') != 'unknown':
            try:
                from core.retrieval_graph.equipment_matcher import create_equipment_matcher
                matcher = create_equipment_matcher()
                
                # Extract available equipment from Sparse KG
                kg_equipments = _extract_kg_equipments_sparse()
                
                if kg_equipments:
                    # Match LLM equipment → KG equipment
                    matched_equipment = matcher.match_equipment(
                        equipment_info['primary_equipment'], 
                        kg_equipments
                    )
                    
                    if matched_equipment:
                        print(f"Equipment match found: '{matched_equipment}' (score > 0.9)")
                    else:
                        print(f"No equipment match (< 0.9), global search")
                else:
                    print("No equipment found in Sparse KG")
                    
            except Exception as e:
                print(f"Warning: Equipment matching failed: {e}, fallback to global")
                matched_equipment = None
        
        # Multi-query search Sparse
        similar_symptoms = get_symptoms_with_variants_sparse(filtered_query, query_variants)
        
        if not similar_symptoms:
            return "No relevant structured information found with multi-query approach."
        
        # Exact triplet retrieval (1:1:1 structure with equipment filter)
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score, triplet_id, equipment in similar_symptoms:
            # Search for exact corresponding triplet (with equipment filter)
            triplets = _query_neo4j_triplets_sparse_with_equipment_filter(
                symptom_name, triplet_id, matched_equipment
            )
            
            for triplet in triplets:
                # Unique key based on triplet_id (Sparse structure)
                triplet_key = triplet['triplet_id']
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # Simple limitation (already ordered by Multi-Query relevance)
        if len(all_triplets) > max_triplets:
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        equipment_info_str = f" (equipment: {matched_equipment})" if matched_equipment else " (global search)"
        print(f"{len(selected)} Sparse triplets selected with Multi-Query{equipment_info_str}")
        
        # Formatting
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
        print(f"Error in Multi-Query Sparse KG: {e}")
        # Fallback to single-query function
        print("Fallback to single-query search...")
        return get_structured_context_sparse_with_equipment(filtered_query, equipment_info, format_type, max_triplets)

# Single-Query Functions (Preserved for Backward Compatibility)

def get_structured_context_sparse_with_equipment(query: str, equipment_info: Dict, 
                                                format_type: str = "detailed", 
                                                max_triplets: Optional[int] = None) -> str:
    """
    Single-query function with equipment matching (existing logic preserved).
    
    Provides backward compatibility for single-query operations with equipment
    matching capabilities for targeted Sparse knowledge graph retrieval.
    
    Args:
        query (str): Single search query
        equipment_info (Dict): Equipment information for matching
        format_type (str): Output format specification
        max_triplets (Optional[int]): Maximum triplet limit
        
    Returns:
        str: Formatted Sparse context with equipment filtering
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"Sparse KG with Single-Query + Equipment Matching")
        
        # Equipment matching
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
                        print(f"Equipment match found: '{matched_equipment}' (score > 0.9)")
                    else:
                        print(f"No equipment match (< 0.9), global search")
                        
            except Exception as e:
                print(f"Warning: Equipment matching failed: {e}, fallback to global")
                matched_equipment = None
        
        # Single-query search
        similar_symptoms = get_similar_symptoms_sparse(query)
        if not similar_symptoms:
            return "No relevant structured information found in Sparse Knowledge Base."
        
        # Exact triplet retrieval
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
        
        # Simple limitation
        if len(all_triplets) > max_triplets:
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        equipment_info_str = f" (equipment: {matched_equipment})" if matched_equipment else " (global search)"
        print(f"{len(selected)} Sparse triplets selected{equipment_info_str}")
        
        # Formatting
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
        print(f"Error in Sparse KG with equipment: {e}")
        # Fallback to original function
        print("Fallback to global Sparse search...")
        return get_structured_context_sparse_original(query, format_type, max_triplets)

def get_structured_context_sparse_original(query: str, format_type: str = "detailed", 
                                          max_triplets: Optional[int] = None) -> str:
    """
    Original function unchanged for backward compatibility.
    
    Maintains original Sparse functionality for legacy compatibility with
    direct 1:1:1 structure search without semantic propagation.
    
    Args:
        query (str): Search query string
        format_type (str): Output format specification
        max_triplets (Optional[int]): Maximum triplet limit
        
    Returns:
        str: Original Sparse knowledge graph context
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"Searching in SPARSE KG (1:1:1 structure)")
        
        # 1. Search similar symptoms (with triplet_id)
        similar_symptoms = get_similar_symptoms_sparse(query)
        if not similar_symptoms:
            return "No relevant structured information found in Sparse Knowledge Base."
        
        # 2. Exact triplet retrieval (1:1 mapping)
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score, triplet_id, equipment in similar_symptoms:
            # Search for exact corresponding triplet
            triplets = query_neo4j_triplets_sparse(symptom_name, triplet_id)
            
            for triplet in triplets:
                # Unique key based on triplet_id (no content deduplication)
                triplet_key = triplet['triplet_id']
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # 3. Simple limitation (no complex sorting - already ordered by relevance)
        if len(all_triplets) > max_triplets:
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        print(f"{len(selected)} Sparse triplets selected")
        
        # 4. Formatting
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
        print(f"Error in Sparse KG: {e}")
        return f"Error retrieving Sparse context: {str(e)}"

# Public Interfaces (3 Levels)

def get_structured_context_sparse_with_multi_query(
    filtered_query: str,
    query_variants: List[str],
    equipment_info: Dict,
    format_type: str = "detailed",
    max_triplets: Optional[int] = None) -> str:
    """
    Multi-query interface - new main interface.
    
    Primary interface for RAG generators when processed_query is available
    with comprehensive multi-query fusion and equipment matching for Sparse KG.
    
    Args:
        filtered_query (str): LLM-optimized query
        query_variants (List[str]): Query variants for fusion
        equipment_info (Dict): Equipment matching information
        format_type (str): Output format specification
        max_triplets (Optional[int]): Maximum triplet limit
        
    Returns:
        str: Multi-query Sparse knowledge graph context
    """
    return get_structured_context_with_variants_and_equipment_sparse(
        filtered_query, query_variants, equipment_info, format_type, max_triplets
    )

def get_structured_context_sparse_with_equipment_filter(query: str, equipment_info: Dict, 
                                                        format_type: str = "detailed", 
                                                        max_triplets: Optional[int] = None) -> str:
    """
    Single-query + equipment interface - backward compatibility.
    
    Interface for RAG generators in single-query mode with equipment
    matching for targeted Sparse knowledge graph retrieval.
    
    Args:
        query (str): Single search query
        equipment_info (Dict): Equipment matching information
        format_type (str): Output format specification
        max_triplets (Optional[int]): Maximum triplet limit
        
    Returns:
        str: Equipment-filtered Sparse context
    """
    return get_structured_context_sparse_with_equipment(query, equipment_info, format_type, max_triplets)

def get_structured_context_sparse(query: str, format_type: str = "detailed", 
                                 max_triplets: Optional[int] = None) -> str:
    """
    Original interface - total backward compatibility.
    
    Interface for legacy calls and classic mode operations with original
    Sparse functionality and direct 1:1:1 structure retrieval.
    
    Args:
        query (str): Search query string
        format_type (str): Output format specification
        max_triplets (Optional[int]): Maximum triplet limit
        
    Returns:
        str: Original Sparse knowledge graph context
    """
    return get_structured_context_sparse_original(query, format_type, max_triplets)

# CLI Testing
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
        print("Usage: python sparse_kg_querier.py 'your query'")
        print("Example: python sparse_kg_querier.py 'motor overheating FANUC'")