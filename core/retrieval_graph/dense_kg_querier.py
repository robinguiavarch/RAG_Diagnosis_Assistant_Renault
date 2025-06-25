"""
Dense KG Querier: Multi-Query Fusion Knowledge Graph Retrieval with Semantic Propagation

This module provides comprehensive knowledge graph querying capabilities for the Dense
knowledge base with standard semantic propagation. It implements multi-query fusion
strategies, equipment matching, and hybrid search capabilities combining BM25, FAISS,
and Levenshtein distance for optimal dense knowledge graph retrieval performance.

Key components:
- Multi-query fusion with filtered queries and variants using MAX score strategy
- Hybrid search combining BM25, FAISS, and Levenshtein distance with fallback mechanisms
- Equipment matching and filtering for targeted dense knowledge graph searches
- Standard semantic propagation for comprehensive triplet discovery
- Neo4j cloud and local connectivity with automatic fallback mechanisms

Dependencies: neo4j, faiss, sentence-transformers, numpy, yaml, pickle
Usage: Import querying functions for Dense knowledge graph retrieval operations
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

# Paths for Dense
script_dir = os.path.dirname(os.path.abspath(__file__))
embedding_dir = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "symptom_embedding_dense")

# Hybrid matcher import with fallback
try:
    from core.retrieval_graph.hybrid_symptom_matcher import create_hybrid_symptom_matcher
    HYBRID_MATCHER_AVAILABLE = True
except ImportError:
    print("Warning: Hybrid matcher not available, falling back to FAISS")
    HYBRID_MATCHER_AVAILABLE = False

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

def get_dense_driver():
    """
    Establish Dense database connection with cloud/local fallback logic.
    
    Implements priority-based connection strategy: cloud first, then local fallback
    with comprehensive error handling and connection validation for Dense KG.
    
    Returns:
        GraphDatabase.driver: Neo4j driver instance for Dense database
    """
    load_dotenv()
    
    # Priority to cloud if enabled
    cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
    
    if cloud_enabled:
        print("Cloud mode Dense (direct connection)")
        uri = os.getenv("NEO4J_DENSE_CLOUD_URI")
        password = os.getenv("NEO4J_DENSE_CLOUD_PASS")
        
        if uri and password:
            print(f"Connecting to cloud: {uri}")
            try:
                driver = GraphDatabase.driver(uri, auth=("neo4j", password))
                # Quick connection test
                with driver.session() as session:
                    session.run("RETURN 1")
                print("Cloud Dense connection successful")
                return driver
            except Exception as e:
                print(f"Cloud Dense connection failed: {e}")
                print("Falling back to local...")
        else:
            print("Cloud credentials missing")
            print("Falling back to local...")
    
    # Local fallback
    print("Local mode Dense")
    db_uri = os.getenv("NEO4J_URI_DENSE", "bolt://host.docker.internal:7687")
    db_user = os.getenv("NEO4J_USER_DENSE", "neo4j")
    db_pass = os.getenv("NEO4J_PASS_DENSE", "password")
    print(f"Connecting to local: {db_uri}")
    return GraphDatabase.driver(db_uri, auth=(db_user, db_pass))

def load_symptom_index():
    """
    Load FAISS index and metadata for Dense symptom embeddings.
    
    Loads the FAISS index for dense knowledge graph structure with
    validation for proper index and metadata file existence.
    
    Returns:
        tuple: (FAISS index, metadata dict) for Dense symptom search
        
    Raises:
        FileNotFoundError: If index or metadata files are missing
    """
    index_path = os.path.join(embedding_dir, "index.faiss")
    metadata_path = os.path.join(embedding_dir, "symptom_embedding_dense.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Index missing in {embedding_dir}")
    
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    return index, metadata

def get_similar_symptoms(query: str) -> List[tuple]:
    """
    Find similar symptoms using hybrid search or FAISS fallback.
    
    Implements sophisticated hybrid search combining BM25, FAISS, and Levenshtein
    distance when available, with graceful fallback to FAISS-only search for
    robust symptom matching in dense knowledge graphs.
    
    Args:
        query (str): Search query for symptom matching
        
    Returns:
        List[tuple]: List of (symptom_name, similarity_score) tuples
    """
    # Check if hybrid search is available and enabled
    if HYBRID_MATCHER_AVAILABLE:
        try:
            # Use hybrid matcher
            matcher = create_hybrid_symptom_matcher()
            if matcher.enabled:
                print("Using HYBRID search")
                hybrid_results = matcher.search_hybrid(query, symptom_top_k)
                
                # Convert to expected format (symptom_name, score)
                results = []
                for result in hybrid_results:
                    results.append((result['symptom_text'], result['hybrid_score']))
                
                return results
            else:
                print("Hybrid search disabled, falling back to FAISS")
        except Exception as e:
            print(f"Hybrid search error: {e}, falling back to FAISS")
    
    # FALLBACK: Classic FAISS search
    print("Using classic FAISS search")
    try:
        model = get_model()
        index, metadata = load_symptom_index()
        symptom_names = metadata['symptom_names']
        
        # Vector search
        query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(query_vec, symptom_top_k * 2)
        
        # Threshold filtering
        results = []
        for i, score in zip(indices[0], scores[0]):
            if score >= threshold and i < len(symptom_names):
                results.append((symptom_names[i], float(score)))
                if len(results) >= symptom_top_k:
                    break
        
        return results
    except Exception as e:
        print(f"Error in symptom search: {e}")
        return []

def query_neo4j_triplets(symptom: str) -> List[Dict]:
    """
    Retrieve triplets for a symptom from Dense knowledge graph.
    
    Executes Neo4j query to extract symptom-cause-remedy triplets with
    equipment information and semantic propagation support.
    
    Args:
        symptom (str): Symptom name for triplet retrieval
        
    Returns:
        List[Dict]: List of triplet dictionaries with metadata
    """
    driver = get_dense_driver()
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Symptom {name: $symptom})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, s.equipment AS equipment
            """, symptom=symptom)
            
            return [record.data() for record in result]
    except Exception as e:
        print(f"Neo4j error: {e}")
        return []
    finally:
        driver.close()

# Multi-Query Fusion Functions

def get_symptoms_with_variants(filtered_query: str, query_variants: List[str]) -> List[tuple]:
    """
    Multi-query fusion for Dense symptom retrieval with MAX score strategy.
    
    Implements sophisticated multi-query fusion combining filtered queries and
    variants with weighted scoring and MAX score selection for optimal symptom
    identification in dense knowledge graphs with semantic propagation.
    
    Args:
        filtered_query (str): LLM-optimized query
        query_variants (List[str]): List of query variants generated by LLM
        
    Returns:
        List[tuple]: Top symptoms with maximum scores from multi-query fusion
    """
    print(f"Multi-Query Fusion Dense KG:")
    print(f"Filtered query: '{filtered_query}'")
    print(f"Variants: {query_variants}")
    
    # Weights by source (filtered query priority)
    weights = {
        "filtered": 1.0,      # LLM optimized query = max weight
        "variant": 0.8        # Variants = reduced weight
    }
    
    symptom_scores = {}
    
    # 1. Search with filtered query (main weight)
    print(f"Searching with filtered query...")
    filtered_symptoms = get_similar_symptoms(filtered_query)
    for symptom, score in filtered_symptoms:
        weighted_score = score * weights["filtered"]
        symptom_scores[symptom] = weighted_score
        print(f"   Filtered: {symptom} → {weighted_score:.3f}")
    
    # 2. Search with variants (reduced weight)
    for i, variant in enumerate(query_variants[:2]):  # Max 2 variants for performance
        if not variant or variant == filtered_query:  # Skip if empty or identical
            continue
            
        print(f"Searching with variant {i+1}: '{variant}'")
        variant_symptoms = get_similar_symptoms(variant)
        
        for symptom, score in variant_symptoms:
            weighted_score = score * weights["variant"]
            
            # MAX Score strategy - keep best score for this symptom
            if symptom in symptom_scores:
                old_score = symptom_scores[symptom]
                new_score = max(old_score, weighted_score)
                symptom_scores[symptom] = new_score
                print(f"   Variant{i+1}: {symptom} → MAX({old_score:.3f}, {weighted_score:.3f}) = {new_score:.3f}")
            else:
                symptom_scores[symptom] = weighted_score
                print(f"   New Variant{i+1}: {symptom} → {weighted_score:.3f}")
    
    # 3. Sort and limit by final score
    sorted_symptoms = sorted(symptom_scores.items(), key=lambda x: x[1], reverse=True)
    final_symptoms = sorted_symptoms[:symptom_top_k]
    
    print(f"Multi-Query Dense: {len(final_symptoms)} symptoms selected (top MAX scores)")
    for i, (symptom, score) in enumerate(final_symptoms, 1):
        print(f"   {i}. {symptom} → {score:.3f}")
    
    return final_symptoms

# Equipment Matching Functions

def _extract_kg_equipments_dense() -> List[str]:
    """
    Extract all unique equipment from Dense knowledge graph.
    
    Queries the Dense knowledge graph to retrieve all unique equipment
    identifiers for equipment matching operations with error handling.
    
    Returns:
        List[str]: Unique equipment identifiers from Dense KG
    """
    try:
        driver = get_dense_driver()
        with driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.equipment IS NOT NULL
                RETURN DISTINCT n.equipment AS equipment
                ORDER BY n.equipment
            """)
            
            equipments = [record["equipment"] for record in result if record["equipment"]]
            print(f"{len(equipments)} equipment found in Dense KG")
            return equipments
            
    except Exception as e:
        print(f"Warning: Equipment extraction error for Dense: {e}")
        return []
    finally:
        if 'driver' in locals():
            driver.close()

def _query_neo4j_triplets_with_equipment_filter(symptom: str, matched_equipment: Optional[str]) -> List[Dict]:
    """
    Retrieve triplets for symptom with optional equipment filtering.
    
    Executes targeted Neo4j queries with equipment-based filtering when available,
    falling back to global search for comprehensive triplet retrieval.
    
    Args:
        symptom (str): Symptom name for triplet retrieval
        matched_equipment (Optional[str]): Equipment filter for targeted search
        
    Returns:
        List[Dict]: Filtered triplets with equipment-specific relevance
    """
    driver = get_dense_driver()
    try:
        with driver.session() as session:
            if matched_equipment:
                # Equipment-filtered query
                result = session.run("""
                    MATCH (s:Symptom {name: $symptom})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                    WHERE s.equipment = $equipment
                    RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, s.equipment AS equipment
                """, symptom=symptom, equipment=matched_equipment)
            else:
                # Global query (current behavior)
                result = session.run("""
                    MATCH (s:Symptom {name: $symptom})-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                    RETURN s.name AS symptom, c.name AS cause, r.name AS remedy, s.equipment AS equipment
                """, symptom=symptom)
            
            return [record.data() for record in result]
    except Exception as e:
        print(f"Neo4j Dense error with equipment: {e}")
        return []
    finally:
        driver.close()

# Main Multi-Query Function

def get_structured_context_with_variants_and_equipment(
    filtered_query: str,
    query_variants: List[str],
    equipment_info: Dict,
    format_type: str = "detailed", 
    max_triplets: Optional[int] = None
) -> str:
    """
    Main function for multi-query fusion with equipment matching in Dense KG.
    
    Implements comprehensive retrieval strategy combining multi-query fusion,
    equipment matching, and semantic propagation for optimal dense knowledge
    graph context generation with hybrid search capabilities.
    
    Args:
        filtered_query (str): LLM-optimized query
        query_variants (List[str]): LLM-generated query variants
        equipment_info (Dict): Equipment information for matching
        format_type (str): Output format ("detailed", "compact", "json")
        max_triplets (Optional[int]): Maximum number of triplets to return
        
    Returns:
        str: Formatted Dense KG context with multi-query results
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"Dense KG with Multi-Query Fusion + Equipment Matching")
        
        # Equipment matching (existing logic preserved)
        matched_equipment = None
        if equipment_info and equipment_info.get('primary_equipment') != 'unknown':
            try:
                from core.retrieval_graph.equipment_matcher import create_equipment_matcher
                matcher = create_equipment_matcher()
                
                # Extract available equipment from Dense KG
                kg_equipments = _extract_kg_equipments_dense()
                
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
                    print("No equipment found in Dense KG")
                    
            except Exception as e:
                print(f"Warning: Equipment matching failed: {e}, fallback to global")
                matched_equipment = None
        
        # Multi-query search
        similar_symptoms = get_symptoms_with_variants(filtered_query, query_variants)
        
        if not similar_symptoms:
            return "No relevant structured information found with multi-query approach."
        
        # Triplet retrieval (with equipment filter and semantic propagation)
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score in similar_symptoms:
            triplets = _query_neo4j_triplets_with_equipment_filter(symptom_name, matched_equipment)
            
            for triplet in triplets:
                triplet_key = (triplet['symptom'], triplet['cause'], triplet['remedy'])
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # Limitation and sorting
        if len(all_triplets) > max_triplets:
            all_triplets.sort(key=lambda x: x['similarity_score'], reverse=True)
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        equipment_info_str = f" (equipment: {matched_equipment})" if matched_equipment else " (global search)"
        print(f"{len(selected)} Dense triplets selected with Multi-Query{equipment_info_str}")
        
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
                    f"Triplet {i} (multi-query score: {t['similarity_score']:.3f}):\n"
                    f"  Symptom: {t['symptom']}\n"
                    f"  Cause: {t['cause']}\n"
                    f"  Remedy: {t['remedy']}\n"
                    f"  Equipment: {t.get('equipment', 'N/A')}\n"
                )
            
            if lines:
                header = f"=== MULTI-QUERY DENSE KNOWLEDGE GRAPH CONTEXT ===\n"
                header += f"Filtered Query: '{filtered_query}'\n"
                header += f"Variants: {query_variants}\n"
                header += f"Equipment filter: {matched_equipment or 'None (global search)'}\n"
                header += f"Method: Multi-Query Fusion (MAX Score) + Equipment Matching\n"
                header += f"Triplets: {len(selected)}/{len(all_triplets)}\n\n"
                return header + "\n".join(lines)
            else:
                return "No structured context available with multi-query approach."
        
    except Exception as e:
        print(f"Error in Multi-Query Dense KG: {e}")
        # Fallback to single-query function
        print("Fallback to single-query search...")
        return get_structured_context_with_equipment(filtered_query, equipment_info, format_type, max_triplets)

# Single-Query Functions (Preserved for Backward Compatibility)

def get_structured_context_with_equipment(query: str, equipment_info: Dict, 
                                         format_type: str = "detailed", 
                                         max_triplets: Optional[int] = None) -> str:
    """
    Single-query function with equipment matching (existing logic preserved).
    
    Provides backward compatibility for single-query operations with equipment
    matching capabilities for targeted Dense knowledge graph retrieval.
    
    Args:
        query (str): Single search query
        equipment_info (Dict): Equipment information for matching
        format_type (str): Output format specification
        max_triplets (Optional[int]): Maximum triplet limit
        
    Returns:
        str: Formatted Dense context with equipment filtering
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        print(f"Dense KG with Single-Query + Equipment Matching")
        
        # Equipment matching
        matched_equipment = None
        if equipment_info and equipment_info.get('primary_equipment') != 'unknown':
            try:
                from .equipment_matcher import create_equipment_matcher
                matcher = create_equipment_matcher()
                
                kg_equipments = _extract_kg_equipments_dense()
                
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
        similar_symptoms = get_similar_symptoms(query)
        if not similar_symptoms:
            return "No relevant structured information found in Dense Knowledge Base."
        
        # Triplet retrieval
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score in similar_symptoms:
            triplets = _query_neo4j_triplets_with_equipment_filter(symptom_name, matched_equipment)
            
            for triplet in triplets:
                triplet_key = (triplet['symptom'], triplet['cause'], triplet['remedy'])
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # Limitation and sorting
        if len(all_triplets) > max_triplets:
            all_triplets.sort(key=lambda x: x['similarity_score'], reverse=True)
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
        equipment_info_str = f" (equipment: {matched_equipment})" if matched_equipment else " (global search)"
        print(f"{len(selected)} Dense triplets selected{equipment_info_str}")
        
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
                    f"Triplet {i} (score: {t['similarity_score']:.3f}):\n"
                    f"  Symptom: {t['symptom']}\n"
                    f"  Cause: {t['cause']}\n"
                    f"  Remedy: {t['remedy']}\n"
                    f"  Equipment: {t.get('equipment', 'N/A')}\n"
                )
            
            if lines:
                header = f"=== DENSE KNOWLEDGE GRAPH CONTEXT ===\n"
                header += f"Query: '{query}'\n"
                header += f"Equipment filter: {matched_equipment or 'None (global search)'}\n"
                header += f"Method: Single-Query + Equipment Matching\n"
                header += f"Triplets: {len(selected)}/{len(all_triplets)}\n\n"
                return header + "\n".join(lines)
            else:
                return "No structured context available in Dense KB."
    
    except Exception as e:
        print(f"Error in Dense KG with equipment: {e}")
        # Fallback to original function
        print("Fallback to global Dense search...")
        return get_structured_context_original(query, format_type, max_triplets)

def get_structured_context_original(query: str, format_type: str = "detailed", 
                                  max_triplets: Optional[int] = None) -> str:
    """
    Original function unchanged for total backward compatibility.
    
    Maintains original Dense functionality for legacy compatibility with
    standard semantic propagation and comprehensive triplet retrieval.
    
    Args:
        query (str): Search query string
        format_type (str): Output format specification
        max_triplets (Optional[int]): Maximum triplet limit
        
    Returns:
        str: Original Dense knowledge graph context
    """
    try:
        if max_triplets is None:
            max_triplets = triplets_limit
        
        # 1. Search similar symptoms
        similar_symptoms = get_similar_symptoms(query)
        if not similar_symptoms:
            return "No relevant structured information found in Knowledge Base."
        
        # 2. Triplet retrieval
        all_triplets = []
        seen = set()
        
        for symptom_name, similarity_score in similar_symptoms:
            triplets = query_neo4j_triplets(symptom_name)
            
            for triplet in triplets:
                triplet_key = (triplet['symptom'], triplet['cause'], triplet['remedy'])
                if triplet_key not in seen:
                    seen.add(triplet_key)
                    triplet['similarity_score'] = similarity_score
                    all_triplets.append(triplet)
        
        # 3. Limitation and sorting
        if len(all_triplets) > max_triplets:
            all_triplets.sort(key=lambda x: x['similarity_score'], reverse=True)
            selected = all_triplets[:max_triplets]
        else:
            selected = all_triplets
        
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
                    f"Triplet {i} (score: {t['similarity_score']:.3f}):\n"
                    f"  Symptom: {t['symptom']}\n"
                    f"  Cause: {t['cause']}\n"
                    f"  Remedy: {t['remedy']}\n"
                )
            
            if lines:
                header = f"=== KNOWLEDGE GRAPH CONTEXT ===\n"
                header += f"Query: '{query}'\n"
                header += f"Triplets: {len(selected)}/{len(all_triplets)}\n\n"
                return header + "\n".join(lines)
            else:
                return "No structured context available."
    
    except Exception as e:
        print(f"Error: {e}")
        return f"Error retrieving context: {str(e)}"

# Public Interfaces (3 Levels)

def get_structured_context_with_multi_query(
    filtered_query: str,
    query_variants: List[str],
    equipment_info: Dict,
    format_type: str = "detailed",
    max_triplets: Optional[int] = None
) -> str:
    """
    Multi-query interface - new main interface.
    
    Primary interface for RAG generators when processed_query is available
    with comprehensive multi-query fusion and equipment matching for Dense KG.
    
    Args:
        filtered_query (str): LLM-optimized query
        query_variants (List[str]): Query variants for fusion
        equipment_info (Dict): Equipment matching information
        format_type (str): Output format specification
        max_triplets (Optional[int]): Maximum triplet limit
        
    Returns:
        str: Multi-query Dense knowledge graph context
    """
    return get_structured_context_with_variants_and_equipment(
        filtered_query, query_variants, equipment_info, format_type, max_triplets
    )

def get_structured_context_with_equipment_filter(query: str, equipment_info: Dict, 
                                                format_type: str = "detailed", 
                                                max_triplets: Optional[int] = None) -> str:
    """
    Single-query + equipment interface - backward compatibility.
    
    Interface for RAG generators in single-query mode with equipment
    matching for targeted Dense knowledge graph retrieval.
    
    Args:
        query (str): Single search query
        equipment_info (Dict): Equipment matching information
        format_type (str): Output format specification
        max_triplets (Optional[int]): Maximum triplet limit
        
    Returns:
        str: Equipment-filtered Dense context
    """
    return get_structured_context_with_equipment(query, equipment_info, format_type, max_triplets)

def get_structured_context(query: str, format_type: str = "detailed", 
                          max_triplets: Optional[int] = None) -> str:
    """
    Original interface - total backward compatibility.
    
    Interface for legacy calls and classic mode operations with original
    Dense functionality and standard semantic propagation.
    
    Args:
        query (str): Search query string
        format_type (str): Output format specification
        max_triplets (Optional[int]): Maximum triplet limit
        
    Returns:
        str: Original Dense knowledge graph context
    """
    return get_structured_context_original(query, format_type, max_triplets)

# CLI Testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Test Dense KG Multi-Query: {query}")
        print("-" * 50)
        
        # Test Multi-Query
        filtered_query = f"ACAL-006 TPE operation error FANUC R-30iB"
        variants = ["ACAL-006 teach pendant error FANUC", "TPE operation failure ACAL-006"]
        equipment_info = {"primary_equipment": "FANUC R-30iB"}
        
        result = get_structured_context_with_multi_query(
            filtered_query, variants, equipment_info
        )
        print(result)
    else:
        print("Usage: python dense_kg_querier.py 'your query'")
        print("Example: python dense_kg_querier.py 'motor overheating FANUC'")