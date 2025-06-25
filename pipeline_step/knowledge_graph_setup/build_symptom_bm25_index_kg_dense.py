"""
BM25 Index Construction for Dense Knowledge Graph Symptoms

This module constructs a BM25 lexical search index specifically for the Dense Knowledge Graph
configuration. The index is built using unique symptom texts from the densified structure,
providing lexical search capabilities for the hybrid metric system in dense configurations.

Key components:
- Dense symptom extraction: Retrieval of unique symptom texts from Neo4j
- BM25 index construction: Whoosh-based lexical indexing with optimized schema
- Cloud/local connection management: Intelligent fallback system for database connectivity

Dependencies: neo4j, whoosh, pyyaml, python-dotenv, pathlib
Usage: docker run --rm -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data 
       -v $(pwd)/config:/app/config -v $(pwd)/pipeline_step:/app/pipeline_step 
       --network host diagnosis-app poetry run python pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_dense.py
"""

import os
import yaml
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, exists_in
from whoosh.analysis import StandardAnalyzer

load_dotenv()

def load_settings():
    """
    Load system configuration from settings file
    
    Returns:
        dict: Loaded configuration settings
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", "settings.yaml")
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_neo4j_connection(kg_type="dense"):
    """
    Establish intelligent Cloud/Local Neo4j connection
    
    Implements cloud-first connection strategy with automatic local fallback.
    Supports multiple Knowledge Graph types with appropriate credential selection.
    
    Args:
        kg_type (str): Knowledge Graph type ("dense", "sparse", or "dense_sc")
        
    Returns:
        neo4j.Driver: Configured Neo4j database driver
    """
    load_dotenv()
    
    # Priority to Cloud if enabled
    cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
    
    if cloud_enabled:
        print(f"CLOUD MODE for {kg_type.upper()}")
        
        if kg_type == "dense":
            uri = os.getenv("NEO4J_DENSE_CLOUD_URI")
            password = os.getenv("NEO4J_DENSE_CLOUD_PASS")
        elif kg_type == "sparse":
            uri = os.getenv("NEO4J_SPARSE_CLOUD_URI")
            password = os.getenv("NEO4J_SPARSE_CLOUD_PASS")
        elif kg_type == "dense_sc":
            uri = os.getenv("NEO4J_DENSE_SC_CLOUD_URI")
            password = os.getenv("NEO4J_DENSE_SC_CLOUD_PASS")
        
        if uri and password:
            print(f"Cloud connection {kg_type}: {uri}")
            return GraphDatabase.driver(uri, auth=("neo4j", password))
        else:
            print(f"Missing cloud credentials for {kg_type}")
            cloud_enabled = False
    
    # Local fallback
    print(f"LOCAL MODE for {kg_type.upper()}")
    
    if kg_type == "dense":
        uri = os.getenv("NEO4J_URI_DENSE", "bolt://host.docker.internal:7687")
        user = os.getenv("NEO4J_USER_DENSE", "neo4j")
        password = os.getenv("NEO4J_PASS_DENSE", "password")
    elif kg_type == "sparse":
        uri = os.getenv("NEO4J_URI_SPARSE", "bolt://host.docker.internal:7689")
        user = os.getenv("NEO4J_USER_SPARSE", "neo4j")
        password = os.getenv("NEO4J_PASS_SPARSE", "password")
    elif kg_type == "dense_sc":
        uri = os.getenv("NEO4J_URI_DENSE_SC", "bolt://host.docker.internal:7690")
        user = os.getenv("NEO4J_USER_DENSE_SC", "neo4j")
        password = os.getenv("NEO4J_PASS_DENSE_SC", "password")
    
    print(f"Local connection {kg_type}: {uri}")
    return GraphDatabase.driver(uri, auth=(user, password))

def extract_symptoms_from_kg():
    """
    Extract all unique symptoms from Dense Knowledge Graph
    
    Retrieves unique symptom nodes from the Dense KG, which contains densified
    relationships where symptoms may be connected to multiple causes. This
    extraction focuses on symptom text for lexical indexing purposes.
    
    Returns:
        list: List of unique symptom dictionaries with metadata
    """
    print("Extracting symptoms from Dense Knowledge Graph...")
    
    driver = get_neo4j_connection("dense")
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Symptom)
                RETURN DISTINCT s.name AS symptom_text, 
                       s.equipment AS equipment,
                       elementId(s) AS symptom_id
                ORDER BY s.name
            """)
            
            symptoms = []
            for record in result:
                symptoms.append({
                    'symptom_id': str(record['symptom_id']),
                    'symptom_text': record['symptom_text'],
                    'equipment': record['equipment'] or 'unknown'
                })
            
            print(f"Extracted {len(symptoms)} unique symptoms")
            return symptoms
            
    except Exception as e:
        print(f"Error extracting symptoms: {e}")
        return []
    finally:
        driver.close()

def build_symptom_bm25_index():
    """
    Construct BM25 index for Dense symptoms
    
    Creates a Whoosh-based BM25 lexical search index using unique symptom texts
    from the Dense Knowledge Graph. The index supports lexical similarity
    calculations as part of the hybrid metric system for dense configurations.
    
    Returns:
        str: Path to the created BM25 index directory
        
    Raises:
        ValueError: When no symptoms are found in the Dense Knowledge Graph
    """
    print("Constructing Dense symptom BM25 index...")
    
    # Configuration - corrected path
    config = load_settings()
    index_path = config["paths"]["bm25_index_path"]  # Corrected path
    
    # Directory creation
    os.makedirs(index_path, exist_ok=True)
    
    # Simple schema for symptoms
    schema = Schema(
        symptom_id=ID(stored=True, unique=True),
        symptom_text=TEXT(analyzer=StandardAnalyzer(), stored=True),
        equipment=ID(stored=True)
    )
    
    # Index creation
    if exists_in(index_path):
        print(f"Existing index removed from {index_path}")
        import shutil
        shutil.rmtree(index_path)
        os.makedirs(index_path, exist_ok=True)
    
    ix = create_in(index_path, schema)
    
    # Symptom extraction
    symptoms = extract_symptoms_from_kg()
    if not symptoms:
        raise ValueError("No symptoms found in Dense KG")
    
    # Indexing
    print(f"Indexing {len(symptoms)} symptoms...")
    writer = ix.writer()
    
    for symptom in symptoms:
        writer.add_document(
            symptom_id=symptom['symptom_id'],
            symptom_text=symptom['symptom_text'],
            equipment=symptom['equipment']
        )
    
    writer.commit()
    
    print(f"Dense BM25 index created: {index_path}")
    print(f"Indexed symptoms: {len(symptoms)}")
    
    return index_path

def main():
    """
    Main pipeline for Dense BM25 index construction
    
    Orchestrates the complete process of extracting unique symptoms from the Dense
    Knowledge Graph and constructing the corresponding BM25 lexical search index.
    Provides comprehensive error handling and progress reporting.
    """
    print("BM25 SYMPTOM INDEX CONSTRUCTION - DENSE KG")
    print("=" * 55)
    
    try:
        index_path = build_symptom_bm25_index()
        
        print("\nCONSTRUCTION COMPLETED")
        print(f"Index created: {index_path}")
        print("Ready for hybrid symptom search")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()