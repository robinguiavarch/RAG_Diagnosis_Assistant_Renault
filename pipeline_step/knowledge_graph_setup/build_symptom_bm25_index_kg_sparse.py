"""
BM25 Index Construction for Sparse Knowledge Graph Symptoms

This module constructs a BM25 lexical search index specifically for the Sparse Knowledge Graph
configuration. The index is built using individual symptom texts from the 1:1:1 structure,
providing lexical search capabilities for the hybrid metric system in sparse configurations.

Key components:
- Sparse symptom extraction: Retrieval of symptom texts with triplet IDs from Neo4j
- BM25 index construction: Whoosh-based lexical indexing with simplified schema
- Cloud/local connection management: Intelligent fallback system for database connectivity

Dependencies: neo4j, whoosh, pyyaml, python-dotenv
Usage: docker run --rm -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data 
       -v $(pwd)/config:/app/config -v $(pwd)/pipeline_step:/app/pipeline_step 
       --network host diagnosis-app poetry run python pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_sparse.py
"""

import os
import yaml
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

def get_neo4j_connection(kg_type="sparse"):
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

def extract_symptoms_from_kg_sparse():
    """
    Extract all symptoms from Sparse Knowledge Graph including duplicates
    
    Retrieves symptom nodes from the Sparse KG maintaining the 1:1:1 structure
    where each triplet has its own symptom instance. This preserves the original
    sparse structure for accurate lexical indexing.
    
    Returns:
        list: List of symptom dictionaries with triplet IDs and equipment metadata
    """
    print("Extracting symptoms from Sparse KG...")
    
    driver = get_neo4j_connection("sparse")
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Symptom)
                RETURN s.name AS symptom_text, 
                       s.equipment AS equipment,
                       s.triplet_id AS triplet_id
                ORDER BY s.triplet_id
            """)
            
            symptoms = []
            for record in result:
                symptoms.append({
                    'symptom_id': f"sparse_{record['triplet_id']}",
                    'symptom_text': record['symptom_text'],
                    'equipment': record['equipment'] or 'unknown'
                })
            
            print(f"Extracted {len(symptoms)} Sparse symptoms (including duplicates)")
            return symptoms
            
    except Exception as e:
        print(f"Error extracting Sparse: {e}")
        return []
    finally:
        driver.close()

def build_symptom_bm25_index_sparse():
    """
    Construct BM25 index for Sparse symptoms
    
    Creates a Whoosh-based BM25 lexical search index using individual symptom texts
    from the Sparse Knowledge Graph. The index maintains the 1:1:1 structure for
    accurate sparse similarity calculations in the hybrid metric system.
    
    Returns:
        str: Path to the created BM25 index directory
        
    Raises:
        ValueError: When no symptoms are found in the Sparse Knowledge Graph
    """
    print("Constructing Sparse symptom BM25 index...")
    
    # Configuration - corrected path
    config = load_settings()
    index_path = config["paths"]["bm25_sparse_index_path"]  # Corrected path
    
    # Directory creation
    os.makedirs(index_path, exist_ok=True)
    
    # Simple schema for sparse structure
    schema = Schema(
        symptom_id=ID(stored=True, unique=True),
        symptom_text=TEXT(analyzer=StandardAnalyzer(), stored=True),
        equipment=ID(stored=True)
    )
    
    # Remove existing index
    if exists_in(index_path):
        import shutil
        shutil.rmtree(index_path)
        os.makedirs(index_path, exist_ok=True)
    
    ix = create_in(index_path, schema)
    
    # Extraction and indexing
    symptoms = extract_symptoms_from_kg_sparse()
    if not symptoms:
        raise ValueError("No symptoms found in Sparse KG")
    
    print(f"Indexing {len(symptoms)} Sparse symptoms...")
    writer = ix.writer()
    
    for symptom in symptoms:
        writer.add_document(
            symptom_id=symptom['symptom_id'],
            symptom_text=symptom['symptom_text'],
            equipment=symptom['equipment']
        )
    
    writer.commit()
    
    print(f"Sparse BM25 index created: {index_path}")
    print(f"Indexed symptoms: {len(symptoms)} (1:1:1 structure)")
    
    return index_path

def main():
    """
    Main pipeline for Sparse BM25 index construction
    
    Orchestrates the complete process of extracting symptoms from the Sparse
    Knowledge Graph and constructing the corresponding BM25 lexical search index.
    Provides comprehensive error handling and progress reporting.
    """
    print("BM25 SYMPTOM INDEX CONSTRUCTION - SPARSE KG")
    print("=" * 55)
    
    try:
        index_path = build_symptom_bm25_index_sparse()
        print(f"\nCONSTRUCTION COMPLETED")
        print(f"Index created: {index_path}")
        print("Ready for Sparse hybrid search")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()