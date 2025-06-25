"""
FAISS Vector Index Construction for Sparse Knowledge Graph

This module constructs a FAISS semantic search index specifically for the Sparse Knowledge Graph
configuration. The index is built using individual symptom texts while preserving the 1:1:1
structure and intentional duplicates, providing semantic search capabilities for the hybrid
metric system in sparse configurations.

Key components:
- Sparse symptom extraction: Retrieval of all symptoms including duplicates from Neo4j
- FAISS index construction: High-performance semantic indexing with normalized embeddings
- Cloud/local connection management: Intelligent fallback system for database connectivity
- Metadata preservation: Comprehensive storage of triplet IDs and equipment information

Dependencies: neo4j, faiss-cpu, sentence-transformers, numpy, pickle, pyyaml, python-dotenv
Usage: docker run --rm -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data 
       -v $(pwd)/config:/app/config -v $(pwd)/pipeline_step:/app/pipeline_step 
       --network host diagnosis-app poetry run python pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_sparse.py
"""

import os
import pickle
import faiss
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import yaml

load_dotenv()

def load_settings():
    """
    Load system configuration from settings.yaml file
    
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

def build_symptom_index_sparse():
    """
    Construct FAISS index for Sparse Knowledge Base symptoms
    
    Creates a high-performance semantic search index using individual symptom texts
    from the Sparse Knowledge Graph. The index preserves the 1:1:1 structure by
    maintaining intentional duplicates, ensuring accurate representation of the
    sparse configuration for hybrid metric calculations.
    
    The function performs the following operations:
    1. Extracts all symptoms including duplicates from Sparse KG
    2. Generates normalized embeddings using SentenceTransformer
    3. Constructs FAISS IndexFlatIP for semantic similarity
    4. Saves index and comprehensive metadata with triplet information
    
    Raises:
        ValueError: When no symptoms are found in the Sparse Knowledge Graph
        Exception: For database connection or index construction errors
    """
    print("Constructing FAISS index for Sparse Knowledge Base symptoms...")
    
    # Configuration
    config = load_settings()
    
    # Intelligent Cloud/Local connection
    driver = get_neo4j_connection("sparse")
    
    # Embedding model
    model_name = config["models"]["embedding_model"]
    print(f"Loading model: {model_name}")
    
    # Output path for Sparse embeddings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "symptom_embeddings_sparse")

    try:
        # Connection test
        with driver.session() as test_session:
            test_session.run("RETURN 1")
        print("Neo4j Sparse connection successful")
        
        # Sparse Knowledge Base symptom extraction
        print("Extracting symptoms from Sparse Knowledge Base...")
        with driver.session() as session:
            # Retrieve all symptoms (with duplicates for Sparse)
            result = session.run("""
                MATCH (s:Symptom) 
                RETURN s.name AS name, s.equipment AS equipment, s.triplet_id AS triplet_id
                ORDER BY s.triplet_id
            """)
            
            symptoms_data = []
            for record in result:
                symptoms_data.append({
                    'name': record["name"],
                    'equipment': record["equipment"] or 'unknown',
                    'triplet_id': record["triplet_id"]
                })
        
        print(f"Extracted {len(symptoms_data)} symptoms from Sparse Knowledge Base")
        
        if not symptoms_data:
            raise ValueError("No symptoms found in Sparse Knowledge Base")
        
        # Name extraction for embedding
        symptom_names = [s['name'] for s in symptoms_data]
        
        # Embedding generation
        print("Generating embeddings with SentenceTransformer...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            symptom_names, 
            show_progress_bar=True, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        print(f"Embeddings generated: {embeddings.shape}")
        
        # FAISS index construction
        print("Constructing FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product for normalized embeddings
        index.add(embeddings.astype('float32'))
        
        print(f"FAISS index created with {index.ntotal} vectors of dimension {dim}")
        
        # Save operations
        print(f"Saving to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # FAISS index save
        index_path = os.path.join(output_dir, "index.faiss")
        faiss.write_index(index, index_path)
        print(f"FAISS index saved: {index_path}")
        
        # Enhanced metadata save for Sparse
        metadata_path = os.path.join(output_dir, "symptom_embedding_sparse.pkl")
        metadata = {
            'symptom_names': symptom_names,
            'symptoms_data': symptoms_data,  # Complete data with equipment and triplet_id
            'model_name': model_name,
            'embedding_dim': dim,
            'total_symptoms': len(symptom_names),
            'source': 'knowledge_base_sparse',
            'connection_mode': 'cloud' if os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true" else 'local'
        }
        
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved: {metadata_path}")
        
        # Final statistics
        unique_symptoms = len(set(symptom_names))
        unique_equipments = set(s['equipment'] for s in symptoms_data)
        
        print("\nSPARSE INDEX STATISTICS:")
        print(f"   Total indexed symptoms: {len(symptom_names)}")
        print(f"   Unique symptoms: {unique_symptoms}")
        print(f"   Preserved duplicates: {len(symptom_names) - unique_symptoms}")
        print(f"   Embedding dimension: {dim}")
        print(f"   Model used: {model_name}")
        print(f"   Source: Sparse Knowledge Base")
        print(f"   Connection mode: {metadata['connection_mode'].upper()}")
        print(f"   FAISS index size: {os.path.getsize(index_path) / 1024 / 1024:.2f} MB")
        
        # Unique equipment types
        print(f"   Equipment types covered: {len(unique_equipments)}")
        for eq in sorted(unique_equipments):
            count = sum(1 for s in symptoms_data if s['equipment'] == eq)
            print(f"     - {eq}: {count} symptoms")
        
        print(f"\nGenerated files:")
        print(f"     - {index_path}")
        print(f"     - {metadata_path}")
        
        print("\nUSAGE:")
        print("   This index can now be used for vector search")
        print("   of symptoms in the RAG pipeline with Sparse Knowledge Graph.")
        print("   1:1:1 structure preserved with intentional duplicates.")
        
    except Exception as e:
        print(f"Error during Sparse index construction: {str(e)}")
        raise
    finally:
        driver.close()
        print("Neo4j connection closed")

def main():
    """
    Main pipeline for Sparse FAISS index construction
    
    Orchestrates the complete process of extracting symptoms from the Sparse
    Knowledge Graph and constructing the corresponding FAISS semantic search index.
    Provides comprehensive error handling and detailed progress reporting while
    preserving the 1:1:1 structure characteristic of sparse configurations.
    """
    print("SPARSE FAISS INDEX CONSTRUCTION STARTUP")
    print("=" * 60)
    print("Objective: Create vector index for Sparse KB symptoms")
    print("Support: Automatic Cloud/Local connection")
    print("Output: data/knowledge_base/symptom_embeddings_sparse/")
    print()
    
    try:
        build_symptom_index_sparse()
        print("\nSPARSE FAISS INDEX CONSTRUCTION COMPLETED SUCCESSFULLY")
        
    except FileNotFoundError as e:
        print(f"ERROR: Missing configuration file: {str(e)}")
        print("   Verify that config/settings.yaml exists.")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}")

if __name__ == "__main__":
    main()