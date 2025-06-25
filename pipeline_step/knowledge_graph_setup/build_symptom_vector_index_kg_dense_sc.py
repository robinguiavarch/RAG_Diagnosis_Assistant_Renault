"""
FAISS Vector Index Construction for Dense S&C Knowledge Graph

This module constructs a FAISS semantic search index specifically for the Dense Symptom & Cause
Knowledge Graph configuration. The index is built using combined symptom and cause texts,
providing enhanced semantic search capabilities for the hybrid metric system with enriched
contextual understanding.

Key components:
- Dense S&C vector extraction: Retrieval and embedding of combined symptom+cause texts
- FAISS index construction: High-performance semantic indexing with normalized embeddings
- Cloud/local connection management: Intelligent fallback system for database connectivity
- Metadata preservation: Comprehensive storage of indexing metadata and statistics

Dependencies: neo4j, faiss-cpu, sentence-transformers, numpy, pickle, pyyaml, python-dotenv
Usage: docker run --rm -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data 
       -v $(pwd)/config:/app/config -v $(pwd)/pipeline_step:/app/pipeline_step 
       --network host diagnosis-app poetry run python pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_dense_sc.py
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

def get_neo4j_connection(kg_type="dense_sc"):
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

def build_symptom_index_dense_sc():
    """
    Construct FAISS index for Dense S&C symptoms
    
    Creates a high-performance semantic search index using combined symptom and cause
    texts from the Dense S&C Knowledge Graph. The index utilizes normalized embeddings
    and Inner Product similarity for optimal search performance in the hybrid metric system.
    
    The function performs the following operations:
    1. Extracts symptoms with combined text from Dense S&C KG
    2. Generates normalized embeddings using SentenceTransformer
    3. Constructs FAISS IndexFlatIP for semantic similarity
    4. Saves index and comprehensive metadata
    
    Raises:
        ValueError: When no S&C symptoms are found in the Knowledge Graph
        Exception: For database connection or index construction errors
    """
    print("Constructing FAISS index for Dense S&C (Symptom + Cause)...")
    
    # Configuration
    config = load_settings()
    
    # Intelligent Cloud/Local connection
    driver = get_neo4j_connection("dense_sc")
    
    # Embedding model
    model_name = config["models"]["embedding_model"]
    print(f"Loading model: {model_name}")
    
    # Output path for Dense S&C embeddings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "symptom_embeddings_dense_s&c")

    try:
        # Connection test
        with driver.session() as test_session:
            test_session.run("RETURN 1")
        print("Neo4j Dense S&C connection successful")
        
        # S&C data extraction
        print("Extracting symptoms with combined text...")
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Symptom)
                WHERE s.combined_text IS NOT NULL
                RETURN DISTINCT s.name AS symptom_name, 
                       s.combined_text AS combined_text,
                       s.equipment AS equipment
                ORDER BY s.name
            """)
            
            symptoms_data = []
            for record in result:
                symptoms_data.append({
                    'symptom_name': record["symptom_name"],
                    'combined_text': record["combined_text"],
                    'equipment': record["equipment"] or 'unknown'
                })
        
        print(f"Extracted {len(symptoms_data)} S&C symptoms")
        
        if not symptoms_data:
            raise ValueError("No S&C symptoms found in the Knowledge Base")
        
        # Text extraction for embedding
        symptom_names = [s['symptom_name'] for s in symptoms_data]
        combined_texts = [s['combined_text'] for s in symptoms_data]
        
        print(f"Examples of combined texts:")
        for i, text in enumerate(combined_texts[:3]):
            print(f"   {i+1}. {text}")
        
        # S&C embedding generation
        print("Generating embeddings with combined texts...")
        model = SentenceTransformer(model_name)
        
        # Embedding combined texts (symptom + cause)
        embeddings = model.encode(
            combined_texts,  # Uses combined texts, not just symptoms
            show_progress_bar=True, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        print(f"S&C embeddings generated: {embeddings.shape}")
        
        # FAISS index construction
        print("Constructing FAISS S&C index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product for normalized embeddings
        index.add(embeddings.astype('float32'))
        
        print(f"FAISS S&C index created with {index.ntotal} vectors of dimension {dim}")
        
        # Save operations
        print(f"Saving to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # FAISS index save
        index_path = os.path.join(output_dir, "index.faiss")
        faiss.write_index(index, index_path)
        print(f"FAISS index saved: {index_path}")
        
        # Enhanced S&C metadata save
        metadata_path = os.path.join(output_dir, "symptom_embedding_dense_s&c.pkl")
        metadata = {
            'symptom_names': symptom_names,
            'combined_texts': combined_texts,  # Combined texts
            'symptoms_data': symptoms_data,    # Complete data
            'model_name': model_name,
            'embedding_dim': dim,
            'total_symptoms': len(symptom_names),
            'source': 'knowledge_base_dense_s&c',
            'indexing_method': 'symptom_plus_cause_combined',  # Indexing method
            'connection_mode': 'cloud' if os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true" else 'local'
        }
        
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"S&C metadata saved: {metadata_path}")
        
        # Final statistics
        unique_symptoms = len(set(symptom_names))
        unique_equipments = set(s['equipment'] for s in symptoms_data)
        
        print("\nDENSE S&C INDEX STATISTICS:")
        print(f"   Indexed symptoms: {len(symptom_names)}")
        print(f"   Unique symptoms: {unique_symptoms}")
        print(f"   Method: Symptom + Cause combined")
        print(f"   Embedding dimension: {dim}")
        print(f"   Model used: {model_name}")
        print(f"   Source: Knowledge Base Dense S&C")
        print(f"   Connection mode: {metadata['connection_mode'].upper()}")
        print(f"   FAISS index size: {os.path.getsize(index_path) / 1024 / 1024:.2f} MB")
        
        # Equipment coverage
        print(f"   Equipment types covered: {len(unique_equipments)}")
        for eq in sorted(unique_equipments):
            count = sum(1 for s in symptoms_data if s['equipment'] == eq)
            print(f"     - {eq}: {count} symptoms")
        
        print(f"\nGenerated files:")
        print(f"     - {index_path}")
        print(f"     - {metadata_path}")
        
        print("\nUSAGE:")
        print("   This index can now be used for enhanced vector search")
        print("   with symptom + cause context in the Dense S&C RAG pipeline.")
        print("   Search will be more contextual through semantic enrichment.")
        
    except Exception as e:
        print(f"Error during S&C index construction: {str(e)}")
        raise
    finally:
        driver.close()
        print("Neo4j connection closed")

def main():
    """
    Main pipeline for Dense S&C FAISS index construction
    
    Orchestrates the complete process of extracting combined symptom and cause texts
    from the Dense S&C Knowledge Graph and constructing the corresponding FAISS
    semantic search index. Provides comprehensive error handling and detailed
    progress reporting throughout the construction process.
    """
    print("DENSE S&C FAISS INDEX CONSTRUCTION STARTUP")
    print("=" * 70)
    print("Objective: Symptom + Cause combined vector index")
    print("Support: Automatic Cloud/Local connection")
    print("Output: data/knowledge_base/symptom_embeddings_dense_s&c/")
    print()
    
    try:
        build_symptom_index_dense_sc()
        print("\nDENSE S&C FAISS INDEX CONSTRUCTION COMPLETED")
        
    except FileNotFoundError as e:
        print(f"ERROR: Missing configuration file: {str(e)}")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}")

if __name__ == "__main__":
    main()