"""
Dense Knowledge Graph Construction with Autonomous Hybrid Metric

This module constructs a Dense Knowledge Graph using an autonomous hybrid metric that combines
Cosine, Jaccard, and Levenshtein similarities. The system creates an enriched graph structure
where symptoms can be connected to multiple causes through semantic similarity propagation,
providing enhanced relationship modeling for the RAG diagnosis system.

Key components:
- Autonomous hybrid metric: Self-contained similarity calculation without external dependencies
- Dense graph construction: Creates enriched relationships through similarity propagation
- Equipment-aware modeling: Incorporates equipment metadata throughout the graph structure
- Cloud/local connectivity: Intelligent connection management for Neo4j deployments

Dependencies: neo4j, pandas, numpy, pyyaml, python-dotenv, sentence-transformers, scikit-learn
Usage: docker run --rm -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data 
       -v $(pwd)/config:/app/config -v $(pwd)/pipeline_step:/app/pipeline_step 
       --network host diagnosis-app poetry run python pipeline_step/knowledge_graph_setup/build_dense_knowledge_graph.py
"""

import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
import yaml

# Import autonomous hybrid metric
from hybrid_metric_build_kgs import create_autonomous_hybrid_metric

# Load environment variables from .env
load_dotenv()

def load_settings():
    """
    Load configuration from settings.yaml file
    
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

# Configuration parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "scr_triplets", "doc-R-30iB_scr_triplets.csv")

# Configuration from settings.yaml
config = load_settings()
SIM_THRESHOLD = config["graph_retrieval"]["dense_similarity_threshold"]
TOP_K = config["graph_retrieval"]["dense_top_k_similar"]

# Neo4j connection
driver = get_neo4j_connection("dense")

def load_and_clean_data():
    """
    Load and clean CSV data by removing duplicates and null values
    
    Performs comprehensive data validation and cleaning including duplicate removal,
    null value handling, and equipment metadata validation. Provides detailed
    statistics on data quality and equipment distribution.
    
    Returns:
        pd.DataFrame: Cleaned and validated dataset ready for graph construction
        
    Raises:
        ValueError: When required 'equipment' column is missing from the dataset
    """
    print("Loading CSV file...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Initial data: {len(df)} rows")
    print(f"Detected columns: {list(df.columns)}")
    
    # Verify equipment column exists
    if 'equipment' not in df.columns:
        print("ERROR: Missing 'equipment' column in CSV")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Required 'equipment' column")
    
    # Remove rows with missing values (including equipment)
    df.dropna(subset=["symptom", "cause", "remedy", "equipment"], inplace=True)
    print(f"After NaN removal: {len(df)} rows")
    
    # Remove exact duplicates
    df_before_dedup = len(df)
    df = df.drop_duplicates(subset=["symptom", "cause", "remedy", "equipment"], keep="first")
    duplicates_removed = df_before_dedup - len(df)
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Final data: {len(df)} rows")
    
    # Display unique equipment types
    unique_equipment = df['equipment'].unique()
    print(f"Unique equipment types found: {len(unique_equipment)}")
    for eq in sorted(unique_equipment):
        count = len(df[df['equipment'] == eq])
        print(f"   • {eq}: {count} triplets")
    
    return df

def clear_database(tx):
    """
    Completely clear the Neo4j database for fresh start
    
    Args:
        tx: Neo4j transaction object
    """
    tx.run("MATCH (n) DETACH DELETE n")

def insert_triplets(tx, s, c, r, equipment):
    """
    Insert a triplet with equipment as property on each node
    
    Creates or merges Symptom, Cause, and Remedy nodes with equipment metadata
    and establishes the appropriate relationships between them.
    
    Args:
        tx: Neo4j transaction object
        s (str): Symptom text
        c (str): Cause text
        r (str): Remedy text
        equipment (str): Equipment type identifier
    """
    tx.run("""
        MERGE (sym:Symptom {name: $s})
        SET sym.equipment = $equipment
        
        MERGE (cause:Cause {name: $c})
        SET cause.equipment = $equipment
        
        MERGE (rem:Remedy {name: $r})
        SET rem.equipment = $equipment
        
        MERGE (sym)-[:CAUSES]->(cause)
        MERGE (cause)-[:TREATED_BY]->(rem)
    """, s=s, c=c, r=r, equipment=equipment)

def compute_similarity_hybrid_autonomous(symptom_list):
    """
    Autonomous hybrid metric calculation (Cosine + Jaccard + Levenshtein)
    
    Utilizes the autonomous hybrid metric system that combines three similarity
    measures without external dependencies. Provides fallback to standard cosine
    similarity if the hybrid metric encounters issues.
    
    Args:
        symptom_list (list): List of unique symptom texts for similarity calculation
        
    Returns:
        np.ndarray: Similarity matrix with hybrid metric scores
    """
    print("Computing autonomous hybrid metric (Cosine + Jaccard + Levenshtein)...")
    
    try:
        # Weight configuration
        weights = {
            'cosine_alpha': 0.4,
            'jaccard_beta': 0.4, 
            'levenshtein_gamma': 0.2
        }
        
        print(f"Configured weights: {weights}")
        
        # Create hybrid metric
        metric = create_autonomous_hybrid_metric(weights)
        
        # Calculate similarity matrix
        sim_matrix = metric.compute_similarity_matrix(symptom_list)
        
        print(f"Autonomous hybrid similarity matrix calculated: {sim_matrix.shape}")
        return sim_matrix
        
    except Exception as e:
        print(f"Error in autonomous hybrid metric: {e}")
        print("Fallback to standard cosine similarity...")
        return compute_similarity_fallback(symptom_list)

def compute_similarity_fallback(symptom_list):
    """
    Fallback to cosine similarity if hybrid metric fails
    
    Args:
        symptom_list (list): List of symptom texts for similarity calculation
        
    Returns:
        np.ndarray: Cosine similarity matrix
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("Fallback: Standard cosine similarity...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode(symptom_list, show_progress_bar=True)
    sim_matrix = cosine_similarity(vectors)
    np.fill_diagonal(sim_matrix, 0)  # Avoid self-similarity
    return sim_matrix

def add_sim_relation(tx, s1, s2, similarity_score):
    """
    Add a SIMILAR_TO relationship with similarity score
    
    Args:
        tx: Neo4j transaction object
        s1 (str): First symptom name
        s2 (str): Second symptom name
        similarity_score (float): Calculated similarity score
    """
    tx.run("""
        MATCH (a:Symptom {name: $s1}), (b:Symptom {name: $s2})
        MERGE (a)-[:SIMILAR_TO {score: $score}]->(b)
    """, s1=s1, s2=s2, score=float(similarity_score))

def propagate_links(tx):
    """
    Propagate CAUSES and TREATED_BY relationships via similarity
    
    Uses SIMILAR_TO relationships to propagate causal and treatment relationships
    from similar symptoms, creating the dense structure characteristic of this KG type.
    
    Args:
        tx: Neo4j transaction object
    """
    tx.run("""
        MATCH (s1:Symptom)-[:SIMILAR_TO]->(s2:Symptom)
        OPTIONAL MATCH (s1)-[:CAUSES]->(c:Cause)
        OPTIONAL MATCH (c)-[:TREATED_BY]->(r:Remedy)
        FOREACH (_ IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s2)-[:CAUSES]->(c)
        )
        FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
            MERGE (c)-[:TREATED_BY]->(r)
        )
    """)

def clean_similar(tx):
    """
    Remove all temporary SIMILAR_TO relationships
    
    Args:
        tx: Neo4j transaction object
    """
    tx.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")

def print_graph_stats(tx):
    """
    Display statistics of the created graph with equipment information
    
    Provides comprehensive statistics including node counts, relationship counts,
    and equipment distribution across the dense knowledge graph.
    
    Args:
        tx: Neo4j transaction object
    """
    result = tx.run("""
        RETURN 
        count{(s:Symptom)} as symptoms,
        count{(c:Cause)} as causes, 
        count{(r:Remedy)} as remedies,
        count{()-[:CAUSES]->()} as causes_relations,
        count{()-[:TREATED_BY]->()} as treated_by_relations
    """)
    
    stats = result.single()
    print("\nDENSE GRAPH STATISTICS:")
    print(f"   • Symptoms: {stats['symptoms']}")
    print(f"   • Causes: {stats['causes']}")
    print(f"   • Remedies: {stats['remedies']}")
    print(f"   • CAUSES relations: {stats['causes_relations']}")
    print(f"   • TREATED_BY relations: {stats['treated_by_relations']}")
    
    # Equipment distribution statistics
    print("\nEQUIPMENT DISTRIBUTION:")
    eq_result = tx.run("""
        MATCH (s:Symptom)
        WHERE s.equipment IS NOT NULL
        RETURN s.equipment as equipment, count(s) as symptom_count
        ORDER BY symptom_count DESC
    """)
    
    for record in eq_result:
        equipment = record['equipment']
        count = record['symptom_count']
        print(f"   • {equipment}: {count} symptoms")

def main():
    """
    Main pipeline for import and enrichment with autonomous hybrid metric
    
    Orchestrates the complete Dense Knowledge Graph construction process including
    data loading, similarity calculation, graph creation, and relationship propagation.
    Utilizes the autonomous hybrid metric for enhanced semantic understanding without
    external dependencies.
    """
    print("DENSE NEO4J PIPELINE STARTUP - AUTONOMOUS HYBRID METRIC")
    print("=" * 75)
    print("NEW: Integrated hybrid metric (Cosine + Jaccard + Levenshtein)")
    print("No external dependencies - Direct cloud construction possible")
    print()
    
    try:
        # Data loading and cleaning
        df = load_and_clean_data()
        
        # Extract unique symptoms for similarity calculation
        symptoms = df["symptom"].unique().tolist()
        print(f"Unique symptoms identified: {len(symptoms)}")
        
        # Calculate autonomous hybrid similarity matrix
        sim_matrix = compute_similarity_hybrid_autonomous(symptoms)
        
        with driver.session() as session:
            print("\nStep 1 - Neo4j database cleanup...")
            session.write_transaction(clear_database)
            
            print("Step 2 - Triplet insertion with equipment...")
            for idx, row in df.iterrows():
                session.write_transaction(insert_triplets, 
                                        row["symptom"], 
                                        row["cause"], 
                                        row["remedy"],
                                        row["equipment"])
                if (idx + 1) % 1000 == 0:
                    print(f"   • {idx + 1}/{len(df)} triplets inserted...")
            
            print("Step 3 - Computing and adding SIMILAR_TO relations...")
            similar_relations_added = 0
            for i, s1 in enumerate(symptoms):
                # Get TOP_K most similar neighbors
                indices = np.argsort(-sim_matrix[i])[:TOP_K]
                for j in indices:
                    if sim_matrix[i][j] >= SIM_THRESHOLD:
                        session.write_transaction(add_sim_relation, 
                                                s1, 
                                                symptoms[j], 
                                                sim_matrix[i][j])
                        similar_relations_added += 1
                        
                if (i + 1) % 100 == 0:
                    print(f"   • {i + 1}/{len(symptoms)} symptoms processed...")
            
            print(f"   • SIMILAR_TO relations created: {similar_relations_added}")
            
            print("Step 4 - Link propagation...")
            session.write_transaction(propagate_links)
            
            print("Step 5 - SIMILAR_TO relation cleanup...")
            session.write_transaction(clean_similar)
            
            print("Step 6 - Statistics generation...")
            session.read_transaction(print_graph_stats)
        
        print("\nDENSE IMPORT WITH AUTONOMOUS HYBRID METRIC COMPLETED")
        print("Characteristics:")
        print("   • Hybrid metric: Cosine + Jaccard + Levenshtein")
        print("   • No external dependencies (BM25/FAISS indexes)")
        print("   • Direct cloud construction possible")
        print("   • Equipment properties on each node")
        print("   • Enriched semantic propagation")
        print()
        print("Connect to Neo4j Browser to explore")
        print("Test query: MATCH (s:Symptom) WHERE s.equipment CONTAINS 'FANUC' RETURN s LIMIT 5")
        
    except FileNotFoundError:
        print(f"ERROR: CSV file not found: {CSV_PATH}")
        print("   Verify that the file exists and the path is correct.")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    main()