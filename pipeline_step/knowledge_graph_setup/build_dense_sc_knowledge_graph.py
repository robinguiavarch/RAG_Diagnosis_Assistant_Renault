"""
Dense S&C Knowledge Graph Construction with Autonomous Hybrid Metric

This module constructs a Dense Symptom & Cause Knowledge Graph using an autonomous hybrid 
metric that operates on combined symptom and cause texts. The system creates enriched 
relationship modeling through similarity-based densification using concatenated symptom 
and cause information, providing enhanced contextual understanding for the RAG diagnosis system.

Key components:
- Combined text processing: Creates symptom+cause concatenated texts for enhanced similarity
- Autonomous hybrid metric: Self-contained similarity calculation without external dependencies  
- Dense S&C graph construction: Creates enriched relationships through combined text similarity propagation
- Equipment-aware modeling: Incorporates equipment metadata with combined text properties

Dependencies: neo4j, pandas, numpy, pyyaml, python-dotenv, sentence-transformers, scikit-learn
Usage: docker run --rm -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data 
       -v $(pwd)/config:/app/config -v $(pwd)/pipeline_step:/app/pipeline_step 
       --network host diagnosis-app poetry run python pipeline_step/knowledge_graph_setup/build_dense_sc_knowledge_graph.py
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

# Configuration parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "scr_triplets", "doc-R-30iB_scr_triplets.csv")

# Configuration from settings.yaml
config = load_settings()
SIM_THRESHOLD = config["graph_retrieval"]["dense_similarity_threshold"]
TOP_K = config["graph_retrieval"]["dense_top_k_similar"]

# Neo4j connection
driver = get_neo4j_connection("dense_sc")

def load_and_clean_data():
    """
    Load and clean CSV data with combined symptom and cause text creation
    
    Performs comprehensive data validation and cleaning while creating combined
    symptom+cause texts for enhanced similarity calculations. This approach
    enables densification based on both symptom and cause information together.
    
    Returns:
        pd.DataFrame: Cleaned and validated dataset with combined text column
        
    Raises:
        ValueError: When required 'equipment' column is missing from the dataset
    """
    print("Loading CSV file...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Initial data: {len(df)} rows")
    print(f"Detected columns: {list(df.columns)}")
    
    # Equipment verification
    if 'equipment' not in df.columns:
        print("ERROR: Missing 'equipment' column")
        raise ValueError("Required 'equipment' column")
    
    # NaN removal
    df.dropna(subset=["symptom", "cause", "remedy", "equipment"], inplace=True)
    print(f"After NaN removal: {len(df)} rows")
    
    # Duplicate removal
    df_before_dedup = len(df)
    df = df.drop_duplicates(subset=["symptom", "cause", "remedy", "equipment"], keep="first")
    duplicates_removed = df_before_dedup - len(df)
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Final data: {len(df)} rows")
    
    # Create combined symptom + cause text
    df['symptom_cause_combined'] = df['symptom'] + " " + df['cause']
    print(f"Combined text created: symptom + cause")
    
    # Equipment display
    unique_equipment = df['equipment'].unique()
    print(f"Unique equipment types: {len(unique_equipment)}")
    for eq in sorted(unique_equipment):
        count = len(df[df['equipment'] == eq])
        print(f"   • {eq}: {count} triplets")
    
    # Examples of combined texts
    print("\nExamples of combined S&C texts:")
    for i, combined in enumerate(df['symptom_cause_combined'].head(3)):
        print(f"   {i+1}. {combined}")
    
    return df

def clear_database(tx):
    """
    Completely clear the Neo4j database for fresh start
    
    Args:
        tx: Neo4j transaction object
    """
    tx.run("MATCH (n) DETACH DELETE n")

def insert_triplets_sc(tx, s, c, r, equipment, combined_text):
    """
    Insert triplet with equipment and combined text as properties
    
    Creates or merges nodes with equipment metadata and stores the combined
    symptom+cause text as a property on symptom nodes for enhanced search
    and similarity capabilities.
    
    Args:
        tx: Neo4j transaction object
        s (str): Symptom text
        c (str): Cause text
        r (str): Remedy text
        equipment (str): Equipment type identifier
        combined_text (str): Combined symptom and cause text
    """
    tx.run("""
        MERGE (sym:Symptom {name: $s})
        SET sym.equipment = $equipment, sym.combined_text = $combined_text
        
        MERGE (cause:Cause {name: $c})
        SET cause.equipment = $equipment
        
        MERGE (rem:Remedy {name: $r})
        SET rem.equipment = $equipment
        
        MERGE (sym)-[:CAUSES]->(cause)
        MERGE (cause)-[:TREATED_BY]->(rem)
    """, s=s, c=c, r=r, equipment=equipment, combined_text=combined_text)

def compute_similarity_sc_hybrid_autonomous(combined_texts, symptoms):
    """
    Autonomous hybrid metric for Dense S&C (Cosine + Jaccard + Levenshtein)
    
    Calculates similarity matrix using combined symptom and cause texts to enable
    densification based on both symptom and cause information. Creates mapping
    between combined texts and corresponding symptom names for relationship creation.
    
    Args:
        combined_texts (list): List of combined symptom+cause texts
        symptoms (list): List of corresponding symptom names
        
    Returns:
        tuple: (similarity_matrix, combined_to_symptom_mapping)
    """
    print("Computing autonomous hybrid metric for S&C (symptom + cause)...")
    
    try:
        # S&C specific weight configuration
        weights = {
            'cosine_alpha': 0.4,
            'jaccard_beta': 0.4,  # More important for combined texts
            'levenshtein_gamma': 0.2
        }
        
        print(f"S&C configured weights: {weights}")
        
        # Create hybrid metric
        metric = create_autonomous_hybrid_metric(weights)
        
        # Calculate similarity matrix on combined texts
        sim_matrix = metric.compute_similarity_matrix(combined_texts)
        
        print(f"Hybrid S&C similarity matrix calculated: {sim_matrix.shape}")
        
        # Create mapping combined_text → symptom
        combined_to_symptom = {}
        for i, combined_text in enumerate(combined_texts):
            combined_to_symptom[combined_text] = symptoms[i]
        
        print(f"Mapping created: {len(combined_to_symptom)} S&C → symptom relations")
        
        return sim_matrix, combined_to_symptom
        
    except Exception as e:
        print(f"Error in autonomous hybrid metric for S&C: {e}")
        print("Fallback to standard cosine similarity...")
        return compute_similarity_sc_fallback(combined_texts, symptoms)

def compute_similarity_sc_fallback(combined_texts, symptoms):
    """
    Fallback to cosine similarity for S&C if hybrid metric fails
    
    Args:
        combined_texts (list): List of combined symptom+cause texts
        symptoms (list): List of corresponding symptom names
        
    Returns:
        tuple: (similarity_matrix, combined_to_symptom_mapping)
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("S&C Fallback: Standard cosine similarity...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode(combined_texts, show_progress_bar=True)
    sim_matrix = cosine_similarity(vectors)
    np.fill_diagonal(sim_matrix, 0)  # Avoid self-similarity
    
    # Simple mapping
    combined_to_symptom = {}
    for i, combined_text in enumerate(combined_texts):
        combined_to_symptom[combined_text] = symptoms[i]
    
    return sim_matrix, combined_to_symptom

def add_sim_relation_sc(tx, s1, s2, similarity_score):
    """
    Add SIMILAR_TO_SC relationship with similarity score
    
    Args:
        tx: Neo4j transaction object
        s1 (str): First symptom name
        s2 (str): Second symptom name
        similarity_score (float): Calculated similarity score
    """
    tx.run("""
        MATCH (a:Symptom {name: $s1}), (b:Symptom {name: $s2})
        MERGE (a)-[:SIMILAR_TO_SC {score: $score}]->(b)
    """, s1=s1, s2=s2, score=float(similarity_score))

def propagate_links_sc(tx):
    """
    Propagate relationships via S&C similarity
    
    Uses SIMILAR_TO_SC relationships to propagate causal and treatment relationships
    from similar symptom+cause combinations, creating the dense structure.
    
    Args:
        tx: Neo4j transaction object
    """
    tx.run("""
        MATCH (s1:Symptom)-[:SIMILAR_TO_SC]->(s2:Symptom)
        OPTIONAL MATCH (s1)-[:CAUSES]->(c:Cause)
        OPTIONAL MATCH (c)-[:TREATED_BY]->(r:Remedy)
        FOREACH (_ IN CASE WHEN c IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s2)-[:CAUSES]->(c)
        )
        FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
            MERGE (c)-[:TREATED_BY]->(r)
        )
    """)

def clean_similar_sc(tx):
    """
    Remove all temporary SIMILAR_TO_SC relationships
    
    Args:
        tx: Neo4j transaction object
    """
    tx.run("MATCH ()-[r:SIMILAR_TO_SC]->() DELETE r")

def print_graph_stats_sc(tx):
    """
    Display statistics of the Dense S&C graph
    
    Provides comprehensive statistics including node counts, relationship counts,
    equipment distribution, and S&C specific metrics for the dense knowledge graph.
    
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
    print("\nDENSE S&C GRAPH STATISTICS:")
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
    
    # S&C specific statistics
    print("\nSYMPTOM + CAUSE STATISTICS:")
    sc_result = tx.run("""
        MATCH (s:Symptom)
        WHERE s.combined_text IS NOT NULL
        RETURN count(s) as symptoms_with_combined
    """)
    
    sc_count = sc_result.single()["symptoms_with_combined"]
    print(f"   • Symptoms with combined text: {sc_count}")

def main():
    """
    Main Dense S&C pipeline with autonomous hybrid metric
    
    Orchestrates the complete Dense S&C Knowledge Graph construction process including
    data loading with combined text creation, similarity calculation on symptom+cause
    pairs, graph creation, and relationship propagation. Utilizes autonomous hybrid
    metric for enhanced contextual understanding without external dependencies.
    """
    print("DENSE S&C NEO4J PIPELINE STARTUP - AUTONOMOUS HYBRID METRIC")
    print("=" * 80)
    print("NEW: Densification based on combined Symptom + Cause")
    print("Integrated hybrid metric (Cosine + Jaccard + Levenshtein)")
    print("No external dependencies - Direct cloud construction possible")
    print()
    
    try:
        # Loading and cleaning
        df = load_and_clean_data()
        
        # Extract unique combined texts and corresponding symptoms
        combined_texts = df["symptom_cause_combined"].unique().tolist()
        symptoms = df["symptom"].unique().tolist()
        print(f"Unique S&C texts identified: {len(combined_texts)}")
        print(f"Unique symptoms: {len(symptoms)}")
        
        # Calculate S&C hybrid autonomous similarity matrix
        sim_matrix, combined_to_symptom = compute_similarity_sc_hybrid_autonomous(combined_texts, symptoms)
        
        with driver.session() as session:
            print("\nStep 1 - Dense S&C Neo4j database cleanup...")
            session.write_transaction(clear_database)
            
            print("Step 2 - Triplet insertion with S&C...")
            for idx, row in df.iterrows():
                session.write_transaction(insert_triplets_sc, 
                                        row["symptom"], 
                                        row["cause"], 
                                        row["remedy"],
                                        row["equipment"],
                                        row["symptom_cause_combined"])
                if (idx + 1) % 1000 == 0:
                    print(f"   • {idx + 1}/{len(df)} triplets inserted...")
            
            print("Step 3 - Computing SIMILAR_TO_SC relations...")
            similar_relations_added = 0
            
            # Use mapping in relations loop
            for i, combined1 in enumerate(combined_texts):
                # Get TOP_K most similar neighbors
                indices = np.argsort(-sim_matrix[i])[:TOP_K]
                for j in indices:
                    if sim_matrix[i][j] >= SIM_THRESHOLD:
                        combined2 = combined_texts[j]
                        # Use mapping combined_text → symptom
                        symptom1 = combined_to_symptom[combined1]
                        symptom2 = combined_to_symptom[combined2]
                        
                        session.write_transaction(add_sim_relation_sc, 
                                                symptom1, 
                                                symptom2, 
                                                sim_matrix[i][j])
                        similar_relations_added += 1
                        
                if (i + 1) % 100 == 0:
                    print(f"   • {i + 1}/{len(combined_texts)} S&C texts processed...")
            
            print(f"   • SIMILAR_TO_SC relations created: {similar_relations_added}")
            
            print("Step 4 - Link propagation...")
            session.write_transaction(propagate_links_sc)
            
            print("Step 5 - Temporary relation cleanup...")
            session.write_transaction(clean_similar_sc)
            
            print("Step 6 - Statistics generation...")
            session.read_transaction(print_graph_stats_sc)
        
        print("\nDENSE S&C KNOWLEDGE BASE WITH AUTONOMOUS HYBRID METRIC CREATION COMPLETED")
        print("Characteristics:")
        print("   • Densification based on combined symptom + cause")
        print("   • Autonomous hybrid metric: Cosine + Jaccard + Levenshtein")
        print("   • No external dependencies (BM25/FAISS indexes)")
        print("   • Direct cloud construction possible")
        print("   • Equipment properties preserved")
        print("   • Enriched semantic propagation")
        print()
        print("Connect to Neo4j Browser to explore")
        print("Compare with standard Dense and Sparse versions")
        print()
        print("S&C test query:")
        print("   MATCH (s:Symptom) WHERE s.combined_text CONTAINS 'motor' RETURN s LIMIT 5")
        
    except FileNotFoundError:
        print(f"ERROR: CSV file not found: {CSV_PATH}")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    main()