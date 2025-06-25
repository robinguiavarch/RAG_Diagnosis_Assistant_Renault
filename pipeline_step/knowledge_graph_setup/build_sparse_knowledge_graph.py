"""
Sparse Knowledge Graph Construction with Cloud/Local Support

This module constructs a Sparse Knowledge Graph that maintains a simple 1:1:1 structure
without densification. Each triplet (Symptom-Cause-Remedy) is preserved as distinct nodes
with direct relationships, maintaining the original data structure for baseline comparison
and specific use cases requiring exact correspondence to source data.

Key components:
- Sparse graph construction: Creates simple 1:1:1 relationships without similarity-based densification
- Equipment-aware modeling: Incorporates equipment metadata on all nodes with triplet traceability
- Cloud/local connectivity: Intelligent connection management for Neo4j deployments
- Data preservation: Maintains all duplicates except identical triplets for complete fidelity

Dependencies: neo4j, pandas, pyyaml, python-dotenv
Usage: docker run --rm -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data 
       -v $(pwd)/config:/app/config -v $(pwd)/pipeline_step:/app/pipeline_step 
       --network host diagnosis-app poetry run python pipeline_step/knowledge_graph_setup/build_sparse_knowledge_graph.py
"""

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import yaml

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

# Configuration parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "scr_triplets", "doc-R-30iB_scr_triplets.csv")

# Neo4j connection
driver = get_neo4j_connection("sparse")

def find_csv_file():
    """
    Locate CSV file by testing multiple potential paths
    
    Implements fallback path resolution to handle different deployment scenarios
    and directory structures commonly encountered in containerized environments.
    
    Returns:
        str: Path to the located CSV file
        
    Raises:
        FileNotFoundError: When CSV file cannot be found in any tested path
    """
    # Main path
    main_path = CSV_PATH
    
    # Alternative paths
    alt_paths = [
        os.path.join(script_dir, "..", "..", "data", "extract_scr", "doc-R-30iB_scr_triplets.csv"),
        os.path.join(script_dir, "..", "data", "extract_scr", "doc-R-30iB_scr_triplets.csv"),
        os.path.join(script_dir, "data", "extract_scr", "doc-R-30iB_scr_triplets.csv"),
        os.path.join("data", "extract_scr", "doc-R-30iB_scr_triplets.csv"),
        os.path.join("data", "knowledge_base", "scr_triplets", "doc-R-30iB_scr_triplets.csv")
    ]
    
    # Test main path
    if os.path.exists(main_path):
        return main_path
    
    # Test alternative paths
    print("Checking alternative paths...")
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            print(f"File found: {alt_path}")
            return alt_path
    
    # No file found
    raise FileNotFoundError(f"CSV file not found in any tested paths")

def load_and_clean_data():
    """
    Load and clean CSV data by removing duplicates and null values
    
    Performs comprehensive data validation and cleaning while preserving the sparse
    structure. Removes only exact duplicates to maintain data fidelity for the
    1:1:1 relationship model characteristic of sparse knowledge graphs.
    
    Returns:
        pd.DataFrame: Cleaned and validated dataset ready for sparse graph construction
        
    Raises:
        ValueError: When required 'equipment' column is missing from the dataset
        FileNotFoundError: When CSV file cannot be located
    """
    print("Loading CSV file...")
    
    try:
        csv_path = find_csv_file()
        df = pd.read_csv(csv_path)
        print(f"CSV file loaded: {csv_path}")
    except FileNotFoundError as e:
        print(f"ERROR: {str(e)}")
        raise
    
    print(f"Initial data: {len(df)} rows")
    print(f"Detected columns: {list(df.columns)}")
    
    # Verify equipment column exists
    if 'equipment' not in df.columns:
        print("ERROR: Missing 'equipment' column in CSV")
        raise ValueError("Required 'equipment' column")
    
    # Remove rows with missing values (including equipment)
    df.dropna(subset=["symptom", "cause", "remedy", "equipment"], inplace=True)
    print(f"After NaN removal: {len(df)} rows")
    
    # Remove exact duplicates (same symptom, cause, remedy, equipment)
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

def insert_triplets_sparse(tx, s, c, r, equipment, triplet_id):
    """
    Insert SCR triplet with equipment while preserving ALL nodes in sparse structure
    
    Creates individual nodes for each element of the triplet without any merging,
    maintaining the 1:1:1 structure characteristic of sparse knowledge graphs.
    Each node receives equipment metadata and a unique triplet ID for traceability.
    
    Args:
        tx: Neo4j transaction object
        s (str): Symptom text
        c (str): Cause text
        r (str): Remedy text
        equipment (str): Equipment type identifier
        triplet_id (int): Unique identifier for this triplet instance
    """
    tx.run("""
        CREATE (sym:Symptom {name: $s, equipment: $equipment, triplet_id: $triplet_id})
        CREATE (cause:Cause {name: $c, equipment: $equipment, triplet_id: $triplet_id})
        CREATE (rem:Remedy {name: $r, equipment: $equipment, triplet_id: $triplet_id})
        CREATE (sym)-[:CAUSES]->(cause)
        CREATE (cause)-[:TREATED_BY]->(rem)
    """, s=s, c=c, r=r, equipment=equipment, triplet_id=triplet_id)

def print_graph_stats(tx):
    """
    Display statistics of the sparse graph with equipment information
    
    Provides comprehensive statistics including node counts, relationship counts,
    and equipment distribution specific to the sparse knowledge graph structure.
    
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
    print("\nSPARSE GRAPH STATISTICS:")
    print(f"   • Symptoms: {stats['symptoms']}")
    print(f"   • Causes: {stats['causes']}")
    print(f"   • Remedies: {stats['remedies']}")
    print(f"   • CAUSES relations: {stats['causes_relations']}")
    print(f"   • TREATED_BY relations: {stats['treated_by_relations']}")
    print(f"   • TOTAL nodes: {stats['symptoms'] + stats['causes'] + stats['remedies']}")
    print(f"   • TOTAL relations: {stats['causes_relations'] + stats['treated_by_relations']}")
    
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

def check_orphaned_nodes(tx):
    """
    Check for orphaned nodes in the sparse graph
    
    Validates graph integrity by identifying nodes without any relationships,
    which should not occur in a properly constructed sparse knowledge graph.
    
    Args:
        tx: Neo4j transaction object
    """
    result = tx.run("""
        MATCH (n)
        WHERE NOT (n)--()
        RETURN count(n) as orphaned_nodes
    """)
    
    orphaned = result.single()["orphaned_nodes"]
    if orphaned > 0:
        print(f"   Warning: Orphaned nodes detected: {orphaned}")
    else:
        print(f"   No orphaned nodes found")

def main():
    """
    Main pipeline for creating a SPARSE Knowledge Base with equipment and cloud support
    
    Orchestrates the complete Sparse Knowledge Graph construction process including
    data loading, database clearing, triplet insertion, and validation. Maintains
    the 1:1:1 structure without any densification or similarity-based processing.
    """
    print("SPARSE NEO4J PIPELINE STARTUP WITH EQUIPMENT AND CLOUD SUPPORT")
    print("=" * 70)
    print("Mode: SPARSE Knowledge Base (simple 1:1:1 structure)")
    print("Support: Automatic Cloud/Local connection")
    print("Equipment properties on each node")
    print()
    
    try:
        # Connection test
        print("Testing connection...")
        with driver.session() as session:
            result = session.run("RETURN 'Connected to Sparse KG!' as message")
            message = result.single()["message"]
            print(f"Connection successful: {message}")
        
        # Data loading and cleaning
        df = load_and_clean_data()
        
        with driver.session() as session:
            print("\nStep 1 - Neo4j Sparse database cleanup...")
            session.execute_write(clear_database)
            
            print("Step 2 - SCR triplet insertion with equipment...")
            for idx, row in df.iterrows():
                session.execute_write(insert_triplets_sparse, 
                                    row["symptom"], 
                                    row["cause"], 
                                    row["remedy"],
                                    row["equipment"],
                                    idx)  # Unique ID for traceability
                if (idx + 1) % 1000 == 0:
                    print(f"   • {idx + 1}/{len(df)} triplets inserted...")
            
            print("Step 3 - Statistics generation...")
            session.execute_read(print_graph_stats)
            session.execute_read(check_orphaned_nodes)
        
        print("\nSPARSE KNOWLEDGE BASE WITH EQUIPMENT CREATION COMPLETED")
        print("Characteristics:")
        print("   • Linear structure: 1 Symptom → 1 Cause → 1 Remedy")
        print("   • Equipment property on each node")
        print("   • NO densification (no hybrid metric)")
        print("   • NO cause/remedy deduplication")
        print("   • Total duplicate preservation (except identical triplets)")
        print("   • Strict 1:1:1 relationships from original CSV")
        print("   • Perfect traceability via triplet_id")
        print("   • Automatic Cloud/Local compatibility")
        print()
        print("Connect to Neo4j Browser to explore")
        print("Compare with Dense Knowledge Bases")
        print()
        print("Equipment test query:")
        print("   MATCH (s:Symptom) WHERE s.equipment CONTAINS 'FANUC' RETURN s LIMIT 5")
        
    except FileNotFoundError:
        csv_path = find_csv_file() if 'find_csv_file' in locals() else CSV_PATH
        print(f"ERROR: CSV file not found: {csv_path}")
        print("   Verify that the file exists and the path is correct.")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    main()