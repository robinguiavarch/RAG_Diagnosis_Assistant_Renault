"""
Sparse Knowledge Graph Visualization Module

This module provides visualization capabilities for Sparse Knowledge Graphs constructed
in the RAG diagnosis system. It creates optimized network visualizations with support
for cloud and local Neo4j deployments, featuring intelligent connection management,
statistical analysis, and customizable rendering options for sparse graph exploration
with emphasis on 1:1:1 structure preservation.

Key components:
- Cloud/local connection management: Intelligent fallback system for Neo4j connectivity
- Sparse-optimized visualization: Efficient rendering highlighting linear structures
- Equipment-aware filtering: Support for equipment-specific sparse graph exploration
- 1:1:1 structure analysis: Statistical validation of sparse graph characteristics

Dependencies: neo4j, networkx, matplotlib, pyyaml, python-dotenv, argparse
Usage: Execute as standalone script with customizable parameters for sparse graph exploration
"""

from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Avoid LaTeX parsing
import argparse
import os
from dotenv import load_dotenv
import random
import sys
from pathlib import Path
import yaml

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_settings():
    """
    Load configuration from settings.yaml file
    
    Returns:
        dict: Loaded configuration settings
    """
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_neo4j_connection(kg_type="sparse"):
    """
    Establish intelligent Cloud/Local Neo4j connection for sparse visualization
    
    Implements cloud-first connection strategy with automatic local fallback,
    matching the connection logic used in the Sparse KG construction pipeline.
    
    Args:
        kg_type (str): Knowledge Graph type ("dense", "sparse", or "dense_sc")
        
    Returns:
        tuple: (neo4j.Driver, user, password) for connection management
    """
    load_dotenv()
    
    # Priority to Cloud if enabled
    cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
    
    if cloud_enabled:
        print(f"CLOUD MODE for {kg_type.upper()}")
        
        if kg_type == "dense":
            uri = os.getenv("NEO4J_DENSE_CLOUD_URI")
            password = os.getenv("NEO4J_DENSE_CLOUD_PASS")
            user = "neo4j"
        elif kg_type == "sparse":
            uri = os.getenv("NEO4J_SPARSE_CLOUD_URI")
            password = os.getenv("NEO4J_SPARSE_CLOUD_PASS")
            user = "neo4j"
        elif kg_type == "dense_sc":
            uri = os.getenv("NEO4J_DENSE_SC_CLOUD_URI")
            password = os.getenv("NEO4J_DENSE_SC_CLOUD_PASS")
            user = "neo4j"
        
        if uri and password:
            print(f"Cloud connection {kg_type}: {uri}")
            return GraphDatabase.driver(uri, auth=(user, password)), user, password
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
    return GraphDatabase.driver(uri, auth=(user, password)), user, password

def plot_sparse_kg_graph(driver, limit=50, show_labels=False, equipment_filter=None):
    """
    Visualize a sample of the Sparse Knowledge Graph with linear structure emphasis
    
    Creates an efficient network visualization of the Sparse Knowledge Graph with
    emphasis on the 1:1:1 linear structure, equipment filtering capabilities, and
    visual distinction from dense graph representations for clear structural analysis.
    
    Args:
        driver: Neo4j database driver instance
        limit (int): Maximum number of triplets to display (default: 50)
        show_labels (bool): Display node names (default: False)
        equipment_filter (str, optional): Filter by specific equipment type
    """
    print("Executing Cypher query for sparse structure...")
    
    # Base query for linear triplet retrieval with optional equipment filtering
    base_query = """
        MATCH (s:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
    """
    
    if equipment_filter:
        base_query += " WHERE s.equipment CONTAINS $equipment_filter"
        
    base_query += """
        RETURN s.name AS s, c.name AS c, r.name AS r, 
               s.equipment AS equipment, s.triplet_id AS triplet_id
        ORDER BY rand()
        LIMIT $limit
    """

    with driver.session() as session:
        if equipment_filter:
            result = session.run(base_query, limit=limit, equipment_filter=equipment_filter)
        else:
            result = session.run(base_query, limit=limit)
        
        data = [record.data() for record in result]

    if not data:
        print("No data found in sparse graph")
        return

    print(f"Constructing sparse graph ({len(data)} triplets)...")
    G = nx.DiGraph()
    
    # Graph construction with node types and triplet metadata
    for row in data:
        # Add nodes with their type, equipment, and triplet information
        G.add_node(row["s"], node_type="Symptom", equipment=row.get("equipment", "unknown"), triplet_id=row.get("triplet_id"))
        G.add_node(row["c"], node_type="Cause", equipment=row.get("equipment", "unknown"), triplet_id=row.get("triplet_id"))
        G.add_node(row["r"], node_type="Remedy", equipment=row.get("equipment", "unknown"), triplet_id=row.get("triplet_id"))
        
        # Add linear relationships (characteristic of sparse structure)
        G.add_edge(row["s"], row["c"], label="CAUSES")
        G.add_edge(row["c"], row["r"], label="TREATED_BY")

    print(f"Sparse graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} relations")

    # Plot configuration optimized for sparse visualization
    plt.figure(figsize=(15, 12))
    
    print("Computing layout for sparse graph structure...")
    # Layout adapted to linear structures
    if G.number_of_nodes() > 100:
        pos = nx.random_layout(G)  # Faster option
    else:
        # Hierarchical layout to better show linear structure
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Colors by node type (distinct palette for sparse differentiation)
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'Unknown')
        if node_type == 'Symptom':
            node_colors.append('#E74C3C')  # Vivid red
        elif node_type == 'Cause':
            node_colors.append('#3498DB')  # Standard blue
        elif node_type == 'Remedy':
            node_colors.append('#2ECC71')  # Green
        else:
            node_colors.append('#F39C12')  # Orange
    
    print("Drawing sparse graph structure...")
    # Draw nodes with enhanced visibility for sparse structure
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=400,
                          alpha=0.9)
    
    # Draw edges with emphasis on linear structure
    nx.draw_networkx_edges(G, pos, 
                          edge_color='#34495E',
                          arrows=True,
                          arrowsize=20,
                          alpha=0.7,
                          width=2)
    
    # Optional labels optimized for sparse visibility
    if show_labels and G.number_of_nodes() <= 25:
        # Truncate long labels for clarity
        labels = {node: (node[:12] + '...' if len(node) > 12 else node) 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, 
                               labels=labels,
                               font_size=7,
                               font_weight='bold',
                               font_color='white',
                               bbox=dict(boxstyle="round,pad=0.2", 
                                       facecolor='black', 
                                       alpha=0.7))
    
    # Title specific to sparse graph characteristics
    title = f"Sparse Knowledge Base - Linear Structure ({len(data)} triplets)"
    if equipment_filter:
        title += f" - Equipment: {equipment_filter}"
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Color legend adapted for sparse representation
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', 
                   markersize=10, label='Symptoms'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', 
                   markersize=10, label='Causes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71', 
                   markersize=10, label='Remedies')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Annotation specific to sparse mode characteristics
    plt.text(0.02, 0.98, "Mode: SPARSE\n• Linear structure\n• 1:1:1 relations\n• No enrichment", 
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    
    print("Sparse graph display completed")
    print("\nSPARSE CHARACTERISTICS:")
    print(f"   • Linear structure: Symptom → Cause → Remedy")
    print(f"   • No semantic propagation")
    print(f"   • 1:1 relations from original CSV")
    print(f"   • Low density: {G.number_of_edges()}/{G.number_of_nodes()} = {G.number_of_edges()/G.number_of_nodes():.2f}")
    
    plt.show()

def get_sparse_graph_stats(driver, equipment_filter=None):
    """
    Display comprehensive statistics of the complete sparse graph
    
    Provides detailed analysis of the Sparse Knowledge Graph including node counts,
    relationship distribution, equipment analysis, and 1:1:1 structure validation
    for understanding the sparse graph characteristics and data fidelity.
    
    Args:
        driver: Neo4j database driver instance
        equipment_filter (str, optional): Filter statistics by specific equipment type
    """
    print("Retrieving sparse graph statistics...")
    
    with driver.session() as session:
        # Basic statistics query for sparse structure
        base_stats_query = """
            RETURN 
            count{(s:Symptom)} as symptoms,
            count{(c:Cause)} as causes, 
            count{(r:Remedy)} as remedies,
            count{()-[:CAUSES]->()} as causes_relations,
            count{()-[:TREATED_BY]->()} as treated_by_relations
        """
        
        if equipment_filter:
            base_stats_query = f"""
                MATCH (s:Symptom) WHERE s.equipment CONTAINS '{equipment_filter}'
                WITH collect(s) as filtered_symptoms
                UNWIND filtered_symptoms as s
                OPTIONAL MATCH (s)-[:CAUSES]->(c:Cause)
                OPTIONAL MATCH (c)-[:TREATED_BY]->(r:Remedy)
                RETURN 
                count(DISTINCT s) as symptoms,
                count(DISTINCT c) as causes,
                count(DISTINCT r) as remedies,
                count{{(s)-[:CAUSES]->(c)}} as causes_relations,
                count{{(c)-[:TREATED_BY]->(r)}} as treated_by_relations
            """
        
        result = session.run(base_stats_query)
        stats = result.single()
        
        print("\nCOMPREHENSIVE SPARSE GRAPH STATISTICS:")
        if equipment_filter:
            print(f"(Filtered by equipment: {equipment_filter})")
        
        print(f"   • Symptoms: {stats['symptoms']}")
        print(f"   • Causes: {stats['causes']}")
        print(f"   • Remedies: {stats['remedies']}")
        print(f"   • CAUSES relations: {stats['causes_relations']}")
        print(f"   • TREATED_BY relations: {stats['treated_by_relations']}")
        
        total_nodes = stats['symptoms'] + stats['causes'] + stats['remedies']
        total_relations = stats['causes_relations'] + stats['treated_by_relations']
        
        print(f"   • TOTAL nodes: {total_nodes}")
        print(f"   • TOTAL relations: {total_relations}")
        
        if total_nodes > 0:
            print(f"   • Density: {total_relations/total_nodes:.2f} relations/node")
        
        # Equipment distribution analysis for sparse structure
        if not equipment_filter:
            print("\nEQUIPMENT DISTRIBUTION:")
            eq_query = """
                MATCH (s:Symptom)
                WHERE s.equipment IS NOT NULL
                RETURN s.equipment as equipment, count(s) as symptom_count
                ORDER BY symptom_count DESC
            """
            
            eq_result = session.run(eq_query)
            for record in eq_result:
                equipment = record['equipment']
                count = record['symptom_count']
                print(f"   • {equipment}: {count} symptoms")
        
        # 1:1:1 structure validation for sparse KG
        print("\nSPARSE STRUCTURE VALIDATION:")
        
        # Validate 1:1:1 relationships
        ratio_query = """
            MATCH (s:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
            WITH s, count(c) as cause_count, count(r) as remedy_count
            RETURN 
            count(s) as total_triplets,
            avg(cause_count) as avg_causes_per_symptom,
            avg(remedy_count) as avg_remedies_per_symptom
        """
        
        if equipment_filter:
            ratio_query = f"""
                MATCH (s:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                WHERE s.equipment CONTAINS '{equipment_filter}'
                WITH s, count(c) as cause_count, count(r) as remedy_count
                RETURN 
                count(s) as total_triplets,
                avg(cause_count) as avg_causes_per_symptom,
                avg(remedy_count) as avg_remedies_per_symptom
            """
        
        ratio_result = session.run(ratio_query)
        ratios = ratio_result.single()
        
        if ratios and ratios['total_triplets'] > 0:
            print(f"   • Total linear triplets: {ratios['total_triplets']}")
            print(f"   • Average causes per symptom: {ratios['avg_causes_per_symptom']:.2f}")
            print(f"   • Average remedies per symptom: {ratios['avg_remedies_per_symptom']:.2f}")
            
            # Validate 1:1:1 structure
            if (abs(ratios['avg_causes_per_symptom'] - 1.0) < 0.1 and 
                abs(ratios['avg_remedies_per_symptom'] - 1.0) < 0.1):
                print("   • Structure: CONFIRMED 1:1:1 linear relationships")
            else:
                print("   • Structure: WARNING - Non-1:1:1 detected")
        else:
            print("   • No complete triplets found")
        
        # Triplet ID analysis for sparse traceability
        triplet_query = """
            MATCH (s:Symptom)
            WHERE s.triplet_id IS NOT NULL
            RETURN count(s) as symptoms_with_triplet_id
        """
        
        if equipment_filter:
            triplet_query = f"""
                MATCH (s:Symptom)
                WHERE s.triplet_id IS NOT NULL AND s.equipment CONTAINS '{equipment_filter}'
                RETURN count(s) as symptoms_with_triplet_id
            """
        
        triplet_result = session.run(triplet_query)
        triplet_count = triplet_result.single()["symptoms_with_triplet_id"]
        
        print(f"   • Symptoms with triplet_id: {triplet_count}")
        print("   • Structure: Linear (1 Symptom → 1 Cause → 1 Remedy)")

def main():
    """
    Main sparse visualization pipeline with intelligent connection management
    
    Orchestrates the complete sparse visualization process including connection establishment,
    statistical analysis, and graph rendering with support for various filtering
    and display options tailored to Sparse Knowledge Graph exploration and 1:1:1 structure analysis.
    """
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Sparse Knowledge Graph Visualization")
    parser.add_argument("--kg-type", choices=["dense", "sparse", "dense_sc"], default="sparse",
                       help="Knowledge Graph type to visualize")
    parser.add_argument("--limit", type=int, default=50,
                       help="Maximum number of triplets to display (default: 50)")
    parser.add_argument("--show-labels", action="store_true",
                       help="Display node names (only if <= 25 nodes)")
    parser.add_argument("--stats-only", action="store_true",
                       help="Display only statistics, without visualization")
    parser.add_argument("--equipment", type=str,
                       help="Filter by specific equipment type (e.g., 'FANUC')")
    
    args = parser.parse_args()

    # Establish connection using the same logic as Sparse KG construction
    try:
        driver, user, password = get_neo4j_connection(args.kg_type)
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 'Connected to Sparse KG!' as message")
            message = result.single()["message"]
            print(f"Connection successful: {message}")
        
        if args.stats_only:
            get_sparse_graph_stats(driver, args.equipment)
        else:
            # Display statistics first
            get_sparse_graph_stats(driver, args.equipment)
            
            # Then visualization
            plot_sparse_kg_graph(driver, args.limit, args.show_labels, args.equipment)
            
    except Exception as e:
        print(f"Connection or visualization error: {str(e)}")
        print("Verify Neo4j connection settings and credentials")
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    main()
    