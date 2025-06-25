"""
Dense Knowledge Graph Visualization Module

This module provides visualization capabilities for Dense Knowledge Graphs constructed
in the RAG diagnosis system. It creates optimized network visualizations with support
for cloud and local Neo4j deployments, featuring intelligent connection management,
statistical analysis, and customizable rendering options for graph exploration.

Key components:
- Cloud/local connection management: Intelligent fallback system for Neo4j connectivity
- Optimized graph visualization: Efficient rendering for large knowledge graphs
- Statistical analysis: Comprehensive graph metrics and distribution analysis
- Equipment-aware filtering: Support for equipment-specific graph exploration

Dependencies: neo4j, networkx, matplotlib, pyyaml, python-dotenv, argparse
Usage: Execute as standalone script with customizable parameters for graph exploration
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

def get_neo4j_connection(kg_type="dense"):
    """
    Establish intelligent Cloud/Local Neo4j connection for visualization
    
    Implements cloud-first connection strategy with automatic local fallback,
    matching the connection logic used in the Dense KG construction pipeline.
    
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

def plot_neo4j_graph_optimized(driver, limit=50, show_labels=False, equipment_filter=None):
    """
    Visualize a sample of the Neo4j Knowledge Graph in an optimized manner
    
    Creates an efficient network visualization of the Dense Knowledge Graph with
    support for equipment filtering, node type differentiation, and customizable
    display options for effective graph exploration and analysis.
    
    Args:
        driver: Neo4j database driver instance
        limit (int): Maximum number of triplets to display (default: 50)
        show_labels (bool): Display node names (default: False)
        equipment_filter (str, optional): Filter by specific equipment type
    """
    print("Executing Cypher query...")
    
    # Base query with optional equipment filtering
    base_query = """
        MATCH (s:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
    """
    
    if equipment_filter:
        base_query += " WHERE s.equipment CONTAINS $equipment_filter"
        
    base_query += """
        RETURN s.name AS s, c.name AS c, r.name AS r, s.equipment AS equipment
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
        print("No data found in the graph")
        return

    print(f"Constructing graph ({len(data)} triplets)...")
    G = nx.DiGraph()
    
    # Graph construction with node types and equipment metadata
    for row in data:
        # Add nodes with their type and equipment information
        G.add_node(row["s"], node_type="Symptom", equipment=row.get("equipment", "unknown"))
        G.add_node(row["c"], node_type="Cause", equipment=row.get("equipment", "unknown"))
        G.add_node(row["r"], node_type="Remedy", equipment=row.get("equipment", "unknown"))
        
        # Add relationships
        G.add_edge(row["s"], row["c"], label="CAUSES")
        G.add_edge(row["c"], row["r"], label="TREATED_BY")

    print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} relations")

    # Plot configuration
    plt.figure(figsize=(15, 12))
    
    print("Computing layout (optimized)...")
    # Faster layout for large graphs
    if G.number_of_nodes() > 100:
        pos = nx.random_layout(G)  # Faster option
    else:
        pos = nx.spring_layout(G, k=1, iterations=30)  # Reduced iterations
    
    # Colors by node type
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'Unknown')
        if node_type == 'Symptom':
            node_colors.append('#FF6B6B')  # Light red
        elif node_type == 'Cause':
            node_colors.append('#4ECDC4')  # Teal
        elif node_type == 'Remedy':
            node_colors.append('#45B7D1')  # Blue
        else:
            node_colors.append('#FFA07A')  # Orange
    
    print("Drawing graph...")
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=300,
                          alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=15,
                          alpha=0.6,
                          width=1.5)
    
    # Optional labels (only if requested and few nodes)
    if show_labels and G.number_of_nodes() <= 30:
        # Truncate long labels
        labels = {node: (node[:15] + '...' if len(node) > 15 else node) 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, 
                               labels=labels,
                               font_size=8,
                               font_weight='bold')
    
    # Title and legend
    title = f"Dense Knowledge Graph SCR - Sample ({len(data)} triplets)"
    if equipment_filter:
        title += f" - Equipment: {equipment_filter}"
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Color legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
                   markersize=10, label='Symptoms'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', 
                   markersize=10, label='Causes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1', 
                   markersize=10, label='Remedies')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    
    print("Display completed")
    print("\nTIPS:")
    print(f"   • Increase --limit to see more data (current: {limit})")
    print(f"   • Use --show-labels to display names (if <= 30 nodes)")
    print(f"   • Use --equipment to filter by equipment type")
    print(f"   • Close window to terminate script")
    
    plt.show()

def get_graph_stats(driver, equipment_filter=None):
    """
    Display comprehensive statistics of the complete graph
    
    Provides detailed analysis of the Dense Knowledge Graph including node counts,
    relationship distribution, equipment analysis, and density metrics for
    understanding the graph structure and content distribution.
    
    Args:
        driver: Neo4j database driver instance
        equipment_filter (str, optional): Filter statistics by specific equipment type
    """
    print("Retrieving graph statistics...")
    
    with driver.session() as session:
        # Basic statistics query
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
        
        print("\nCOMPREHENSIVE GRAPH STATISTICS:")
        if equipment_filter:
            print(f"(Filtered by equipment: {equipment_filter})")
        
        print(f"   • Symptoms: {stats['symptoms']}")
        print(f"   • Causes: {stats['causes']}")
        print(f"   • Remedies: {stats['remedies']}")
        print(f"   • CAUSES relations: {stats['causes_relations']}")
        print(f"   • TREATED_BY relations: {stats['treated_by_relations']}")
        print(f"   • TOTAL nodes: {stats['symptoms'] + stats['causes'] + stats['remedies']}")
        print(f"   • TOTAL relations: {stats['causes_relations'] + stats['treated_by_relations']}")
        
        # Equipment distribution analysis
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
        
        # Density analysis for Dense KG
        print("\nDENSE GRAPH ANALYSIS:")
        density_query = """
            MATCH (s:Symptom)-[:CAUSES]->(c:Cause)
            WITH s, count(c) as cause_count
            WHERE cause_count > 1
            RETURN count(s) as dense_symptoms, max(cause_count) as max_causes, avg(cause_count) as avg_causes
        """
        
        if equipment_filter:
            density_query = f"""
                MATCH (s:Symptom)-[:CAUSES]->(c:Cause)
                WHERE s.equipment CONTAINS '{equipment_filter}'
                WITH s, count(c) as cause_count
                WHERE cause_count > 1
                RETURN count(s) as dense_symptoms, max(cause_count) as max_causes, avg(cause_count) as avg_causes
            """
        
        density_result = session.run(density_query)
        density = density_result.single()
        
        if density and density['dense_symptoms'] > 0:
            print(f"   • Densified symptoms: {density['dense_symptoms']}")
            print(f"   • Maximum causes per symptom: {density['max_causes']}")
            print(f"   • Average causes per dense symptom: {density['avg_causes']:.2f}")
        else:
            print("   • No densification detected (1:1:1 structure)")

def main():
    """
    Main visualization pipeline with intelligent connection management
    
    Orchestrates the complete visualization process including connection establishment,
    statistical analysis, and graph rendering with support for various filtering
    and display options tailored to Dense Knowledge Graph exploration.
    """
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Optimized Dense Knowledge Graph Visualization")
    parser.add_argument("--kg-type", choices=["dense", "sparse", "dense_sc"], default="dense",
                       help="Knowledge Graph type to visualize")
    parser.add_argument("--limit", type=int, default=50,
                       help="Maximum number of triplets to display (default: 50)")
    parser.add_argument("--show-labels", action="store_true",
                       help="Display node names (only if <= 30 nodes)")
    parser.add_argument("--stats-only", action="store_true",
                       help="Display only statistics, without visualization")
    parser.add_argument("--equipment", type=str,
                       help="Filter by specific equipment type (e.g., 'FANUC')")
    
    args = parser.parse_args()

    # Establish connection using the same logic as Dense KG construction
    try:
        driver, user, password = get_neo4j_connection(args.kg_type)
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 'Connected!' as message")
            message = result.single()["message"]
            print(f"Connection successful: {message}")
        
        if args.stats_only:
            get_graph_stats(driver, args.equipment)
        else:
            # Display statistics first
            get_graph_stats(driver, args.equipment)
            
            # Then visualization
            plot_neo4j_graph_optimized(driver, args.limit, args.show_labels, args.equipment)
            
    except Exception as e:
        print(f"Connection or visualization error: {str(e)}")
        print("Verify Neo4j connection settings and credentials")
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    main()