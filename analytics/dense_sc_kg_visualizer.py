"""
Dense S&C Knowledge Graph Visualization Module

This module provides visualization capabilities for Dense Symptom & Cause Knowledge Graphs
constructed in the RAG diagnosis system. It creates optimized network visualizations with support
for cloud and local Neo4j deployments, featuring intelligent connection management,
statistical analysis, and customizable rendering options for Dense S&C graph exploration
with emphasis on combined symptom+cause text analysis.

Key components:
- Cloud/local connection management: Intelligent fallback system for Neo4j connectivity
- Dense S&C optimized visualization: Efficient rendering highlighting combined text relationships
- Equipment-aware filtering: Support for equipment-specific Dense S&C graph exploration
- Combined text analysis: Statistical validation of symptom+cause densification characteristics

Dependencies: neo4j, networkx, matplotlib, pyyaml, python-dotenv, argparse
Usage: Execute as standalone script with customizable parameters for Dense S&C graph exploration
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

def get_neo4j_connection(kg_type="dense_sc"):
    """
    Establish intelligent Cloud/Local Neo4j connection for Dense S&C visualization
    
    Implements cloud-first connection strategy with automatic local fallback,
    matching the connection logic used in the Dense S&C KG construction pipeline.
    
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

def plot_dense_sc_kg_graph(driver, limit=50, show_labels=False, equipment_filter=None):
    """
    Visualize a sample of the Dense S&C Knowledge Graph with combined text emphasis
    
    Creates an efficient network visualization of the Dense S&C Knowledge Graph with
    emphasis on the combined symptom+cause relationships, equipment filtering capabilities,
    and visual distinction from other graph types for clear structural analysis of
    densified symptom and cause combinations.
    
    Args:
        driver: Neo4j database driver instance
        limit (int): Maximum number of triplets to display (default: 50)
        show_labels (bool): Display node names (default: False)
        equipment_filter (str, optional): Filter by specific equipment type
    """
    print("Executing Cypher query for Dense S&C structure...")
    
    # Base query for Dense S&C triplet retrieval with combined text
    base_query = """
        MATCH (s:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
    """
    
    if equipment_filter:
        base_query += " WHERE s.equipment CONTAINS $equipment_filter"
        
    base_query += """
        RETURN s.name AS s, c.name AS c, r.name AS r, 
               s.equipment AS equipment, s.combined_text AS combined_text
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
        print("No data found in Dense S&C graph")
        return

    print(f"Constructing Dense S&C graph ({len(data)} triplets)...")
    G = nx.DiGraph()
    
    # Graph construction with node types and combined text metadata
    for row in data:
        # Add nodes with their type, equipment, and combined text information
        G.add_node(row["s"], node_type="Symptom", 
                  equipment=row.get("equipment", "unknown"), 
                  combined_text=row.get("combined_text", ""))
        G.add_node(row["c"], node_type="Cause", 
                  equipment=row.get("equipment", "unknown"))
        G.add_node(row["r"], node_type="Remedy", 
                  equipment=row.get("equipment", "unknown"))
        
        # Add relationships (potentially densified through S&C similarity)
        G.add_edge(row["s"], row["c"], label="CAUSES")
        G.add_edge(row["c"], row["r"], label="TREATED_BY")

    print(f"Dense S&C graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} relations")

    # Plot configuration optimized for Dense S&C visualization
    plt.figure(figsize=(16, 12))
    
    print("Computing layout for Dense S&C graph structure...")
    # Layout adapted to dense structures with combined text relationships
    if G.number_of_nodes() > 100:
        pos = nx.random_layout(G)  # Faster option
    else:
        # Spring layout with enhanced parameters for dense S&C visualization
        pos = nx.spring_layout(G, k=1.5, iterations=40, seed=42)
    
    # Colors by node type (distinctive palette for Dense S&C)
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'Unknown')
        if node_type == 'Symptom':
            node_colors.append('#8E44AD')  # Purple for S&C symptoms
        elif node_type == 'Cause':
            node_colors.append('#E67E22')  # Orange for causes
        elif node_type == 'Remedy':
            node_colors.append('#27AE60')  # Green for remedies
        else:
            node_colors.append('#95A5A6')  # Gray for unknown
    
    print("Drawing Dense S&C graph structure...")
    # Draw nodes with enhanced visibility for Dense S&C structure
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=350,
                          alpha=0.85)
    
    # Draw edges with emphasis on dense relationship structure
    nx.draw_networkx_edges(G, pos, 
                          edge_color='#2C3E50',
                          arrows=True,
                          arrowsize=18,
                          alpha=0.75,
                          width=1.8)
    
    # Optional labels optimized for Dense S&C visibility
    if show_labels and G.number_of_nodes() <= 30:
        # Truncate long labels for clarity
        labels = {node: (node[:10] + '...' if len(node) > 10 else node) 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, 
                               labels=labels,
                               font_size=7,
                               font_weight='bold',
                               font_color='white',
                               bbox=dict(boxstyle="round,pad=0.15", 
                                       facecolor='navy', 
                                       alpha=0.8))
    
    # Title specific to Dense S&C graph characteristics
    title = f"Dense S&C Knowledge Graph - Combined Text Structure ({len(data)} triplets)"
    if equipment_filter:
        title += f" - Equipment: {equipment_filter}"
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Color legend adapted for Dense S&C representation
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#8E44AD', 
                   markersize=10, label='Symptoms (S&C)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E67E22', 
                   markersize=10, label='Causes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#27AE60', 
                   markersize=10, label='Remedies')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Annotation specific to Dense S&C mode characteristics
    plt.text(0.02, 0.98, "Mode: DENSE S&C\n• Combined symptom+cause\n• Hybrid metric densification\n• Enhanced similarity", 
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    plt.axis('off')  # Hide axes
    plt.tight_layout()
    
    print("Dense S&C graph display completed")
    print("\nDENSE S&C CHARACTERISTICS:")
    print(f"   • Combined text structure: Symptom + Cause → Enhanced similarity")
    print(f"   • Hybrid metric densification with S&C context")
    print(f"   • Enriched relationships through combined text propagation")
    print(f"   • Graph density: {G.number_of_edges()}/{G.number_of_nodes()} = {G.number_of_edges()/G.number_of_nodes():.2f}")
    
    plt.show()

def get_dense_sc_graph_stats(driver, equipment_filter=None):
    """
    Display comprehensive statistics of the complete Dense S&C graph
    
    Provides detailed analysis of the Dense S&C Knowledge Graph including node counts,
    relationship distribution, equipment analysis, combined text validation, and
    densification metrics for understanding the S&C graph characteristics and
    enhanced similarity-based structure.
    
    Args:
        driver: Neo4j database driver instance
        equipment_filter (str, optional): Filter statistics by specific equipment type
    """
    print("Retrieving Dense S&C graph statistics...")
    
    with driver.session() as session:
        # Basic statistics query for Dense S&C structure
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
        
        print("\nCOMPREHENSIVE DENSE S&C GRAPH STATISTICS:")
        if equipment_filter:
            print(f"(Filtered by equipment: {equipment_filter})")
        
        print(f"   • Symptoms: {stats['symptoms']}")
        print(f"   • Causes: {stats['causes']}")
        print(f"   • Remedies: {stats['remedies']}")
        print(f"   • CAUSES relations: {stats['causes_relations']}")
        print(f"   • TREATED_BY relations: {stats['treated_by_relations']}")
        print(f"   • TOTAL nodes: {stats['symptoms'] + stats['causes'] + stats['remedies']}")
        print(f"   • TOTAL relations: {stats['causes_relations'] + stats['treated_by_relations']}")
        
        # Equipment distribution analysis for Dense S&C structure
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
        
        # Combined text analysis for Dense S&C
        print("\nCOMBINED TEXT ANALYSIS:")
        
        # Validate combined text presence
        combined_text_query = """
            MATCH (s:Symptom)
            WHERE s.combined_text IS NOT NULL
            RETURN count(s) as symptoms_with_combined,
                   avg(size(s.combined_text)) as avg_combined_length,
                   s.combined_text as example_text
            LIMIT 1
        """
        
        if equipment_filter:
            combined_text_query = f"""
                MATCH (s:Symptom)
                WHERE s.combined_text IS NOT NULL AND s.equipment CONTAINS '{equipment_filter}'
                RETURN count(s) as symptoms_with_combined,
                       avg(size(s.combined_text)) as avg_combined_length,
                       s.combined_text as example_text
                LIMIT 1
            """
        
        combined_result = session.run(combined_text_query)
        combined_stats = combined_result.single()
        
        if combined_stats and combined_stats['symptoms_with_combined'] > 0:
            print(f"   • Symptoms with combined text: {combined_stats['symptoms_with_combined']}")
            print(f"   • Average combined text length: {combined_stats['avg_combined_length']:.1f} characters")
            if combined_stats['example_text']:
                example = combined_stats['example_text'][:80] + "..." if len(combined_stats['example_text']) > 80 else combined_stats['example_text']
                print(f"   • Example combined text: {example}")
        else:
            print("   • No combined text found")
        
        # Densification analysis for Dense S&C KG
        print("\nDENSE S&C DENSIFICATION ANALYSIS:")
        
        # Validate densification through multiple cause relationships
        density_query = """
            MATCH (s:Symptom)-[:CAUSES]->(c:Cause)
            WITH s, count(c) as cause_count
            WHERE cause_count > 1
            RETURN count(s) as densified_symptoms, 
                   max(cause_count) as max_causes_per_symptom,
                   avg(cause_count) as avg_causes_per_densified_symptom
        """
        
        if equipment_filter:
            density_query = f"""
                MATCH (s:Symptom)-[:CAUSES]->(c:Cause)
                WHERE s.equipment CONTAINS '{equipment_filter}'
                WITH s, count(c) as cause_count
                WHERE cause_count > 1
                RETURN count(s) as densified_symptoms, 
                       max(cause_count) as max_causes_per_symptom,
                       avg(cause_count) as avg_causes_per_densified_symptom
            """
        
        density_result = session.run(density_query)
        density_stats = density_result.single()
        
        if density_stats and density_stats['densified_symptoms'] > 0:
            print(f"   • Densified symptoms (multiple causes): {density_stats['densified_symptoms']}")
            print(f"   • Maximum causes per symptom: {density_stats['max_causes_per_symptom']}")
            print(f"   • Average causes per densified symptom: {density_stats['avg_causes_per_densified_symptom']:.2f}")
            print("   • Densification method: Combined symptom+cause similarity")
        else:
            print("   • No densification detected (1:1:1 structure preserved)")
        
        # S&C specific propagation analysis
        print("\nS&C PROPAGATION METRICS:")
        
        # Analyze remedy sharing across similar symptom+cause combinations
        remedy_sharing_query = """
            MATCH (s1:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
            MATCH (s2:Symptom)-[:CAUSES]->(c)-[:TREATED_BY]->(r)
            WHERE s1 <> s2
            RETURN count(DISTINCT r) as shared_remedies,
                   count(DISTINCT c) as shared_causes,
                   count(DISTINCT [s1, s2]) as symptom_pairs_sharing_solutions
        """
        
        if equipment_filter:
            remedy_sharing_query = f"""
                MATCH (s1:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                MATCH (s2:Symptom)-[:CAUSES]->(c)-[:TREATED_BY]->(r)
                WHERE s1 <> s2 AND s1.equipment CONTAINS '{equipment_filter}' AND s2.equipment CONTAINS '{equipment_filter}'
                RETURN count(DISTINCT r) as shared_remedies,
                       count(DISTINCT c) as shared_causes,
                       count(DISTINCT [s1, s2]) as symptom_pairs_sharing_solutions
            """
        
        sharing_result = session.run(remedy_sharing_query)
        sharing_stats = sharing_result.single()
        
        if sharing_stats:
            print(f"   • Shared remedies across symptoms: {sharing_stats['shared_remedies']}")
            print(f"   • Shared causes across symptoms: {sharing_stats['shared_causes']}")
            print(f"   • Symptom pairs sharing solutions: {sharing_stats['symptom_pairs_sharing_solutions']}")
            print("   • Structure: Enhanced through S&C combined similarity")

def main():
    """
    Main Dense S&C visualization pipeline with intelligent connection management
    
    Orchestrates the complete Dense S&C visualization process including connection establishment,
    statistical analysis, and graph rendering with support for various filtering
    and display options tailored to Dense S&C Knowledge Graph exploration and
    combined symptom+cause text analysis.
    """
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Dense S&C Knowledge Graph Visualization")
    parser.add_argument("--kg-type", choices=["dense", "sparse", "dense_sc"], default="dense_sc",
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

    # Establish connection using the same logic as Dense S&C KG construction
    try:
        driver, user, password = get_neo4j_connection(args.kg_type)
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 'Connected to Dense S&C KG!' as message")
            message = result.single()["message"]
            print(f"Connection successful: {message}")
        
        if args.stats_only:
            get_dense_sc_graph_stats(driver, args.equipment)
        else:
            # Display statistics first
            get_dense_sc_graph_stats(driver, args.equipment)
            
            # Then visualization
            plot_dense_sc_kg_graph(driver, args.limit, args.show_labels, args.equipment)
            
    except Exception as e:
        print(f"Connection or visualization error: {str(e)}")
        print("Verify Neo4j connection settings and credentials")
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    main()