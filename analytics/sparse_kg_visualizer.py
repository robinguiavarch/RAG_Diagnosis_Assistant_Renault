from py2neo import Graph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # pour éviter tout parsing LaTeX
import argparse
import os
from dotenv import load_dotenv
import random

import sys
from pathlib import Path
import yaml

# Ajouter la racine du projet au Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_settings():
    """Charge la configuration depuis settings.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
# Charger .env si dispo
load_dotenv()

def plot_sparse_kg_graph(uri, user, password, limit=50, show_labels=False):
    """
    Visualise un échantillon de la Knowledge Base SPARSE Neo4j
    
    Args:
        uri: URI Neo4j
        user: Utilisateur Neo4j  
        password: Mot de passe Neo4j
        limit: Nombre maximum de triplets à afficher (default: 50)
        show_labels: Afficher les noms des nœuds (default: False)
    """
    print("🔌 Connexion à Neo4j SPARSE (port 7689)...")
    graph = Graph(uri, auth=(user, password))

    print(f"📦 Requête Cypher SPARSE (LIMIT {limit})...")
    # Requête pour récupérer les triplets linéaires du graphe sparse
    data = graph.run(f"""
        MATCH (s:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
        RETURN s.name AS s, c.name AS c, r.name AS r
        ORDER BY rand()
        LIMIT {limit}
    """).data()

    if not data:
        print("❌ Aucune donnée trouvée dans le graphe sparse !")
        return

    print(f"📈 Construction du graphe SPARSE ({len(data)} triplets)...")
    G = nx.DiGraph()
    
    # Construction du graphe avec types de nœuds
    for row in data:
        # Ajout des nœuds avec leur type
        G.add_node(row["s"], node_type="Symptom")
        G.add_node(row["c"], node_type="Cause")
        G.add_node(row["r"], node_type="Remedy")
        
        # Ajout des relations (structure linéaire)
        G.add_edge(row["s"], row["c"], label="CAUSES")
        G.add_edge(row["c"], row["r"], label="TREATED_BY")

    print(f"📊 Graphe SPARSE construit : {G.number_of_nodes()} nœuds, {G.number_of_edges()} relations")

    # Configuration du plot
    plt.figure(figsize=(15, 12))
    
    print("🎨 Calcul du layout pour graphe SPARSE...")
    # Layout adapté aux structures linéaires
    if G.number_of_nodes() > 100:
        pos = nx.random_layout(G)  # Plus rapide
    else:
        # Layout hiérarchique pour mieux voir la structure linéaire
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Couleurs par type de nœud (palette différente pour distinguer du dense)
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'Unknown')
        if node_type == 'Symptom':
            node_colors.append('#E74C3C')  # Rouge plus vif
        elif node_type == 'Cause':
            node_colors.append('#3498DB')  # Bleu standard
        elif node_type == 'Remedy':
            node_colors.append('#2ECC71')  # Vert
        else:
            node_colors.append('#F39C12')  # Orange
    
    print("🖼️ Dessin du graphe SPARSE...")
    # Dessin des nœuds
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=400,
                          alpha=0.9)
    
    # Dessin des arêtes (plus épaisses pour montrer la structure linéaire)
    nx.draw_networkx_edges(G, pos, 
                          edge_color='#34495E',
                          arrows=True,
                          arrowsize=20,
                          alpha=0.7,
                          width=2)
    
    # Labels optionnels (seulement si demandé et peu de nœuds)
    if show_labels and G.number_of_nodes() <= 25:
        # Tronquer les labels longs
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
    
    # Titre spécifique au graphe sparse
    plt.title(f"Knowledge Base SPARSE - Structure Linéaire ({len(data)} triplets)", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Légende des couleurs (adaptée au sparse)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', 
                   markersize=10, label='Symptômes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', 
                   markersize=10, label='Causes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71', 
                   markersize=10, label='Remèdes')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Annotation spécifique au mode sparse
    plt.text(0.02, 0.98, "Mode: SPARSE\n• Structure linéaire\n• 1:1:1 relations\n• Pas d'enrichissement", 
             transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    plt.axis('off')  # Masquer les axes
    plt.tight_layout()
    
    print("✅ Affichage du graphe SPARSE terminé !")
    print("\n💡 CARACTÉRISTIQUES SPARSE :")
    print(f"   • Structure linéaire : Symptôme → Cause → Remède")
    print(f"   • Aucune propagation sémantique")
    print(f"   • Relations 1:1 du CSV original")
    print(f"   • Densité faible : {G.number_of_edges()}/{G.number_of_nodes()} = {G.number_of_edges()/G.number_of_nodes():.2f}")
    
    plt.show()

def get_sparse_graph_stats(uri, user, password):
    """Affiche les statistiques du graphe sparse complet"""
    print("\n📊 Récupération des statistiques du graphe SPARSE (port 7689)...")
    graph = Graph(uri, auth=(user, password))
    
    stats = graph.run("""
        RETURN 
        count{(s:Symptom)} as symptoms,
        count{(c:Cause)} as causes, 
        count{(r:Remedy)} as remedies,
        count{()-[:CAUSES]->()} as causes_relations,
        count{()-[:TREATED_BY]->()} as treated_by_relations
    """).data()[0]
    
    print("\n📈 STATISTIQUES DU GRAPHE SPARSE COMPLET :")
    print(f"   • Symptômes : {stats['symptoms']}")
    print(f"   • Causes : {stats['causes']}")
    print(f"   • Remèdes : {stats['remedies']}")
    print(f"   • Relations CAUSES : {stats['causes_relations']}")
    print(f"   • Relations TREATED_BY : {stats['treated_by_relations']}")
    
    total_nodes = stats['symptoms'] + stats['causes'] + stats['remedies']
    total_relations = stats['causes_relations'] + stats['treated_by_relations']
    
    print(f"   • TOTAL nœuds : {total_nodes}")
    print(f"   • TOTAL relations : {total_relations}")
    print(f"   • Densité : {total_relations/total_nodes:.2f} relations/nœud")
    print(f"   • Structure : Linéaire (1 Symptôme → 1 Cause → 1 Remède)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation de la Knowledge Base SPARSE Neo4j")
    settings = load_settings()
    parser.add_argument("--uri", default=settings["neo4j"]["sparse_uri"])
    parser.add_argument("--user", default=settings["neo4j"]["sparse_user"])  
    parser.add_argument("--password", default=settings["neo4j"]["sparse_password"])
    parser.add_argument("--limit", type=int, default=50,
                       help="Nombre maximum de triplets à afficher (default: 50)")
    parser.add_argument("--show-labels", action="store_true",
                       help="Afficher les noms des nœuds (seulement si <= 25 nœuds)")
    parser.add_argument("--stats-only", action="store_true",
                       help="Afficher seulement les statistiques, sans graphique")
    
    args = parser.parse_args()

    if args.stats_only:
        get_sparse_graph_stats(args.uri, args.user, args.password)
    else:
        # Afficher d'abord les stats
        get_sparse_graph_stats(args.uri, args.user, args.password)
        
        # Puis la visualisation
        plot_sparse_kg_graph(args.uri, args.user, args.password, 
                            args.limit, args.show_labels)