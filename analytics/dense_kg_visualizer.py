from py2neo import Graph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # pour éviter tout parsing LaTeX
import argparse
import os
from dotenv import load_dotenv
import random

# AJOUTER en début de script (après les imports existants)
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

def plot_neo4j_graph_optimized(uri, user, password, limit=50, show_labels=False):
    """
    Visualise un échantillon du Knowledge Graph Neo4j de manière optimisée
    
    Args:
        uri: URI Neo4j
        user: Utilisateur Neo4j  
        password: Mot de passe Neo4j
        limit: Nombre maximum de triplets à afficher (default: 50)
        show_labels: Afficher les noms des nœuds (default: False)
    """
    print("🔌 Connexion à Neo4j...")
    graph = Graph(uri, auth=(user, password))

    print(f"📦 Requête Cypher (LIMIT {limit})...")
    # Requête avec échantillonnage aléatoire
    data = graph.run(f"""
        MATCH (s:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
        RETURN s.name AS s, c.name AS c, r.name AS r
        ORDER BY rand()
        LIMIT {limit}
    """).data()

    if not data:
        print("❌ Aucune donnée trouvée dans le graphe !")
        return

    print(f"📈 Construction du graphe ({len(data)} triplets)...")
    G = nx.DiGraph()
    
    # Construction du graphe avec types de nœuds
    for row in data:
        # Ajout des nœuds avec leur type
        G.add_node(row["s"], node_type="Symptom")
        G.add_node(row["c"], node_type="Cause")
        G.add_node(row["r"], node_type="Remedy")
        
        # Ajout des relations
        G.add_edge(row["s"], row["c"], label="CAUSES")
        G.add_edge(row["c"], row["r"], label="TREATED_BY")

    print(f"📊 Graphe construit : {G.number_of_nodes()} nœuds, {G.number_of_edges()} relations")

    # Configuration du plot
    plt.figure(figsize=(15, 12))
    
    print("🎨 Calcul du layout (optimisé)...")
    # Layout plus rapide pour gros graphes
    if G.number_of_nodes() > 100:
        pos = nx.random_layout(G)  # Plus rapide
    else:
        pos = nx.spring_layout(G, k=1, iterations=30)  # Réduit les itérations
    
    # Couleurs par type de nœud
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'Unknown')
        if node_type == 'Symptom':
            node_colors.append('#FF6B6B')  # Rouge clair
        elif node_type == 'Cause':
            node_colors.append('#4ECDC4')  # Bleu-vert
        elif node_type == 'Remedy':
            node_colors.append('#45B7D1')  # Bleu
        else:
            node_colors.append('#FFA07A')  # Orange
    
    print("🖼️ Dessin du graphe...")
    # Dessin des nœuds
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=300,
                          alpha=0.8)
    
    # Dessin des arêtes
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=15,
                          alpha=0.6,
                          width=1.5)
    
    # Labels optionnels (seulement si demandé et peu de nœuds)
    if show_labels and G.number_of_nodes() <= 30:
        # Tronquer les labels longs
        labels = {node: (node[:15] + '...' if len(node) > 15 else node) 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, 
                               labels=labels,
                               font_size=8,
                               font_weight='bold')
    
    # Titre et légende
    plt.title(f"Knowledge Graph SCR - Échantillon ({len(data)} triplets)", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Légende des couleurs
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
                   markersize=10, label='Symptômes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', 
                   markersize=10, label='Causes'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1', 
                   markersize=10, label='Remèdes')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.axis('off')  # Masquer les axes
    plt.tight_layout()
    
    print("✅ Affichage terminé !")
    print("\n💡 CONSEILS :")
    print(f"   • Augmentez --limit pour voir plus de données (actuel: {limit})")
    print(f"   • Utilisez --show-labels pour afficher les noms (si <= 30 nœuds)")
    print(f"   • Fermez la fenêtre pour terminer le script")
    
    plt.show()

def get_graph_stats(uri, user, password):
    """Affiche les statistiques du graphe complet"""
    print("📊 Récupération des statistiques du graphe...")
    graph = Graph(uri, auth=(user, password))
    
    stats = graph.run("""
        RETURN 
        count{(s:Symptom)} as symptoms,
        count{(c:Cause)} as causes, 
        count{(r:Remedy)} as remedies,
        count{()-[:CAUSES]->()} as causes_relations,
        count{()-[:TREATED_BY]->()} as treated_by_relations
    """).data()[0]
    
    print("\n📈 STATISTIQUES DU GRAPHE COMPLET :")
    print(f"   • Symptômes : {stats['symptoms']}")
    print(f"   • Causes : {stats['causes']}")
    print(f"   • Remèdes : {stats['remedies']}")
    print(f"   • Relations CAUSES : {stats['causes_relations']}")
    print(f"   • Relations TREATED_BY : {stats['treated_by_relations']}")
    print(f"   • TOTAL nœuds : {stats['symptoms'] + stats['causes'] + stats['remedies']}")
    print(f"   • TOTAL relations : {stats['causes_relations'] + stats['treated_by_relations']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualisation optimisée du Knowledge Graph Neo4j")
    settings = load_settings()
    parser.add_argument("--uri", default=settings["neo4j"]["dense_uri"])
    parser.add_argument("--user", default=settings["neo4j"]["dense_user"])
    parser.add_argument("--password", default=settings["neo4j"]["dense_password"])
    parser.add_argument("--limit", type=int, default=50,
                       help="Nombre maximum de triplets à afficher (default: 50)")
    parser.add_argument("--show-labels", action="store_true",
                       help="Afficher les noms des nœuds (seulement si <= 30 nœuds)")
    parser.add_argument("--stats-only", action="store_true",
                       help="Afficher seulement les statistiques, sans graphique")
    
    args = parser.parse_args()

    if args.stats_only:
        get_graph_stats(args.uri, args.user, args.password)
    else:
        # Afficher d'abord les stats
        get_graph_stats(args.uri, args.user, args.password)
        
        # Puis la visualisation
        plot_neo4j_graph_optimized(args.uri, args.user, args.password, 
                                  args.limit, args.show_labels)