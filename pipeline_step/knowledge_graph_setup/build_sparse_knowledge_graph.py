"""
Pour lancer:
docker run --rm \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/pipeline_step:/app/pipeline_step \
  --network host \
  diagnosis-app \
  poetry run python pipeline_step/knowledge_graph_setup/build_sparse_knowledge_graph.py
"""

"""
Construction Knowledge Graph Sparse avec support Cloud/Local
Structure simple 1:1:1 sans densification
"""

import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import yaml

# === Chargement des variables d'environnement (.env) ===
load_dotenv()

def load_settings():
    """Charge la configuration depuis settings.yaml"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", "settings.yaml")
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_neo4j_connection(kg_type="sparse"):
    """
    🌐 Connexion intelligente Cloud/Local
    kg_type: "dense", "sparse", ou "dense_sc"
    """
    load_dotenv()
    
    # Priorité au Cloud si activé
    cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
    
    if cloud_enabled:
        print(f"🌐 MODE CLOUD pour {kg_type.upper()}")
        
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
            print(f"🔌 Connexion Cloud {kg_type}: {uri}")
            return GraphDatabase.driver(uri, auth=("neo4j", password))
        else:
            print(f"❌ Credentials cloud manquants pour {kg_type}")
            cloud_enabled = False
    
    # Fallback Local
    print(f"🏠 MODE LOCAL pour {kg_type.upper()}")
    
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
    
    print(f"🔌 Connexion Local {kg_type}: {uri}")
    return GraphDatabase.driver(uri, auth=(user, password))

# === PARAMÈTRES ===
script_dir = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "scr_triplets", "doc-R-30iB_scr_triplets.csv")

# === CONNEXION NEO4J ===
driver = get_neo4j_connection("sparse")

def find_csv_file():
    """Trouve le fichier CSV en testant plusieurs chemins"""
    # Chemin principal
    main_path = CSV_PATH
    
    # Chemins alternatifs
    alt_paths = [
        os.path.join(script_dir, "..", "..", "data", "extract_scr", "doc-R-30iB_scr_triplets.csv"),
        os.path.join(script_dir, "..", "data", "extract_scr", "doc-R-30iB_scr_triplets.csv"),
        os.path.join(script_dir, "data", "extract_scr", "doc-R-30iB_scr_triplets.csv"),
        os.path.join("data", "extract_scr", "doc-R-30iB_scr_triplets.csv"),
        os.path.join("data", "knowledge_base", "scr_triplets", "doc-R-30iB_scr_triplets.csv")
    ]
    
    # Test du chemin principal
    if os.path.exists(main_path):
        return main_path
    
    # Test des chemins alternatifs
    print("📂 Vérification chemins alternatifs...")
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            print(f"✅ Fichier trouvé: {alt_path}")
            return alt_path
    
    # Aucun fichier trouvé
    raise FileNotFoundError(f"Fichier CSV introuvable dans tous les chemins testés")

# === CHARGEMENT ET NETTOYAGE DES DONNÉES ===
def load_and_clean_data():
    """Charge et nettoie les données CSV en supprimant les doublons"""
    print("📂 Chargement du fichier CSV...")
    
    try:
        csv_path = find_csv_file()
        df = pd.read_csv(csv_path)
        print(f"✅ Fichier CSV chargé: {csv_path}")
    except FileNotFoundError as e:
        print(f"❌ ERREUR: {str(e)}")
        raise
    
    print(f"📊 Données initiales : {len(df)} lignes")
    print(f"📋 Colonnes détectées : {list(df.columns)}")
    
    # Vérification que la colonne equipment existe
    if 'equipment' not in df.columns:
        print("❌ ERREUR: Colonne 'equipment' manquante dans le CSV")
        raise ValueError("Colonne 'equipment' requise")
    
    # Suppression des lignes avec des valeurs manquantes (incluant equipment)
    df.dropna(subset=["symptom", "cause", "remedy", "equipment"], inplace=True)
    print(f"📊 Après suppression NaN : {len(df)} lignes")
    
    # Suppression des doublons exacts (même symptom, cause, remedy, equipment)
    df_before_dedup = len(df)
    df = df.drop_duplicates(subset=["symptom", "cause", "remedy", "equipment"], keep="first")
    duplicates_removed = df_before_dedup - len(df)
    print(f"📊 Doublons supprimés : {duplicates_removed}")
    print(f"📊 Données finales : {len(df)} lignes")
    
    # Affichage des équipements uniques
    unique_equipment = df['equipment'].unique()
    print(f"🏭 Équipements uniques trouvés : {len(unique_equipment)}")
    for eq in sorted(unique_equipment):
        count = len(df[df['equipment'] == eq])
        print(f"   • {eq}: {count} triplets")
    
    return df

# === NETTOYAGE COMPLET DE LA BASE ===
def clear_database(tx):
    """Vide complètement la base Neo4j pour un fresh start"""
    tx.run("MATCH (n) DETACH DELETE n")

# === INSERTION DES TRIPLETS SCR (APPROCHE SPARSE) ===
def insert_triplets_sparse(tx, s, c, r, equipment, triplet_id):
    """Insert un triplet SCR avec equipment en préservant TOUS les nœuds"""
    tx.run("""
        CREATE (sym:Symptom {name: $s, equipment: $equipment, triplet_id: $triplet_id})
        CREATE (cause:Cause {name: $c, equipment: $equipment, triplet_id: $triplet_id})
        CREATE (rem:Remedy {name: $r, equipment: $equipment, triplet_id: $triplet_id})
        CREATE (sym)-[:CAUSES]->(cause)
        CREATE (cause)-[:TREATED_BY]->(rem)
    """, s=s, c=c, r=r, equipment=equipment, triplet_id=triplet_id)

# === STATISTIQUES ET VALIDATION ===
def print_graph_stats(tx):
    """Affiche les statistiques du graphe sparse avec equipment"""
    result = tx.run("""
        RETURN 
        count{(s:Symptom)} as symptoms,
        count{(c:Cause)} as causes, 
        count{(r:Remedy)} as remedies,
        count{()-[:CAUSES]->()} as causes_relations,
        count{()-[:TREATED_BY]->()} as treated_by_relations
    """)
    
    stats = result.single()
    print("\n📈 STATISTIQUES DU GRAPHE SPARSE :")
    print(f"   • Symptômes : {stats['symptoms']}")
    print(f"   • Causes : {stats['causes']}")
    print(f"   • Remèdes : {stats['remedies']}")
    print(f"   • Relations CAUSES : {stats['causes_relations']}")
    print(f"   • Relations TREATED_BY : {stats['treated_by_relations']}")
    print(f"   • TOTAL nœuds : {stats['symptoms'] + stats['causes'] + stats['remedies']}")
    print(f"   • TOTAL relations : {stats['causes_relations'] + stats['treated_by_relations']}")
    
    # Statistiques par équipement
    print("\n🏭 RÉPARTITION PAR ÉQUIPEMENT :")
    eq_result = tx.run("""
        MATCH (s:Symptom)
        WHERE s.equipment IS NOT NULL
        RETURN s.equipment as equipment, count(s) as symptom_count
        ORDER BY symptom_count DESC
    """)
    
    for record in eq_result:
        equipment = record['equipment']
        count = record['symptom_count']
        print(f"   • {equipment}: {count} symptômes")

def check_orphaned_nodes(tx):
    """Vérifie s'il y a des nœuds orphelins dans le graphe sparse"""
    result = tx.run("""
        MATCH (n)
        WHERE NOT (n)--()
        RETURN count(n) as orphaned_nodes
    """)
    
    orphaned = result.single()["orphaned_nodes"]
    if orphaned > 0:
        print(f"   ⚠️  Nœuds orphelins détectés : {orphaned}")
    else:
        print(f"   ✅ Aucun nœud orphelin")

# === PIPELINE PRINCIPAL (SPARSE + EQUIPMENT + CLOUD) ===
def main():
    """Pipeline de création d'une Knowledge Base SPARSE avec equipment et support cloud"""
    print("🚀 DÉMARRAGE DU PIPELINE NEO4J SPARSE + EQUIPMENT + CLOUD")
    print("=" * 70)
    print("📝 Mode : Knowledge Base SPARSE (structure simple 1:1:1)")
    print("🌐 Support : Cloud/Local automatique")
    print("🆕 Propriétés equipment sur chaque nœud")
    print()
    
    try:
        # Test de connexion
        print("🔌 Test de connexion...")
        with driver.session() as session:
            result = session.run("RETURN 'Connected to Sparse KG!' as message")
            message = result.single()["message"]
            print(f"✅ {message}")
        
        # Chargement et nettoyage des données
        df = load_and_clean_data()
        
        with driver.session() as session:
            print("\n📌 Étape 1 - Nettoyage de la base Neo4j Sparse...")
            session.execute_write(clear_database)
            
            print("📌 Étape 2 - Insertion des triplets SCR avec equipment...")
            for idx, row in df.iterrows():
                session.execute_write(insert_triplets_sparse, 
                                    row["symptom"], 
                                    row["cause"], 
                                    row["remedy"],
                                    row["equipment"],
                                    idx)  # ID unique pour traçabilité
                if (idx + 1) % 1000 == 0:
                    print(f"   • {idx + 1}/{len(df)} triplets insérés...")
            
            print("📌 Étape 3 - Génération des statistiques...")
            session.execute_read(print_graph_stats)
            session.execute_read(check_orphaned_nodes)
        
        print("\n✅ CRÉATION KNOWLEDGE BASE SPARSE + EQUIPMENT TERMINÉE !")
        print("📊 Caractéristiques :")
        print("   • Structure linéaire : 1 Symptôme → 1 Cause → 1 Remède")
        print("   • Propriété equipment sur chaque nœud")
        print("   • AUCUNE densification (pas de métrique hybride)")
        print("   • AUCUNE déduplication des causes/remèdes")
        print("   • Préservation totale des doublons (sauf triplets identiques)")
        print("   • Relations 1:1:1 strictes du CSV original")
        print("   • Traçabilité parfaite via triplet_id")
        print("   • 🌐 Compatible Cloud/Local automatique")
        print()
        print("🔗 Connectez-vous à Neo4j Browser pour explorer")
        print("💡 Comparez avec les Knowledge Bases Dense !")
        print()
        print("🔍 Requête test equipment :")
        print("   MATCH (s:Symptom) WHERE s.equipment CONTAINS 'FANUC' RETURN s LIMIT 5")
        
    except FileNotFoundError:
        csv_path = find_csv_file() if 'find_csv_file' in locals() else CSV_PATH
        print(f"❌ ERREUR : Fichier CSV introuvable : {csv_path}")
        print("   Vérifiez que le fichier existe et que le chemin est correct.")
    except Exception as e:
        print(f"❌ ERREUR INATTENDUE : {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    main()