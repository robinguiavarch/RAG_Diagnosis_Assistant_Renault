"""
Construction Knowledge Graph Dense avec métrique hybride AUTONOME
Métrique: Cosine + Jaccard + Levenshtein (sans dépendances externes)
Pour lancer: 
docker run --rm \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/pipeline_step:/app/pipeline_step \
  --network host \
  diagnosis-app \
  poetry run python pipeline_step/knowledge_graph_setup/build_dense_knowledge_graph.py
"""

import os
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
import yaml

# === Import de la métrique hybride autonome ===
from hybrid_metric_build_kgs import create_autonomous_hybrid_metric

# === Chargement des variables d'environnement (.env) ===
load_dotenv()

def load_settings():
    """Charge la configuration depuis settings.yaml"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", "settings.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_neo4j_connection(kg_type="dense"):
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


# Configuration depuis settings.yaml
config = load_settings()
SIM_THRESHOLD = config["graph_retrieval"]["dense_similarity_threshold"]
TOP_K = config["graph_retrieval"]["dense_top_k_similar"]

# === CONNEXION NEO4J ===
driver = get_neo4j_connection("dense")

# === CHARGEMENT ET NETTOYAGE DES DONNÉES ===
def load_and_clean_data():
    """Charge et nettoie les données CSV en supprimant les doublons"""
    print("📂 Chargement du fichier CSV...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"📊 Données initiales : {len(df)} lignes")
    print(f"📋 Colonnes détectées : {list(df.columns)}")
    
    # Vérification que la colonne equipment existe
    if 'equipment' not in df.columns:
        print("❌ ERREUR: Colonne 'equipment' manquante dans le CSV")
        print(f"📋 Colonnes disponibles : {list(df.columns)}")
        raise ValueError("Colonne 'equipment' requise")
    
    # Suppression des lignes avec des valeurs manquantes (incluant equipment)
    df.dropna(subset=["symptom", "cause", "remedy", "equipment"], inplace=True)
    print(f"📊 Après suppression NaN : {len(df)} lignes")
    
    # Suppression des doublons exacts
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

# === ÉTAPE 1 – Création des triplets SCR avec equipment ===
def clear_database(tx):
    """Vide complètement la base Neo4j pour un fresh start"""
    tx.run("MATCH (n) DETACH DELETE n")

def insert_triplets(tx, s, c, r, equipment):
    """Insert un triplet avec equipment comme propriété sur chaque nœud"""
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

# === ÉTAPE 2 – Similarité HYBRIDE AUTONOME ===
def compute_similarity_hybrid_autonomous(symptom_list):
    """
    🆕 Métrique hybride AUTONOME (Cosine + Jaccard + Levenshtein)
    Pas de dépendance externe, tout intégré
    """
    print("🧠 Calcul métrique hybride AUTONOME (Cosine + Jaccard + Levenshtein)...")
    
    try:
        # Configuration des poids
        weights = {
            'cosine_alpha': 0.4,
            'jaccard_beta': 0.4, 
            'levenshtein_gamma': 0.2
        }
        
        print(f"⚖️ Poids configurés: {weights}")
        
        # Création de la métrique hybride
        metric = create_autonomous_hybrid_metric(weights)
        
        # Calcul de la matrice de similarité
        sim_matrix = metric.compute_similarity_matrix(symptom_list)
        
        print(f"✅ Matrice de similarité HYBRIDE AUTONOME calculée : {sim_matrix.shape}")
        return sim_matrix
        
    except Exception as e:
        print(f"❌ Erreur métrique hybride autonome: {e}")
        print("🔄 Fallback vers cosine similarity standard...")
        return compute_similarity_fallback(symptom_list)

def compute_similarity_fallback(symptom_list):
    """Fallback vers cosine similarity si métrique hybride échoue"""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("🔄 Fallback: Cosine similarity standard...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode(symptom_list, show_progress_bar=True)
    sim_matrix = cosine_similarity(vectors)
    np.fill_diagonal(sim_matrix, 0)  # Éviter auto-similarité
    return sim_matrix

# === ÉTAPE 3 – Création des relations SIMILAR_TO (temporaires) ===
def add_sim_relation(tx, s1, s2, similarity_score):
    """Ajoute une relation SIMILAR_TO avec le score de similarité"""
    tx.run("""
        MATCH (a:Symptom {name: $s1}), (b:Symptom {name: $s2})
        MERGE (a)-[:SIMILAR_TO {score: $score}]->(b)
    """, s1=s1, s2=s2, score=float(similarity_score))

# === ÉTAPE 4 – Propagation CAUSES / TREATED_BY ===
def propagate_links(tx):
    """Propage les relations CAUSES et TREATED_BY via similarité"""
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

# === ÉTAPE 5 – Suppression des SIMILAR_TO ===
def clean_similar(tx):
    """Supprime toutes les relations SIMILAR_TO temporaires"""
    tx.run("MATCH ()-[r:SIMILAR_TO]->() DELETE r")

# === STATISTIQUES ET VALIDATION ===
def print_graph_stats(tx):
    """Affiche les statistiques du graphe créé avec equipment"""
    result = tx.run("""
        RETURN 
        count{(s:Symptom)} as symptoms,
        count{(c:Cause)} as causes, 
        count{(r:Remedy)} as remedies,
        count{()-[:CAUSES]->()} as causes_relations,
        count{()-[:TREATED_BY]->()} as treated_by_relations
    """)
    
    stats = result.single()
    print("\n📈 STATISTIQUES DU GRAPHE DENSE CRÉÉ :")
    print(f"   • Symptômes : {stats['symptoms']}")
    print(f"   • Causes : {stats['causes']}")
    print(f"   • Remèdes : {stats['remedies']}")
    print(f"   • Relations CAUSES : {stats['causes_relations']}")
    print(f"   • Relations TREATED_BY : {stats['treated_by_relations']}")
    
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

# === MAIN PIPELINE ===
def main():
    """Pipeline principal d'import et enrichissement avec métrique hybride autonome"""
    print("🚀 DÉMARRAGE DU PIPELINE NEO4J DENSE - MÉTRIQUE HYBRIDE AUTONOME")
    print("=" * 75)
    print("🆕 NOUVEAU : Métrique hybride intégrée (Cosine + Jaccard + Levenshtein)")
    print("✅ Aucune dépendance externe - Construction directe cloud possible")
    print()
    
    try:
        # Chargement et nettoyage des données
        df = load_and_clean_data()
        
        # Extraction des symptômes uniques pour calcul de similarité
        symptoms = df["symptom"].unique().tolist()
        print(f"🔍 Symptômes uniques identifiés : {len(symptoms)}")
        
        # Calcul de la matrice de similarité hybride autonome
        sim_matrix = compute_similarity_hybrid_autonomous(symptoms)
        
        with driver.session() as session:
            print("\n📌 Étape 1 - Nettoyage de la base Neo4j...")
            session.write_transaction(clear_database)
            
            print("📌 Étape 2 - Insertion des triplets avec equipment...")
            for idx, row in df.iterrows():
                session.write_transaction(insert_triplets, 
                                        row["symptom"], 
                                        row["cause"], 
                                        row["remedy"],
                                        row["equipment"])
                if (idx + 1) % 1000 == 0:
                    print(f"   • {idx + 1}/{len(df)} triplets insérés...")
            
            print("📌 Étape 3 - Calcul et ajout des relations SIMILAR_TO...")
            similar_relations_added = 0
            for i, s1 in enumerate(symptoms):
                # Récupération des TOP_K voisins les plus similaires
                indices = np.argsort(-sim_matrix[i])[:TOP_K]
                for j in indices:
                    if sim_matrix[i][j] >= SIM_THRESHOLD:
                        session.write_transaction(add_sim_relation, 
                                                s1, 
                                                symptoms[j], 
                                                sim_matrix[i][j])
                        similar_relations_added += 1
                        
                if (i + 1) % 100 == 0:
                    print(f"   • {i + 1}/{len(symptoms)} symptômes traités...")
            
            print(f"   • Relations SIMILAR_TO créées : {similar_relations_added}")
            
            print("📌 Étape 4 - Propagation des liens...")
            session.write_transaction(propagate_links)
            
            print("📌 Étape 5 - Suppression des relations SIMILAR_TO...")
            session.write_transaction(clean_similar)
            
            print("📌 Étape 6 - Génération des statistiques...")
            session.read_transaction(print_graph_stats)
        
        print("\n✅ IMPORT DENSE AVEC MÉTRIQUE HYBRIDE AUTONOME TERMINÉ !")
        print("🎯 Caractéristiques :")
        print("   • Métrique hybride: Cosine + Jaccard + Levenshtein")
        print("   • Aucune dépendance externe (index BM25/FAISS)")
        print("   • Construction directe cloud possible")
        print("   • Equipment properties sur chaque nœud")
        print("   • Propagation sémantique enrichie")
        print()
        print("🔗 Connectez-vous à Neo4j Browser pour explorer")
        print("💡 Test: MATCH (s:Symptom) WHERE s.equipment CONTAINS 'FANUC' RETURN s LIMIT 5")
        
    except FileNotFoundError:
        print(f"❌ ERREUR : Fichier CSV introuvable : {CSV_PATH}")
        print("   Vérifiez que le fichier existe et que le chemin est correct.")
    except Exception as e:
        print(f"❌ ERREUR INATTENDUE : {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    main()