"""
Construction index BM25 des symptômes pour KG Dense S&C - Version Simple
Index basé sur les textes combinés symptôme + cause
Pour lancer:
docker run --rm \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/pipeline_step:/app/pipeline_step \
  --network host \
  diagnosis-app \
  poetry run python pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_dense_sc.py
"""

import os
import yaml
from neo4j import GraphDatabase
from dotenv import load_dotenv
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, exists_in
from whoosh.analysis import StandardAnalyzer

load_dotenv()

def load_settings():
    """Charge la configuration"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "..", "config", "settings.yaml")
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_neo4j_connection(kg_type="dense_sc"):
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

def extract_symptoms_from_kg_dense_sc():
    """Extrait les symptômes avec texte combiné du KG Dense S&C"""
    print("📊 Extraction des symptômes du KG Dense S&C...")
    
    driver = get_neo4j_connection("dense_sc")
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Symptom)
                WHERE s.combined_text IS NOT NULL
                RETURN DISTINCT s.name AS symptom_name,
                       s.combined_text AS combined_text,
                       s.equipment AS equipment,
                       id(s) AS symptom_id
                ORDER BY s.name
            """)
            
            symptoms = []
            for record in result:
                symptoms.append({
                    'symptom_id': f"dense_sc_{record['symptom_id']}",
                    'symptom_text': record['combined_text'],  # 🆕 Utilise le texte combiné
                    'symptom_name': record['symptom_name'],
                    'equipment': record['equipment'] or 'unknown'
                })
            
            print(f"✅ {len(symptoms)} symptômes Dense S&C extraits")
            return symptoms
            
    except Exception as e:
        print(f"❌ Erreur extraction Dense S&C: {e}")
        return []
    finally:
        driver.close()

def build_symptom_bm25_index_dense_sc():
    """Construit l'index BM25 des symptômes Dense S&C"""
    print("🚀 Construction de l'index BM25 des symptômes Dense S&C...")
    
    # Configuration - 🔧 CORRECTION DU CHEMIN
    config = load_settings()
    index_path = config["paths"]["bm25_dense_sc_index_path"]  # 🆕 CHEMIN CORRIGÉ
    
    # Création du répertoire
    os.makedirs(index_path, exist_ok=True)
    
    # Schéma enrichi pour S&C
    schema = Schema(
        symptom_id=ID(stored=True, unique=True),
        symptom_text=TEXT(analyzer=StandardAnalyzer(), stored=True),  # Texte combiné
        symptom_name=TEXT(analyzer=StandardAnalyzer(), stored=True),  # Nom original
        equipment=ID(stored=True)
    )
    
    # Suppression index existant
    if exists_in(index_path):
        import shutil
        shutil.rmtree(index_path)
        os.makedirs(index_path, exist_ok=True)
    
    ix = create_in(index_path, schema)
    
    # Extraction et indexation
    symptoms = extract_symptoms_from_kg_dense_sc()
    if not symptoms:
        raise ValueError("Aucun symptôme S&C trouvé dans le KG Dense S&C")
    
    print(f"📝 Indexation de {len(symptoms)} symptômes Dense S&C...")
    print(f"📝 Exemple de texte indexé: {symptoms[0]['symptom_text']}")
    
    writer = ix.writer()
    
    for symptom in symptoms:
        writer.add_document(
            symptom_id=symptom['symptom_id'],
            symptom_text=symptom['symptom_text'],      # Texte combiné pour recherche
            symptom_name=symptom['symptom_name'],      # Nom original
            equipment=symptom['equipment']
        )
    
    writer.commit()
    
    print(f"✅ Index BM25 Dense S&C créé: {index_path}")
    print(f"📊 Symptômes indexés: {len(symptoms)} (symptôme + cause)")
    
    return index_path

def main():
    """Pipeline principal"""
    print("🚀 CONSTRUCTION INDEX BM25 SYMPTÔMES - KG DENSE S&C")
    print("=" * 60)
    
    try:
        index_path = build_symptom_bm25_index_dense_sc()
        print(f"\n✅ CONSTRUCTION TERMINÉE !")
        print(f"📁 Index créé: {index_path}")
        print("🎯 Prêt pour la recherche hybride Dense S&C")
        print("🔗 Recherche enrichie symptôme + cause")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()