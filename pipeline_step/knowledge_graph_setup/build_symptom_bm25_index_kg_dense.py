"""
Construction index BM25 des symptômes uniquement - Version Simple
Indexe SEULEMENT les symptômes de la Knowledge Base Dense pour recherche lexicale
Pour lancer:
docker run --rm \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/pipeline_step:/app/pipeline_step \
  --network host \
  diagnosis-app \
  poetry run python pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_dense.py
"""

import os
import yaml
from pathlib import Path
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

def extract_symptoms_from_kg():
    """Extrait tous les symptômes uniques du Knowledge Graph Dense"""
    print("📊 Extraction des symptômes du Knowledge Graph Dense...")
    
    driver = get_neo4j_connection("dense")
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Symptom)
                RETURN DISTINCT s.name AS symptom_text, 
                       s.equipment AS equipment,
                       elementId(s) AS symptom_id
                ORDER BY s.name
            """)
            
            symptoms = []
            for record in result:
                symptoms.append({
                    'symptom_id': str(record['symptom_id']),
                    'symptom_text': record['symptom_text'],
                    'equipment': record['equipment'] or 'unknown'
                })
            
            print(f"✅ {len(symptoms)} symptômes uniques extraits")
            return symptoms
            
    except Exception as e:
        print(f"❌ Erreur extraction: {e}")
        return []
    finally:
        driver.close()

def build_symptom_bm25_index():
    """Construit l'index BM25 des symptômes"""
    print("🚀 Construction de l'index BM25 des symptômes...")
    
    # Configuration - 🔧 CORRECTION DU CHEMIN
    config = load_settings()
    index_path = config["paths"]["bm25_index_path"]  # 🆕 CHEMIN CORRIGÉ
    
    # Création du répertoire
    os.makedirs(index_path, exist_ok=True)
    
    # Schéma simple pour les symptômes
    schema = Schema(
        symptom_id=ID(stored=True, unique=True),
        symptom_text=TEXT(analyzer=StandardAnalyzer(), stored=True),
        equipment=ID(stored=True)
    )
    
    # Création de l'index
    if exists_in(index_path):
        print(f"⚠️ Index existant supprimé dans {index_path}")
        import shutil
        shutil.rmtree(index_path)
        os.makedirs(index_path, exist_ok=True)
    
    ix = create_in(index_path, schema)
    
    # Extraction des symptômes
    symptoms = extract_symptoms_from_kg()
    if not symptoms:
        raise ValueError("Aucun symptôme trouvé dans le KG Dense")
    
    # Indexation
    print(f"📝 Indexation de {len(symptoms)} symptômes...")
    writer = ix.writer()
    
    for symptom in symptoms:
        writer.add_document(
            symptom_id=symptom['symptom_id'],
            symptom_text=symptom['symptom_text'],
            equipment=symptom['equipment']
        )
    
    writer.commit()
    
    print(f"✅ Index BM25 Dense créé: {index_path}")
    print(f"📊 Symptômes indexés: {len(symptoms)}")
    
    return index_path

def main():
    """Pipeline principal"""
    print("🚀 CONSTRUCTION INDEX BM25 SYMPTÔMES - KG DENSE")
    print("=" * 55)
    
    try:
        index_path = build_symptom_bm25_index()
        
        print("\n✅ CONSTRUCTION TERMINÉE !")
        print(f"📁 Index créé: {index_path}")
        print("🎯 Prêt pour la recherche hybride des symptômes")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()