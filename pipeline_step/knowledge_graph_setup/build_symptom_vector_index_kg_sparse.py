"""
Construction index FAISS pour Knowledge Graph SPARSE avec connexion Cloud/Local
Index basé sur les symptômes du KG Sparse (avec doublons)
Pour lancer:
docker run --rm \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/pipeline_step:/app/pipeline_step \
  --network host \
  diagnosis-app \
  poetry run python pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_sparse.py
"""

import os
import pickle
import faiss
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import yaml

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

def build_symptom_index_sparse():
    """
    Construit l'index FAISS pour les symptômes de la Knowledge Base SPARSE
    Sauvegarde dans data/knowledge_base/symptom_embeddings_sparse/
    """
    print("🚀 Construction de l'index FAISS pour les symptômes de la Knowledge Base SPARSE...")
    
    # === CONFIGURATION ===
    config = load_settings()
    
    # 🆕 CONNEXION CLOUD/LOCAL INTELLIGENTE
    driver = get_neo4j_connection("sparse")
    
    # Modèle d'embedding
    model_name = config["models"]["embedding_model"]
    print(f"📦 Chargement du modèle : {model_name}")
    
    # Chemin de sortie pour les embeddings Sparse
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "symptom_embeddings_sparse")

    try:
        # Test de connexion
        with driver.session() as test_session:
            test_session.run("RETURN 1")
        print("✅ Connexion Neo4j Sparse réussie")
        
        # === EXTRACTION DES SYMPTÔMES DE LA KNOWLEDGE BASE SPARSE ===
        print("📊 Extraction des symptômes de la Knowledge Base Sparse...")
        with driver.session() as session:
            # Récupération de tous les symptômes (avec doublons pour Sparse)
            result = session.run("""
                MATCH (s:Symptom) 
                RETURN s.name AS name, s.equipment AS equipment, s.triplet_id AS triplet_id
                ORDER BY s.triplet_id
            """)
            
            symptoms_data = []
            for record in result:
                symptoms_data.append({
                    'name': record["name"],
                    'equipment': record["equipment"] or 'unknown',
                    'triplet_id': record["triplet_id"]
                })
        
        print(f"✅ {len(symptoms_data)} symptômes extraits de la Knowledge Base Sparse")
        
        if not symptoms_data:
            raise ValueError("❌ Aucun symptôme trouvé dans la Knowledge Base Sparse!")
        
        # Extraction des noms pour embedding
        symptom_names = [s['name'] for s in symptoms_data]
        
        # === GÉNÉRATION DES EMBEDDINGS ===
        print("🧠 Génération des embeddings avec SentenceTransformer...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            symptom_names, 
            show_progress_bar=True, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Embeddings générés : {embeddings.shape}")
        
        # === CONSTRUCTION DE L'INDEX FAISS ===
        print("🔧 Construction de l'index FAISS...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product pour embeddings normalisés
        index.add(embeddings.astype('float32'))
        
        print(f"✅ Index FAISS créé avec {index.ntotal} vecteurs de dimension {dim}")
        
        # === SAUVEGARDE ===
        print(f"💾 Sauvegarde dans : {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarde de l'index FAISS
        index_path = os.path.join(output_dir, "index.faiss")
        faiss.write_index(index, index_path)
        print(f"✅ Index FAISS sauvegardé : {index_path}")
        
        # Sauvegarde des métadonnées enrichies pour Sparse
        metadata_path = os.path.join(output_dir, "symptom_embedding_sparse.pkl")
        metadata = {
            'symptom_names': symptom_names,
            'symptoms_data': symptoms_data,  # 🆕 Données complètes avec equipment et triplet_id
            'model_name': model_name,
            'embedding_dim': dim,
            'total_symptoms': len(symptom_names),
            'source': 'knowledge_base_sparse',
            'connection_mode': 'cloud' if os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true" else 'local'
        }
        
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"✅ Métadonnées sauvegardées : {metadata_path}")
        
        # === STATISTIQUES FINALES ===
        unique_symptoms = len(set(symptom_names))
        unique_equipments = set(s['equipment'] for s in symptoms_data)
        
        print("\n📈 STATISTIQUES DE L'INDEX SPARSE :")
        print(f"   • Symptômes indexés (total) : {len(symptom_names)}")
        print(f"   • Symptômes uniques : {unique_symptoms}")
        print(f"   • Doublons conservés : {len(symptom_names) - unique_symptoms}")
        print(f"   • Dimension des embeddings : {dim}")
        print(f"   • Modèle utilisé : {model_name}")
        print(f"   • Source : Knowledge Base Sparse")
        print(f"   • Mode connexion : {metadata['connection_mode'].upper()}")
        print(f"   • Taille index FAISS : {os.path.getsize(index_path) / 1024 / 1024:.2f} MB")
        
        # Equipements uniques
        print(f"   • Équipements couverts : {len(unique_equipments)}")
        for eq in sorted(unique_equipments):
            count = sum(1 for s in symptoms_data if s['equipment'] == eq)
            print(f"     - {eq}: {count} symptômes")
        
        print(f"\n📁 Fichiers générés :")
        print(f"     - {index_path}")
        print(f"     - {metadata_path}")
        
        print("\n🎯 UTILISATION :")
        print("   Cet index peut maintenant être utilisé pour la recherche vectorielle")
        print("   des symptômes dans le pipeline RAG avec Knowledge Graph Sparse.")
        print("   Structure 1:1:1 préservée avec doublons intentionnels.")
        
    except Exception as e:
        print(f"❌ ERREUR lors de la construction de l'index Sparse : {str(e)}")
        raise
    finally:
        driver.close()
        print("🔌 Connexion Neo4j fermée")

def main():
    """Pipeline principal de construction de l'index FAISS Sparse"""
    print("🚀 DÉMARRAGE DE LA CONSTRUCTION DE L'INDEX FAISS SPARSE")
    print("=" * 60)
    print("📝 Objectif : Créer un index vectoriel des symptômes de la KB Sparse")
    print("🌐 Support : Cloud/Local automatique")
    print("🎯 Sortie : data/knowledge_base/symptom_embeddings_sparse/")
    print()
    
    try:
        build_symptom_index_sparse()
        print("\n✅ CONSTRUCTION DE L'INDEX FAISS SPARSE TERMINÉE AVEC SUCCÈS !")
        
    except FileNotFoundError as e:
        print(f"❌ ERREUR : Fichier de configuration manquant : {str(e)}")
        print("   Vérifiez que config/settings.yaml existe.")
    except Exception as e:
        print(f"❌ ERREUR INATTENDUE : {str(e)}")

if __name__ == "__main__":
    main()