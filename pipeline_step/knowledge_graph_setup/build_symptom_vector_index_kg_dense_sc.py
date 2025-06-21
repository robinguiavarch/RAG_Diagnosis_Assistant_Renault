"""
Construction index FAISS pour Knowledge Graph Dense S&C avec connexion Cloud/Local
Index basé sur les textes combinés symptôme + cause
Pour lancer:
docker run --rm \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/pipeline_step:/app/pipeline_step \
  --network host \
  diagnosis-app \
  poetry run python pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_dense_sc.py
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

def build_symptom_index_dense_sc():
    """
    Construit l'index FAISS pour les symptômes Dense S&C
    Sauvegarde dans data/knowledge_base/symptom_embeddings_dense_s&c/
    """
    print("🚀 Construction de l'index FAISS pour Dense S&C (Symptôme + Cause)...")
    
    # === CONFIGURATION ===
    config = load_settings()
    
    # 🆕 CONNEXION CLOUD/LOCAL INTELLIGENTE
    driver = get_neo4j_connection("dense_sc")
    
    # Modèle d'embedding
    model_name = config["models"]["embedding_model"]
    print(f"📦 Chargement du modèle : {model_name}")
    
    # Chemin de sortie pour les embeddings Dense S&C
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "symptom_embeddings_dense_s&c")

    try:
        # Test de connexion
        with driver.session() as test_session:
            test_session.run("RETURN 1")
        print("✅ Connexion Neo4j Dense S&C réussie")
        
        # === EXTRACTION DES DONNÉES S&C ===
        print("📊 Extraction des symptômes avec texte combiné...")
        with driver.session() as session:
            result = session.run("""
                MATCH (s:Symptom)
                WHERE s.combined_text IS NOT NULL
                RETURN DISTINCT s.name AS symptom_name, 
                       s.combined_text AS combined_text,
                       s.equipment AS equipment
                ORDER BY s.name
            """)
            
            symptoms_data = []
            for record in result:
                symptoms_data.append({
                    'symptom_name': record["symptom_name"],
                    'combined_text': record["combined_text"],
                    'equipment': record["equipment"] or 'unknown'
                })
        
        print(f"✅ {len(symptoms_data)} symptômes S&C extraits")
        
        if not symptoms_data:
            raise ValueError("❌ Aucun symptôme S&C trouvé dans la Knowledge Base!")
        
        # Extraction des textes pour embedding
        symptom_names = [s['symptom_name'] for s in symptoms_data]
        combined_texts = [s['combined_text'] for s in symptoms_data]
        
        print(f"📝 Exemples de textes combinés :")
        for i, text in enumerate(combined_texts[:3]):
            print(f"   {i+1}. {text}")
        
        # === GÉNÉRATION DES EMBEDDINGS S&C ===
        print("🧠 Génération des embeddings avec textes combinés...")
        model = SentenceTransformer(model_name)
        
        # 🆕 EMBEDDING DES TEXTES COMBINÉS (symptôme + cause)
        embeddings = model.encode(
            combined_texts,  # Utilise les textes combinés, pas seulement les symptômes
            show_progress_bar=True, 
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Embeddings S&C générés : {embeddings.shape}")
        
        # === CONSTRUCTION DE L'INDEX FAISS ===
        print("🔧 Construction de l'index FAISS S&C...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product pour embeddings normalisés
        index.add(embeddings.astype('float32'))
        
        print(f"✅ Index FAISS S&C créé avec {index.ntotal} vecteurs de dimension {dim}")
        
        # === SAUVEGARDE ===
        print(f"💾 Sauvegarde dans : {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarde de l'index FAISS
        index_path = os.path.join(output_dir, "index.faiss")
        faiss.write_index(index, index_path)
        print(f"✅ Index FAISS sauvegardé : {index_path}")
        
        # 🆕 Sauvegarde des métadonnées enrichies S&C
        metadata_path = os.path.join(output_dir, "symptom_embedding_dense_s&c.pkl")
        metadata = {
            'symptom_names': symptom_names,
            'combined_texts': combined_texts,  # 🆕 Textes combinés
            'symptoms_data': symptoms_data,    # 🆕 Données complètes
            'model_name': model_name,
            'embedding_dim': dim,
            'total_symptoms': len(symptom_names),
            'source': 'knowledge_base_dense_s&c',
            'indexing_method': 'symptom_plus_cause_combined',  # 🆕 Méthode d'indexation
            'connection_mode': 'cloud' if os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true" else 'local'
        }
        
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        print(f"✅ Métadonnées S&C sauvegardées : {metadata_path}")
        
        # === STATISTIQUES FINALES ===
        unique_symptoms = len(set(symptom_names))
        unique_equipments = set(s['equipment'] for s in symptoms_data)
        
        print("\n📈 STATISTIQUES DE L'INDEX DENSE S&C :")
        print(f"   • Symptômes indexés : {len(symptom_names)}")
        print(f"   • Symptômes uniques : {unique_symptoms}")
        print(f"   • Méthode : Symptôme + Cause combinés")
        print(f"   • Dimension des embeddings : {dim}")
        print(f"   • Modèle utilisé : {model_name}")
        print(f"   • Source : Knowledge Base Dense S&C")
        print(f"   • Mode connexion : {metadata['connection_mode'].upper()}")
        print(f"   • Taille index FAISS : {os.path.getsize(index_path) / 1024 / 1024:.2f} MB")
        
        # Equipements couverts
        print(f"   • Équipements couverts : {len(unique_equipments)}")
        for eq in sorted(unique_equipments):
            count = sum(1 for s in symptoms_data if s['equipment'] == eq)
            print(f"     - {eq}: {count} symptômes")
        
        print(f"\n📁 Fichiers générés :")
        print(f"     - {index_path}")
        print(f"     - {metadata_path}")
        
        print("\n🎯 UTILISATION :")
        print("   Cet index peut maintenant être utilisé pour la recherche vectorielle")
        print("   enrichie par symptôme + cause dans le pipeline RAG Dense S&C.")
        print("   La recherche sera plus contextuelle grâce à l'enrichissement sémantique.")
        
    except Exception as e:
        print(f"❌ ERREUR lors de la construction de l'index S&C : {str(e)}")
        raise
    finally:
        driver.close()
        print("🔌 Connexion Neo4j fermée")

def main():
    """Pipeline principal de construction de l'index FAISS Dense S&C"""
    print("🚀 DÉMARRAGE DE LA CONSTRUCTION DE L'INDEX FAISS DENSE S&C")
    print("=" * 70)
    print("📝 Objectif : Index vectoriel Symptôme + Cause combinés")
    print("🌐 Support : Cloud/Local automatique")
    print("🎯 Sortie : data/knowledge_base/symptom_embeddings_dense_s&c/")
    print()
    
    try:
        build_symptom_index_dense_sc()
        print("\n✅ CONSTRUCTION DE L'INDEX FAISS DENSE S&C TERMINÉE !")
        
    except FileNotFoundError as e:
        print(f"❌ ERREUR : Fichier de configuration manquant : {str(e)}")
    except Exception as e:
        print(f"❌ ERREUR INATTENDUE : {str(e)}")

if __name__ == "__main__":
    main()