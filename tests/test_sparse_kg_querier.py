#!/usr/bin/env python3
"""
Script de diagnostic pour la connexion Neo4j Sparse
Test du sparse_kg_querier.py
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

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


def test_neo4j_sparse_connection():
    """Test de connexion Neo4j Sparse avec différentes configurations"""
    
    print("🔧 DIAGNOSTIC CONNEXION NEO4J SPARSE")
    print("=" * 50)
    
    # Affichage des variables d'environnement Sparse
    print("📋 Variables d'environnement Sparse:")
    print(f"   NEO4J_URI_SPARSE: {os.getenv('NEO4J_URI_SPARSE', 'NON DÉFINIE')}")
    print(f"   NEO4J_USER_SPARSE: {os.getenv('NEO4J_USER_SPARSE', 'NON DÉFINIE')}")
    print(f"   NEO4J_PASS_SPARSE: {'***' if os.getenv('NEO4J_PASS_SPARSE') else 'NON DÉFINIE'}")
    print()
    
    # Test de configuration Sparse
    settings = load_settings()
    configurations = [
        {
            "name": "Configuration Sparse (settings.yaml)",
            "uri": settings["neo4j"]["sparse_uri"],
            "user": settings["neo4j"]["sparse_user"],
            "password": settings["neo4j"]["sparse_password"]
        },
        {
            "name": "Configuration Sparse (variables env)",
            "uri": os.getenv("NEO4J_URI_SPARSE", "bolt://localhost:7689"),
            "user": os.getenv("NEO4J_USER_SPARSE", "neo4j"),
            "password": os.getenv("NEO4J_PASS_SPARSE", "password")
        }
    ]
    
    for config in configurations:
        print(f"🔍 Test: {config['name']}")
        print(f"   URI: {config['uri']}")
        print(f"   User: {config['user']}")
        print(f"   Password: {'***' if config['password'] else '(vide)'}")
        
        try:
            driver = GraphDatabase.driver(
                config['uri'], 
                auth=(config['user'], config['password'])
            )
            
            # Test de connexion
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    print("   ✅ CONNEXION RÉUSSIE!")
                    
                    # Test de la structure de la base Sparse
                    result = session.run("""
                        RETURN 
                        count{(s:Symptom)} as symptoms,
                        count{(c:Cause)} as causes,
                        count{(r:Remedy)} as remedies
                    """)
                    stats = result.single()
                    print(f"   📊 Base de données Sparse:")
                    print(f"      Symptômes: {stats['symptoms']}")
                    print(f"      Causes: {stats['causes']}")
                    print(f"      Remèdes: {stats['remedies']}")
                    
                    # 🆕 Test spécifique Sparse - triplet_id
                    result = session.run("""
                        MATCH (s:Symptom)
                        WHERE s.triplet_id IS NOT NULL
                        RETURN count(s) as symptoms_with_triplet_id,
                               min(s.triplet_id) as min_triplet_id,
                               max(s.triplet_id) as max_triplet_id
                    """)
                    sparse_stats = result.single()
                    if sparse_stats:
                        print(f"   🔗 Spécificité Sparse:")
                        print(f"      Symptômes avec triplet_id: {sparse_stats['symptoms_with_triplet_id']}")
                        print(f"      Range triplet_id: {sparse_stats['min_triplet_id']} - {sparse_stats['max_triplet_id']}")
                    
                    # Test structure 1:1:1
                    result = session.run("""
                        MATCH (s:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                        WHERE s.triplet_id = c.triplet_id AND c.triplet_id = r.triplet_id
                        RETURN count(*) as valid_triplets
                    """)
                    triplet_stats = result.single()
                    if triplet_stats:
                        print(f"      Triplets 1:1:1 valides: {triplet_stats['valid_triplets']}")
                    
                    driver.close()
                    return config
            
            driver.close()
            
        except Exception as e:
            print(f"   ❌ Échec: {str(e)}")
        
        print()
    
    print("❌ AUCUNE CONFIGURATION SPARSE N'A FONCTIONNÉ")
    return None

def test_sparse_querier_functionality():
    """Test des fonctions principales du sparse_kg_querier"""
    print("\n🧪 TEST DES FONCTIONS SPARSE KG QUERIER")
    print("=" * 50)
    
    try:
        # Import du module à tester
        from core.retrieval_graph.sparse_kg_querier import (
            get_structured_context_sparse,
            get_similar_symptoms_sparse,
            load_symptom_index_sparse
        )
        
        print("✅ Import des modules réussi")
        
        # Test 1: Chargement de l'index
        print("\n🔍 Test 1: Chargement index FAISS Sparse")
        try:
            index, metadata = load_symptom_index_sparse()
            print(f"   ✅ Index chargé: {index.ntotal} vecteurs")
            print(f"   ✅ Métadonnées: {len(metadata.get('symptom_names', []))} symptômes")
            if 'symptoms_data' in metadata:
                print(f"   ✅ Données Sparse: {len(metadata['symptoms_data'])} éléments avec triplet_id")
        except Exception as e:
            print(f"   ❌ Erreur chargement index: {e}")
            return False
        
        # Test 2: Recherche de symptômes similaires
        print("\n🔍 Test 2: Recherche symptômes similaires Sparse")
        try:
            test_query = "motor overheating error"
            similar_symptoms = get_similar_symptoms_sparse(test_query)
            print(f"   ✅ Requête: '{test_query}'")
            print(f"   ✅ Symptômes trouvés: {len(similar_symptoms)}")
            for i, symptom_data in enumerate(similar_symptoms[:3]):
                if len(symptom_data) >= 4:  # (name, score, triplet_id, equipment)
                    name, score, triplet_id, equipment = symptom_data
                    print(f"      {i+1}. {name} (score: {score:.3f}, triplet_id: {triplet_id}, equipment: {equipment})")
                else:
                    print(f"      {i+1}. {symptom_data}")
        except Exception as e:
            print(f"   ❌ Erreur recherche symptômes: {e}")
            return False
        
        # Test 3: Contexte structuré complet
        print("\n🔍 Test 3: Génération contexte structuré Sparse")
        try:
            context = get_structured_context_sparse(test_query, format_type="compact")
            print(f"   ✅ Contexte généré: {len(context)} caractères")
            if context and not context.startswith("No relevant"):
                lines = context.split('\n')[:3]  # Premières lignes
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
            else:
                print(f"   ⚠️ Aucun contexte pertinent trouvé")
        except Exception as e:
            print(f"   ❌ Erreur génération contexte: {e}")
            return False
        
        print("\n✅ TOUS LES TESTS SPARSE RÉUSSIS!")
        return True
        
    except ImportError as e:
        print(f"❌ Erreur import modules Sparse: {e}")
        print("💡 Vérifiez que sparse_kg_querier.py existe et est correct")
        return False

def get_neo4j_sparse_info():
    """Récupère les informations sur l'instance Neo4j Sparse"""
    print("📋 INFORMATIONS NEO4J SPARSE")
    print("=" * 50)
    print("💡 Vérifications à faire pour Sparse:")
    print("   1. La base Sparse est-elle créée et démarrée? (port 7689)")
    print("   2. Le script build_sparse_knowledge_graph.py a-t-il été exécuté?")
    print("   3. L'index FAISS Sparse a-t-il été créé?")
    print()
    print("🔧 Pour créer la base Sparse:")
    print("   1. python pipeline_step/knowledge_graph_setup/build_sparse_knowledge_graph.py")
    print("   2. python pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_sparse.py")
    print("   3. python pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_sparse.py")
    print()
    print("📁 Fichiers requis:")
    print("   - data/knowledge_base/symptom_embeddings_sparse/index.faiss")
    print("   - data/knowledge_base/symptom_embeddings_sparse/symptom_embedding_sparse.pkl")
    print()
    print("🔍 Caractéristiques Sparse:")
    print("   - Structure 1:1:1 (pas de propagation sémantique)")
    print("   - Préservation des doublons via triplet_id")
    print("   - Relations directes CSV → Neo4j")

if __name__ == "__main__":
    # Test connexion
    working_config = test_neo4j_sparse_connection()
    
    if working_config:
        print(f"🎯 CONFIGURATION SPARSE FONCTIONNELLE:")
        print(f"   NEO4J_URI_SPARSE={working_config['uri']}")
        print(f"   NEO4J_USER_SPARSE={working_config['user']}")
        print(f"   NEO4J_PASS_SPARSE={working_config['password']}")
        
        # Test fonctionnalités
        success = test_sparse_querier_functionality()
        
        if success:
            print("\n🎉 SPARSE KG QUERIER OPÉRATIONNEL!")
        else:
            print("\n⚠️ SPARSE KG QUERIER PARTIELLEMENT FONCTIONNEL")
    else:
        get_neo4j_sparse_info()