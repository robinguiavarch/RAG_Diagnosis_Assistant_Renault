#!/usr/bin/env python3
"""
Script de diagnostic pour la connexion Neo4j Dense S&C
Test du dense_s&c_kg_querier.py
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


def test_neo4j_dense_sc_connection():
    """Test de connexion Neo4j Dense S&C avec différentes configurations"""
    
    print("🔧 DIAGNOSTIC CONNEXION NEO4J DENSE S&C")
    print("=" * 55)
    
    # Affichage des variables d'environnement Dense S&C
    print("📋 Variables d'environnement Dense S&C:")
    print(f"   NEO4J_URI_DENSE_SC: {os.getenv('NEO4J_URI_DENSE_SC', 'NON DÉFINIE')}")
    print(f"   NEO4J_USER_DENSE_SC: {os.getenv('NEO4J_USER_DENSE_SC', 'NON DÉFINIE')}")
    print(f"   NEO4J_PASS_DENSE_SC: {'***' if os.getenv('NEO4J_PASS_DENSE_SC') else 'NON DÉFINIE'}")
    print()
    
    # Test de configuration Dense S&C
    settings = load_settings()
    configurations = [
        {
            "name": "Configuration Dense S&C (settings.yaml)",
            "uri": settings["neo4j"]["dense_sc_uri"],
            "user": settings["neo4j"]["dense_sc_user"],
            "password": settings["neo4j"]["dense_sc_password"]
        },
        {
            "name": "Configuration Dense S&C (variables env)",
            "uri": os.getenv("NEO4J_URI_DENSE_SC", "bolt://host.docker.internal:7690"),
            "user": os.getenv("NEO4J_USER_DENSE_SC", "neo4j"),
            "password": os.getenv("NEO4J_PASS_DENSE_SC", "password")
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
                    
                    # Test de la structure de la base Dense S&C
                    result = session.run("""
                        RETURN 
                        count{(s:Symptom)} as symptoms,
                        count{(c:Cause)} as causes,
                        count{(r:Remedy)} as remedies
                    """)
                    stats = result.single()
                    print(f"   📊 Base de données Dense S&C:")
                    print(f"      Symptômes: {stats['symptoms']}")
                    print(f"      Causes: {stats['causes']}")
                    print(f"      Remèdes: {stats['remedies']}")
                    
                    # 🆕 Test spécifique Dense S&C - texte combiné
                    result = session.run("""
                        MATCH (s:Symptom)
                        WHERE s.combined_text IS NOT NULL
                        RETURN count(s) as symptoms_with_combined,
                               s.combined_text as example_combined
                        LIMIT 1
                    """)
                    sc_stats = result.single()
                    if sc_stats:
                        print(f"   🔗 Spécificité Dense S&C:")
                        print(f"      Symptômes avec texte combiné: {sc_stats['symptoms_with_combined']}")
                        if sc_stats['example_combined']:
                            print(f"      Exemple texte combiné: {sc_stats['example_combined'][:80]}...")
                    
                    driver.close()
                    return config
            
            driver.close()
            
        except Exception as e:
            print(f"   ❌ Échec: {str(e)}")
        
        print()
    
    print("❌ AUCUNE CONFIGURATION DENSE S&C N'A FONCTIONNÉ")
    return None

def test_dense_sc_querier_functionality():
    """Test des fonctions principales du dense_s&c_kg_querier"""
    print("\n🧪 TEST DES FONCTIONS DENSE S&C KG QUERIER")
    print("=" * 55)
    
    try:
        # Import du module à tester
        from core.retrieval_graph.dense_sc_kg_querier import (
            get_structured_context_dense_sc,
            get_similar_symptoms_dense_sc,
            load_symptom_index_dense_sc
        )
        
        print("✅ Import des modules réussi")
        
        # Test 1: Chargement de l'index
        print("\n🔍 Test 1: Chargement index FAISS Dense S&C")
        try:
            index, metadata = load_symptom_index_dense_sc()
            print(f"   ✅ Index chargé: {index.ntotal} vecteurs")
            print(f"   ✅ Métadonnées: {len(metadata.get('symptom_names', []))} symptômes")
            if 'combined_texts' in metadata:
                print(f"   ✅ Textes combinés: {len(metadata['combined_texts'])} éléments")
        except Exception as e:
            print(f"   ❌ Erreur chargement index: {e}")
            return False
        
        # Test 2: Recherche de symptômes similaires
        print("\n🔍 Test 2: Recherche symptômes similaires Dense S&C")
        try:
            test_query = "motor overheating error"
            similar_symptoms = get_similar_symptoms_dense_sc(test_query)
            print(f"   ✅ Requête: '{test_query}'")
            print(f"   ✅ Symptômes trouvés: {len(similar_symptoms)}")
            for i, (symptom, score) in enumerate(similar_symptoms[:3]):
                print(f"      {i+1}. {symptom} (score: {score:.3f})")
        except Exception as e:
            print(f"   ❌ Erreur recherche symptômes: {e}")
            return False
        
        # Test 3: Contexte structuré complet
        print("\n🔍 Test 3: Génération contexte structuré Dense S&C")
        try:
            context = get_structured_context_dense_sc(test_query, format_type="compact")
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
        
        print("\n✅ TOUS LES TESTS DENSE S&C RÉUSSIS!")
        return True
        
    except ImportError as e:
        print(f"❌ Erreur import modules Dense S&C: {e}")
        print("💡 Vérifiez que dense_sc_kg_querier.py existe et est correct")
        return False

def get_neo4j_dense_sc_info():
    """Récupère les informations sur l'instance Neo4j Dense S&C"""
    print("📋 INFORMATIONS NEO4J DENSE S&C")
    print("=" * 50)
    print("💡 Vérifications à faire pour Dense S&C:")
    print("   1. La base Dense S&C est-elle créée et démarrée? (port 7690)")
    print("   2. Le script build_dense_s&c_knowledge_graph.py a-t-il été exécuté?")
    print("   3. L'index FAISS Dense S&C a-t-il été créé?")
    print()
    print("🔧 Pour créer la base Dense S&C:")
    print("   1. python pipeline_step/knowledge_graph_setup/build_dense_s&c_knowledge_graph.py")
    print("   2. python pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_dense_s&c.py")
    print("   3. python pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_dense_s&c.py")
    print()
    print("📁 Fichiers requis:")
    print("   - data/knowledge_base/symptom_embeddings_dense_s&c/index.faiss")
    print("   - data/knowledge_base/symptom_embeddings_dense_s&c/symptom_embedding_dense_s&c.pkl")

if __name__ == "__main__":
    # Test connexion
    working_config = test_neo4j_dense_sc_connection()
    
    if working_config:
        print(f"🎯 CONFIGURATION DENSE S&C FONCTIONNELLE:")
        print(f"   NEO4J_URI_DENSE_SC={working_config['uri']}")
        print(f"   NEO4J_USER_DENSE_SC={working_config['user']}")
        print(f"   NEO4J_PASS_DENSE_SC={working_config['password']}")
        
        # Test fonctionnalités
        success = test_dense_sc_querier_functionality()
        
        if success:
            print("\n🎉 DENSE S&C KG QUERIER OPÉRATIONNEL!")
        else:
            print("\n⚠️ DENSE S&C KG QUERIER PARTIELLEMENT FONCTIONNEL")
    else:
        get_neo4j_dense_sc_info()