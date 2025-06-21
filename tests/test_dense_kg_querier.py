#!/usr/bin/env python3
"""
Script de diagnostic pour la connexion Neo4j
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


def test_neo4j_connection():
    """Test de connexion Neo4j avec différentes configurations"""
    
    print("🔧 DIAGNOSTIC CONNEXION NEO4J")
    print("=" * 50)
    
    # Affichage des variables d'environnement
    print("📋 Variables d'environnement:")
    print(f"   NEO4J_URI_DENSE: {os.getenv('NEO4J_URI_DENSE', 'NON DÉFINIE')}")
    print(f"   NEO4J_USER_DENSE: {os.getenv('NEO4J_USER_DENSE', 'NON DÉFINIE')}")
    print(f"   NEO4J_PASS_DENSE: {'***' if os.getenv('NEO4J_PASS_DENSE') else 'NON DÉFINIE'}")
    print(f"   NEO4J_URI_SPARSE: {os.getenv('NEO4J_URI_SPARSE', 'NON DÉFINIE')}")
    print(f"   NEO4J_USER_SPARSE: {os.getenv('NEO4J_USER_SPARSE', 'NON DÉFINIE')}")
    print(f"   NEO4J_PASS_SPARSE: {'***' if os.getenv('NEO4J_PASS_SPARSE') else 'NON DÉFINIE'}")
    print()
    
    # Test 1: Configuration par défaut
    settings = load_settings()
    configurations = [
        {
            "name": "Configuration Dense (settings.yaml)",
            "uri": settings["neo4j"]["dense_uri"],
            "user": settings["neo4j"]["dense_user"],
            "password": settings["neo4j"]["dense_password"]
        },
        {
            "name": "Configuration Sparse (settings.yaml)",
            "uri": settings["neo4j"]["sparse_uri"],
            "user": settings["neo4j"]["sparse_user"],
            "password": settings["neo4j"]["sparse_password"]
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
                    
                    # Test de la structure de la base
                    result = session.run("""
                        RETURN 
                        count{(s:Symptom)} as symptoms,
                        count{(c:Cause)} as causes,
                        count{(r:Remedy)} as remedies
                    """)
                    stats = result.single()
                    print(f"   📊 Base de données:")
                    print(f"      Symptômes: {stats['symptoms']}")
                    print(f"      Causes: {stats['causes']}")
                    print(f"      Remèdes: {stats['remedies']}")
                    
                    driver.close()
                    return config
            
            driver.close()
            
        except Exception as e:
            print(f"   ❌ Échec: {str(e)}")
        
        print()
    
    print("❌ AUCUNE CONFIGURATION N'A FONCTIONNÉ")
    return None

def get_neo4j_info():
    """Récupère les informations sur l'instance Neo4j"""
    print("📋 INFORMATIONS NEO4J DESKTOP")
    print("=" * 50)
    print("💡 Vérifications à faire dans Neo4j Desktop:")
    print("   1. La base 'SCR-KnowledgeGraph' est-elle démarrée?")
    print("   2. Quel est le mot de passe défini?")
    print("   3. Les ports sont-ils corrects?")
    print()
    print("🔧 Pour réinitialiser le mot de passe:")
    print("   1. Arrêtez la base dans Neo4j Desktop")
    print("   2. Clic droit → 'Reset password'")
    print("   3. Redémarrez la base")
    print()
    print("🔍 Pour voir les informations de connexion:")
    print("   1. Cliquez sur votre base dans Neo4j Desktop")
    print("   2. Onglet 'Details' → Connection details")

if __name__ == "__main__":
    working_config = test_neo4j_connection()
    
    if working_config:
        print(f"🎯 CONFIGURATION FONCTIONNELLE TROUVÉE:")
        print(f"   Ajoutez ceci dans votre fichier .env:")
        print(f"   NEO4J_URI={working_config['uri']}")
        print(f"   NEO4J_USER={working_config['user']}")
        print(f"   NEO4J_PASS={working_config['password']}")
    else:
        get_neo4j_info()