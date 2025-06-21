#!/usr/bin/env python3
"""
Script de test simple pour vérifier les connexions Neo4j Cloud
Teste la première instance Dense créée
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

def test_dense_cloud_connection():
    """Test de connexion à l'instance Dense Cloud"""
    print("🔍 Test connexion Dense Cloud...")
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Récupérer les credentials
    uri = os.getenv('NEO4J_DENSE_CLOUD_URI')
    password = os.getenv('NEO4J_DENSE_CLOUD_PASS')
    username = 'neo4j'
    
    if not uri or not password:
        print("❌ Variables d'environnement manquantes !")
        print(f"URI: {uri}")
        print(f"Password: {'***' if password else 'MANQUANT'}")
        return False
    
    try:
        # Test de connexion
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test d'une requête simple
        with driver.session() as session:
            result = session.run("RETURN 'Hello Neo4j Cloud!' as message")
            record = result.single()
            message = record["message"]
            
            print(f"✅ Dense Cloud: CONNECTÉ !")
            print(f"   📍 URI: {uri}")
            print(f"   💬 Réponse: {message}")
            
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Dense Cloud: ÉCHEC de connexion")
        print(f"   📍 URI: {uri}")
        print(f"   🚨 Erreur: {str(e)}")
        return False

def main():
    """Fonction principale de test"""
    print("=" * 60)
    print("🚀 TEST CONNEXIONS NEO4J CLOUD")
    print("=" * 60)
    
    # Test Dense Cloud
    dense_ok = test_dense_cloud_connection()
    
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS")
    print("=" * 60)
    
    if dense_ok:
        print("✅ Dense Cloud: OPÉRATIONNEL")
        print("\n🎉 SUCCÈS ! Votre première instance cloud fonctionne !")
        print("🔄 Vous pouvez maintenant créer les 2 autres instances.")
    else:
        print("❌ Dense Cloud: PROBLÈME")
        print("\n🔧 Vérifiez vos credentials dans le .env")

if __name__ == "__main__":
    main()