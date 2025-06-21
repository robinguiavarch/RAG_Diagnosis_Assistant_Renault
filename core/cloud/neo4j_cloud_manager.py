"""
Manager simple pour Neo4j Cloud
Path: core/cloud/neo4j_cloud_manager.py
"""

import os
import yaml
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def load_settings():
    """Charge config"""
    with open("config/settings.yaml", 'r') as f:
        return yaml.safe_load(f)

def is_cloud_enabled():
    """Check si cloud activé"""
    return os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"


def get_dense_driver():
    """Connexion Dense avec debug détaillé"""
    load_dotenv()
    
    print("🔍 DEBUG: Début get_dense_driver()")
    
    # Vérification des variables d'environnement
    cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
    print(f"🔍 DEBUG: NEO4J_CLOUD_ENABLED = {cloud_enabled}")
    
    dense_cloud_uri = os.getenv("NEO4J_DENSE_CLOUD_URI")
    dense_cloud_pass = os.getenv("NEO4J_DENSE_CLOUD_PASS")
    print(f"🔍 DEBUG: NEO4J_DENSE_CLOUD_URI = {dense_cloud_uri}")
    print(f"🔍 DEBUG: NEO4J_DENSE_CLOUD_PASS = {'***' if dense_cloud_pass else 'VIDE'}")
    
    try:
        print("🔍 DEBUG: Tentative import cloud manager...")
        from core.cloud.neo4j_cloud_manager import get_driver_with_fallback
        print("🔍 DEBUG: Import cloud manager réussi")
        
        print("🔍 DEBUG: Appel get_driver_with_fallback...")
        driver, source = get_driver_with_fallback("dense")
        print(f"🔍 DEBUG: Résultat = source: {source}")
        
        if source == "cloud":
            print("🌐 Utilisation Neo4j Cloud Dense")
        else:
            print("🏠 Utilisation Neo4j Local Dense")
        return driver
        
    except ImportError as e:
        print(f"🔍 DEBUG: ImportError = {e}")
        print("🔍 DEBUG: Fallback vers connexion locale directe")
        # Fallback si cloud manager absent
        db_uri = os.getenv("NEO4J_URI_DENSE", "bolt://host.docker.internal:7687")
        db_user = os.getenv("NEO4J_USER_DENSE", "neo4j")
        db_pass = os.getenv("NEO4J_PASS_DENSE", "password")
        print(f"🔍 DEBUG: Connexion locale: {db_uri}")
        return GraphDatabase.driver(db_uri, auth=(db_user, db_pass))
    except Exception as e:
        print(f"🔍 DEBUG: Exception générale = {e}")
        print("🔍 DEBUG: Fallback vers connexion locale directe")
        db_uri = os.getenv("NEO4J_URI_DENSE", "bolt://host.docker.internal:7687")
        db_user = os.getenv("NEO4J_USER_DENSE", "neo4j")
        db_pass = os.getenv("NEO4J_PASS_DENSE", "password")
        return GraphDatabase.driver(db_uri, auth=(db_user, db_pass))

def get_local_driver(kg_type):
    """Connexion locale selon type KG"""
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
    else:
        raise ValueError(f"Type KG inconnu: {kg_type}")
    
    return GraphDatabase.driver(uri, auth=(user, password))

def get_driver_with_fallback(kg_type):
    """Driver avec fallback cloud→local"""
    if is_cloud_enabled():
        try:
            cloud_driver = get_cloud_driver(kg_type)
            if cloud_driver:
                # Test rapide connexion
                with cloud_driver.session() as session:
                    session.run("RETURN 1")
                return cloud_driver, "cloud"
        except Exception:
            pass
    
    # Fallback local
    local_driver = get_local_driver(kg_type)
    return local_driver, "local"