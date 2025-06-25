"""
Neo4j Cloud Manager: Cloud and Local Database Connection Management

This module provides unified connection management for Neo4j databases in the RAG
diagnosis system. It supports both cloud and local Neo4j instances with automatic
fallback mechanisms, enabling seamless operation across different deployment environments.

Key components:
- Cloud and local Neo4j driver initialization with authentication
- Environment-based configuration loading and validation
- Automatic fallback from cloud to local connections
- Multi-instance support for different knowledge graph types (dense, sparse, dense_sc)
- Connection testing and validation mechanisms

Dependencies: neo4j, python-dotenv, yaml
Usage: Import functions to establish database connections with automatic cloud/local detection
"""

import os
import yaml
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def load_settings():
    """
    Load configuration settings from YAML file.
    
    Reads the main configuration file for database connection parameters
    and system settings.
    
    Returns:
        dict: Configuration dictionary containing all project settings
        
    Raises:
        FileNotFoundError: If settings.yaml file is not found
        yaml.YAMLError: If YAML file is malformed
    """
    with open("config/settings.yaml", 'r') as f:
        return yaml.safe_load(f)

def is_cloud_enabled():
    """
    Check if cloud Neo4j deployment is enabled via environment variables.
    
    Determines whether the system should attempt cloud connections based
    on the NEO4J_CLOUD_ENABLED environment variable.
    
    Returns:
        bool: True if cloud mode is enabled, False otherwise
    """
    return os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"

def get_dense_driver():
    """
    Establish connection to Dense knowledge graph with detailed debugging.
    
    Attempts to connect to the Dense knowledge graph instance with comprehensive
    error handling and fallback mechanisms. Provides detailed logging for
    troubleshooting connection issues.
    
    Returns:
        neo4j.Driver: Database driver instance for Dense knowledge graph
        
    Raises:
        Exception: If both cloud and local connections fail
    """
    load_dotenv()
    
    print("DEBUG: Starting get_dense_driver()")
    
    # Environment variable verification
    cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
    print(f"DEBUG: NEO4J_CLOUD_ENABLED = {cloud_enabled}")
    
    dense_cloud_uri = os.getenv("NEO4J_DENSE_CLOUD_URI")
    dense_cloud_pass = os.getenv("NEO4J_DENSE_CLOUD_PASS")
    print(f"DEBUG: NEO4J_DENSE_CLOUD_URI = {dense_cloud_uri}")
    print(f"DEBUG: NEO4J_DENSE_CLOUD_PASS = {'***' if dense_cloud_pass else 'EMPTY'}")
    
    try:
        print("DEBUG: Attempting cloud manager import...")
        from core.cloud.neo4j_cloud_manager import get_driver_with_fallback
        print("DEBUG: Cloud manager import successful")
        
        print("DEBUG: Calling get_driver_with_fallback...")
        driver, source = get_driver_with_fallback("dense")
        print(f"DEBUG: Result = source: {source}")
        
        if source == "cloud":
            print("Using Neo4j Cloud Dense")
        else:
            print("Using Neo4j Local Dense")
        return driver
        
    except ImportError as e:
        print(f"DEBUG: ImportError = {e}")
        print("DEBUG: Fallback to direct local connection")
        # Fallback if cloud manager is absent
        db_uri = os.getenv("NEO4J_URI_DENSE", "bolt://host.docker.internal:7687")
        db_user = os.getenv("NEO4J_USER_DENSE", "neo4j")
        db_pass = os.getenv("NEO4J_PASS_DENSE", "password")
        print(f"DEBUG: Local connection: {db_uri}")
        return GraphDatabase.driver(db_uri, auth=(db_user, db_pass))
    except Exception as e:
        print(f"DEBUG: General exception = {e}")
        print("DEBUG: Fallback to direct local connection")
        db_uri = os.getenv("NEO4J_URI_DENSE", "bolt://host.docker.internal:7687")
        db_user = os.getenv("NEO4J_USER_DENSE", "neo4j")
        db_pass = os.getenv("NEO4J_PASS_DENSE", "password")
        return GraphDatabase.driver(db_uri, auth=(db_user, db_pass))

def get_local_driver(kg_type):
    """
    Establish local Neo4j connection based on knowledge graph type.
    
    Creates database driver for local Neo4j instances with type-specific
    connection parameters for different knowledge graph configurations.
    
    Args:
        kg_type (str): Knowledge graph type ('dense', 'sparse', 'dense_sc')
        
    Returns:
        neo4j.Driver: Database driver instance for specified knowledge graph type
        
    Raises:
        ValueError: If unknown knowledge graph type is specified
    """
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
        raise ValueError(f"Unknown KG type: {kg_type}")
    
    return GraphDatabase.driver(uri, auth=(user, password))

def get_driver_with_fallback(kg_type):
    """
    Get database driver with automatic cloud to local fallback.
    
    Attempts cloud connection first if enabled, then falls back to local
    connection if cloud is unavailable. Includes connection testing to
    ensure driver functionality.
    
    Args:
        kg_type (str): Knowledge graph type ('dense', 'sparse', 'dense_sc')
        
    Returns:
        tuple: (neo4j.Driver, str) - Driver instance and source type ('cloud' or 'local')
        
    Raises:
        Exception: If both cloud and local connections fail
    """
    if is_cloud_enabled():
        try:
            cloud_driver = get_cloud_driver(kg_type)
            if cloud_driver:
                # Quick connection test
                with cloud_driver.session() as session:
                    session.run("RETURN 1")
                return cloud_driver, "cloud"
        except Exception:
            pass
    
    # Local fallback
    local_driver = get_local_driver(kg_type)
    return local_driver, "local"