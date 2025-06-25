#!/usr/bin/env python3
"""
Dense S&C Knowledge Graph Querier Diagnostic Script

This module provides comprehensive diagnostic capabilities for the Dense Symptom & Cause
Knowledge Graph querier functionality. It validates Neo4j connections, tests index loading,
symptom similarity search, and structured context generation specific to the Dense S&C
configuration with combined symptom and cause text processing.

Key components:
- Connection validation: Tests Neo4j Dense S&C database connectivity with fallback configurations
- Index verification: Validates FAISS index loading and metadata for combined text processing
- Functionality testing: Tests symptom similarity search and context generation capabilities
- Diagnostic reporting: Provides detailed feedback and troubleshooting guidance

Dependencies: neo4j, python-dotenv, pyyaml, pathlib
Usage: Execute as standalone diagnostic script to validate Dense S&C KG querier functionality
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import sys
from pathlib import Path
import yaml

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_settings():
    """
    Load configuration from settings.yaml file
    
    Returns:
        dict: Loaded configuration settings
    """
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_neo4j_dense_sc_connection():
    """
    Test Neo4j Dense S&C connection with different configurations
    
    Performs comprehensive connection testing using both settings.yaml and environment
    variable configurations. Validates database connectivity and analyzes Dense S&C
    specific structure including combined text properties.
    
    Returns:
        dict or None: Working configuration if successful, None if all tests fail
    """
    
    print("NEO4J DENSE S&C CONNECTION DIAGNOSTICS")
    print("=" * 55)
    
    # Display Dense S&C environment variables
    print("Dense S&C environment variables:")
    print(f"   NEO4J_URI_DENSE_SC: {os.getenv('NEO4J_URI_DENSE_SC', 'NOT DEFINED')}")
    print(f"   NEO4J_USER_DENSE_SC: {os.getenv('NEO4J_USER_DENSE_SC', 'NOT DEFINED')}")
    print(f"   NEO4J_PASS_DENSE_SC: {'***' if os.getenv('NEO4J_PASS_DENSE_SC') else 'NOT DEFINED'}")
    print()
    
    # Test Dense S&C configurations
    settings = load_settings()
    configurations = [
        {
            "name": "Dense S&C Configuration (settings.yaml)",
            "uri": settings["neo4j"]["dense_sc_uri"],
            "user": settings["neo4j"]["dense_sc_user"],
            "password": settings["neo4j"]["dense_sc_password"]
        },
        {
            "name": "Dense S&C Configuration (environment variables)",
            "uri": os.getenv("NEO4J_URI_DENSE_SC", "bolt://host.docker.internal:7690"),
            "user": os.getenv("NEO4J_USER_DENSE_SC", "neo4j"),
            "password": os.getenv("NEO4J_PASS_DENSE_SC", "password")
        }
    ]
    
    for config in configurations:
        print(f"Testing: {config['name']}")
        print(f"   URI: {config['uri']}")
        print(f"   User: {config['user']}")
        print(f"   Password: {'***' if config['password'] else '(empty)'}")
        
        try:
            driver = GraphDatabase.driver(
                config['uri'], 
                auth=(config['user'], config['password'])
            )
            
            # Connection test
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    print("   CONNECTION SUCCESSFUL")
                    
                    # Test Dense S&C database structure
                    result = session.run("""
                        RETURN 
                        count{(s:Symptom)} as symptoms,
                        count{(c:Cause)} as causes,
                        count{(r:Remedy)} as remedies
                    """)
                    stats = result.single()
                    print(f"   Dense S&C database:")
                    print(f"      Symptoms: {stats['symptoms']}")
                    print(f"      Causes: {stats['causes']}")
                    print(f"      Remedies: {stats['remedies']}")
                    
                    # Dense S&C specific test - combined text
                    result = session.run("""
                        MATCH (s:Symptom)
                        WHERE s.combined_text IS NOT NULL
                        RETURN count(s) as symptoms_with_combined,
                               s.combined_text as example_combined
                        LIMIT 1
                    """)
                    sc_stats = result.single()
                    if sc_stats:
                        print(f"   Dense S&C specifics:")
                        print(f"      Symptoms with combined text: {sc_stats['symptoms_with_combined']}")
                        if sc_stats['example_combined']:
                            print(f"      Example combined text: {sc_stats['example_combined'][:80]}...")
                    
                    driver.close()
                    return config
            
            driver.close()
            
        except Exception as e:
            print(f"   FAILED: {str(e)}")
        
        print()
    
    print("NO DENSE S&C CONFIGURATION WORKED")
    return None

def test_dense_sc_querier_functionality():
    """
    Test core functionality of the Dense S&C KG querier
    
    Performs comprehensive testing of the Dense S&C querier including index loading,
    symptom similarity search with combined text processing, and structured context
    generation. Validates the complete pipeline functionality.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\nDENSE S&C KG QUERIER FUNCTIONALITY TESTS")
    print("=" * 55)
    
    try:
        # Import module to test
        from core.retrieval_graph.dense_sc_kg_querier import (
            get_structured_context_dense_sc,
            get_similar_symptoms_dense_sc,
            load_symptom_index_dense_sc
        )
        
        print("Module import successful")
        
        # Test 1: Index loading
        print("\nTest 1: Dense S&C FAISS index loading")
        try:
            index, metadata = load_symptom_index_dense_sc()
            print(f"   Index loaded: {index.ntotal} vectors")
            print(f"   Metadata: {len(metadata.get('symptom_names', []))} symptoms")
            if 'combined_texts' in metadata:
                print(f"   Combined texts: {len(metadata['combined_texts'])} elements")
        except Exception as e:
            print(f"   Index loading error: {e}")
            return False
        
        # Test 2: Similar symptom search
        print("\nTest 2: Dense S&C similar symptom search")
        try:
            test_query = "motor overheating error"
            similar_symptoms = get_similar_symptoms_dense_sc(test_query)
            print(f"   Query: '{test_query}'")
            print(f"   Symptoms found: {len(similar_symptoms)}")
            for i, (symptom, score) in enumerate(similar_symptoms[:3]):
                print(f"      {i+1}. {symptom} (score: {score:.3f})")
        except Exception as e:
            print(f"   Symptom search error: {e}")
            return False
        
        # Test 3: Complete structured context
        print("\nTest 3: Dense S&C structured context generation")
        try:
            context = get_structured_context_dense_sc(test_query, format_type="compact")
            print(f"   Context generated: {len(context)} characters")
            if context and not context.startswith("No relevant"):
                lines = context.split('\n')[:3]  # First lines
                for line in lines:
                    if line.strip():
                        print(f"      {line}")
            else:
                print(f"   Warning: No relevant context found")
        except Exception as e:
            print(f"   Context generation error: {e}")
            return False
        
        print("\nALL DENSE S&C TESTS PASSED")
        return True
        
    except ImportError as e:
        print(f"Dense S&C module import error: {e}")
        print("Note: Verify that dense_sc_kg_querier.py exists and is correct")
        return False

def get_neo4j_dense_sc_info():
    """
    Display information about Neo4j Dense S&C instance requirements
    
    Provides comprehensive guidance on Dense S&C setup requirements, including
    database creation, index building, and file dependencies for proper
    Dense S&C KG querier functionality.
    """
    print("NEO4J DENSE S&C INFORMATION")
    print("=" * 50)
    print("Required verifications for Dense S&C:")
    print("   1. Is Dense S&C database created and started (port 7690)")
    print("   2. Has build_dense_s&c_knowledge_graph.py been executed")
    print("   3. Has Dense S&C FAISS index been created")
    print()
    print("To create Dense S&C database:")
    print("   1. python pipeline_step/knowledge_graph_setup/build_dense_s&c_knowledge_graph.py")
    print("   2. python pipeline_step/knowledge_graph_setup/build_symptom_vector_index_kg_dense_s&c.py")
    print("   3. python pipeline_step/knowledge_graph_setup/build_symptom_bm25_index_kg_dense_s&c.py")
    print()
    print("Required files:")
    print("   - data/knowledge_base/symptom_embeddings_dense_s&c/index.faiss")
    print("   - data/knowledge_base/symptom_embeddings_dense_s&c/symptom_embedding_dense_s&c.pkl")

def main():
    """
    Main diagnostic pipeline for Dense S&C KG querier validation
    
    Orchestrates the complete diagnostic process including connection testing,
    functionality validation, and reporting. Provides comprehensive feedback
    on Dense S&C KG querier status and troubleshooting guidance.
    """
    # Connection test
    working_config = test_neo4j_dense_sc_connection()
    
    if working_config:
        print(f"FUNCTIONAL DENSE S&C CONFIGURATION:")
        print(f"   NEO4J_URI_DENSE_SC={working_config['uri']}")
        print(f"   NEO4J_USER_DENSE_SC={working_config['user']}")
        print(f"   NEO4J_PASS_DENSE_SC={working_config['password']}")
        
        # Functionality tests
        success = test_dense_sc_querier_functionality()
        
        if success:
            print("\nDENSE S&C KG QUERIER OPERATIONAL")
        else:
            print("\nDENSE S&C KG QUERIER PARTIALLY FUNCTIONAL")
    else:
        get_neo4j_dense_sc_info()

if __name__ == "__main__":
    main()