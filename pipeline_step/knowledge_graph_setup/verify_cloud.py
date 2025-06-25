#!/usr/bin/env python3
"""
Cloud Knowledge Graph Verification Module

This module provides verification capabilities for the three Cloud Knowledge Graphs
used in the RAG diagnosis system. It validates the proper construction and connectivity
of Dense, Sparse, and Dense S&C Knowledge Graphs.

Key components:
- CloudKGVerifier: Main verification class handling all three KG types
- Connection management: Automated cloud/local fallback logic
- Comprehensive validation: Structure, statistics, and content verification

Dependencies: neo4j, python-dotenv
Usage: docker run --rm -v $(pwd)/.env:/app/.env -v $(pwd)/data:/app/data 
       -v $(pwd)/config:/app/config -v $(pwd)/migrate_to_cloud:/app/migrate_to_cloud 
       --network host diagnosis-app poetry run python pipeline_step/knowledge_graph_setup/verify_cloud.py
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

class CloudKGVerifier:
    """Knowledge Graph Cloud verification system for the RAG diagnosis assistant"""
    
    def __init__(self):
        """
        Initialize the Knowledge Graph verifier
        
        Loads environment variables and initializes connection storage
        """
        load_dotenv()
        self.connections = {}
        
    def get_connection(self, kg_type):
        """
        Establish connection to the specified Knowledge Graph type
        
        Implements cloud-first connection strategy with local fallback.
        Manages connection caching to avoid redundant connections.
        
        Args:
            kg_type (str): Type of Knowledge Graph ('dense', 'sparse', 'dense_sc')
        
        Returns:
            neo4j.Driver: Neo4j database driver instance
            
        Raises:
            Exception: When connection fails for both cloud and local instances
        """
        if kg_type in self.connections:
            return self.connections[kg_type]
            
        # Priority to Cloud if enabled
        cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
        
        if cloud_enabled:
            print(f"Cloud connection {kg_type.upper()}")
            
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
                driver = GraphDatabase.driver(uri, auth=("neo4j", password))
                self.connections[kg_type] = driver
                return driver
            else:
                print(f"Missing cloud credentials for {kg_type}")
                cloud_enabled = False
        
        # Local fallback
        print(f"Local connection {kg_type.upper()}")
        
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
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        self.connections[kg_type] = driver
        return driver
    
    def verify_dense_kg(self):
        """
        Verify the Dense Knowledge Graph structure and content
        
        Performs comprehensive validation of the Dense KG including:
        - Basic connectivity test
        - Node and relationship statistics
        - Equipment distribution analysis
        - Densification verification (symptoms with multiple causes)
        
        Returns:
            bool: True if verification successful, False otherwise
        """
        print("\nDENSE KG VERIFICATION")
        print("-" * 50)
        
        try:
            driver = self.get_connection("dense")
            
            with driver.session() as session:
                # Connection test
                result = session.run("RETURN 'Dense KG Connected!' as message")
                message = result.single()["message"]
                print(f"Connection: {message}")
                
                # Global statistics
                stats_query = """
                RETURN 
                count{(s:Symptom)} as symptoms,
                count{(c:Cause)} as causes, 
                count{(r:Remedy)} as remedies,
                count{()-[:CAUSES]->()} as causes_relations,
                count{()-[:TREATED_BY]->()} as treated_by_relations
                """
                
                result = session.run(stats_query)
                stats = result.single()
                
                print(f"Symptoms: {stats['symptoms']}")
                print(f"Causes: {stats['causes']}")
                print(f"Remedies: {stats['remedies']}")
                print(f"CAUSES relations: {stats['causes_relations']}")
                print(f"TREATED_BY relations: {stats['treated_by_relations']}")
                
                # Equipment verification
                eq_query = """
                MATCH (s:Symptom)
                WHERE s.equipment IS NOT NULL
                RETURN s.equipment as equipment, count(s) as count
                ORDER BY count DESC LIMIT 5
                """
                
                result = session.run(eq_query)
                print("Top equipment types:")
                for record in result:
                    print(f"   • {record['equipment']}: {record['count']}")
                
                # Densification test (verify symptoms have multiple causes)
                density_query = """
                MATCH (s:Symptom)-[:CAUSES]->(c:Cause)
                WITH s, count(c) as cause_count
                WHERE cause_count > 1
                RETURN count(s) as dense_symptoms, avg(cause_count) as avg_causes
                """
                
                result = session.run(density_query)
                density = result.single()
                
                if density['dense_symptoms'] > 0:
                    print(f"Densification detected: {density['dense_symptoms']} symptoms with propagation")
                    print(f"Average causes per dense symptom: {density['avg_causes']:.2f}")
                else:
                    print("Warning: No densification detected (1:1:1 structure)")
                
                return True
                
        except Exception as e:
            print(f"Error in Dense KG: {str(e)}")
            return False
    
    def verify_sparse_kg(self):
        """
        Verify the Sparse Knowledge Graph structure and content
        
        Performs validation of the Sparse KG including:
        - Basic connectivity test
        - Node and relationship statistics
        - 1:1:1 structure verification
        - Triplet ID presence validation
        
        Returns:
            bool: True if verification successful, False otherwise
        """
        print("\nSPARSE KG VERIFICATION")
        print("-" * 50)
        
        try:
            driver = self.get_connection("sparse")
            
            with driver.session() as session:
                # Connection test
                result = session.run("RETURN 'Sparse KG Connected!' as message")
                message = result.single()["message"]
                print(f"Connection: {message}")
                
                # Global statistics
                stats_query = """
                RETURN 
                count{(s:Symptom)} as symptoms,
                count{(c:Cause)} as causes, 
                count{(r:Remedy)} as remedies,
                count{()-[:CAUSES]->()} as causes_relations,
                count{()-[:TREATED_BY]->()} as treated_by_relations
                """
                
                result = session.run(stats_query)
                stats = result.single()
                
                print(f"Symptoms: {stats['symptoms']}")
                print(f"Causes: {stats['causes']}")
                print(f"Remedies: {stats['remedies']}")
                print(f"CAUSES relations: {stats['causes_relations']}")
                print(f"TREATED_BY relations: {stats['treated_by_relations']}")
                
                # 1:1:1 structure verification
                ratio_query = """
                MATCH (s:Symptom)-[:CAUSES]->(c:Cause)-[:TREATED_BY]->(r:Remedy)
                WITH s, count(c) as cause_count, count(r) as remedy_count
                RETURN 
                count(s) as total_symptoms,
                avg(cause_count) as avg_causes,
                avg(remedy_count) as avg_remedies
                """
                
                result = session.run(ratio_query)
                ratios = result.single()
                
                print(f"Average causes per symptom ratio: {ratios['avg_causes']:.2f}")
                print(f"Average remedies per symptom ratio: {ratios['avg_remedies']:.2f}")
                
                if abs(ratios['avg_causes'] - 1.0) < 0.1 and abs(ratios['avg_remedies'] - 1.0) < 0.1:
                    print("1:1:1 structure confirmed")
                else:
                    print("Warning: Non-1:1:1 structure detected")
                
                # Triplet ID verification
                triplet_query = """
                MATCH (s:Symptom)
                WHERE s.triplet_id IS NOT NULL
                RETURN count(s) as symptoms_with_id
                """
                
                result = session.run(triplet_query)
                id_count = result.single()["symptoms_with_id"]
                
                if id_count > 0:
                    print(f"Triplet IDs present: {id_count} symptoms")
                else:
                    print("Warning: No triplet_id found")
                
                return True
                
        except Exception as e:
            print(f"Error in Sparse KG: {str(e)}")
            return False
    
    def verify_dense_sc_kg(self):
        """
        Verify the Dense Symptom & Cause Knowledge Graph structure and content
        
        Performs validation of the Dense S&C KG including:
        - Basic connectivity test
        - Node and relationship statistics
        - Combined text verification (symptom + cause)
        - S&C densification validation
        
        Returns:
            bool: True if verification successful, False otherwise
        """
        print("\nDENSE S&C KG VERIFICATION")
        print("-" * 50)
        
        try:
            driver = self.get_connection("dense_sc")
            
            with driver.session() as session:
                # Connection test
                result = session.run("RETURN 'Dense S&C KG Connected!' as message")
                message = result.single()["message"]
                print(f"Connection: {message}")
                
                # Global statistics
                stats_query = """
                RETURN 
                count{(s:Symptom)} as symptoms,
                count{(c:Cause)} as causes, 
                count{(r:Remedy)} as remedies,
                count{()-[:CAUSES]->()} as causes_relations,
                count{()-[:TREATED_BY]->()} as treated_by_relations
                """
                
                result = session.run(stats_query)
                stats = result.single()
                
                print(f"Symptoms: {stats['symptoms']}")
                print(f"Causes: {stats['causes']}")
                print(f"Remedies: {stats['remedies']}")
                print(f"CAUSES relations: {stats['causes_relations']}")
                print(f"TREATED_BY relations: {stats['treated_by_relations']}")
                
                # Combined S&C text verification
                combined_query = """
                MATCH (s:Symptom)
                WHERE s.combined_text IS NOT NULL
                RETURN count(s) as symptoms_with_combined,
                       s.combined_text as example_text
                LIMIT 1
                """
                
                result = session.run(combined_query)
                record = result.single()
                
                if record and record['symptoms_with_combined'] > 0:
                    print(f"Combined S&C texts: {record['symptoms_with_combined']} symptoms")
                    print(f"Example: {record['example_text'][:100]}...")
                else:
                    print("Warning: No combined S&C text found")
                
                # S&C densification test
                density_query = """
                MATCH (s:Symptom)-[:CAUSES]->(c:Cause)
                WITH s, count(c) as cause_count
                WHERE cause_count > 1
                RETURN count(s) as dense_symptoms, max(cause_count) as max_causes
                """
                
                result = session.run(density_query)
                density = result.single()
                
                if density['dense_symptoms'] > 0:
                    print(f"S&C densification detected: {density['dense_symptoms']} symptoms")
                    print(f"Maximum causes for one symptom: {density['max_causes']}")
                else:
                    print("Warning: No S&C densification detected")
                
                return True
                
        except Exception as e:
            print(f"Error in Dense S&C KG: {str(e)}")
            return False
    
    def close_connections(self):
        """
        Close all active database connections
        
        Safely closes all cached Neo4j driver connections to prevent
        resource leaks and connection pool exhaustion.
        """
        for driver in self.connections.values():
            driver.close()

def main():
    """
    Main verification function for all three Knowledge Graphs
    
    Orchestrates the complete verification process for Dense, Sparse,
    and Dense S&C Knowledge Graphs. Provides comprehensive reporting
    and guidance for next steps based on verification results.
    """
    print("VERIFICATION OF 3 CLOUD KNOWLEDGE GRAPHS")
    print("=" * 70)
    
    verifier = CloudKGVerifier()
    
    try:
        # Verification of all 3 KGs
        results = {
            'dense': verifier.verify_dense_kg(),
            'sparse': verifier.verify_sparse_kg(),
            'dense_sc': verifier.verify_dense_sc_kg()
        }
        
        # Final summary
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        
        for kg_type, success in results.items():
            status = "OPERATIONAL" if success else "PROBLEM DETECTED"
            kg_name = {
                'dense': 'Dense KG (hybrid metric)',
                'sparse': 'Sparse KG (1:1:1 structure)',
                'dense_sc': 'Dense S&C KG (symptom + cause)'
            }[kg_type]
            
            print(f"{status} {kg_name}")
        
        # Global assessment
        all_success = all(results.values())
        
        if all_success:
            print("\nCOMPLETE SUCCESS: All Cloud Knowledge Graphs are operational")
            print("Next step: Proceed to Phase 3 - Index Construction")
        else:
            failed_kgs = [kg for kg, success in results.items() if not success]
            print(f"\nPROBLEMS DETECTED in: {', '.join(failed_kgs)}")
            print("Action required: Verify construction of these KGs before continuing")
        
    except Exception as e:
        print(f"\nGLOBAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        verifier.close_connections()

if __name__ == "__main__":
    main()