#!/usr/bin/env python3
"""
Script de vérification des 3 Knowledge Graphs Cloud
Vérifie la construction correcte des KG Dense, Sparse et Dense S&C
Pour lancer:
 docker run --rm \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/migrate_to_cloud:/app/migrate_to_cloud \
  --network host \
  diagnosis-app \
  poetry run python pipeline_step/knowledge_graph_setup/verify_cloud.py
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

class CloudKGVerifier:
    """Vérificateur des Knowledge Graphs Cloud"""
    
    def __init__(self):
        load_dotenv()
        self.connections = {}
        
    def get_connection(self, kg_type):
        """Obtient une connexion vers le KG spécifié"""
        if kg_type in self.connections:
            return self.connections[kg_type]
            
        # Priorité au Cloud si activé
        cloud_enabled = os.getenv("NEO4J_CLOUD_ENABLED", "false").lower() == "true"
        
        if cloud_enabled:
            print(f"🌐 Connexion Cloud {kg_type.upper()}")
            
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
                print(f"❌ Credentials cloud manquants pour {kg_type}")
                cloud_enabled = False
        
        # Fallback Local
        print(f"🏠 Connexion Local {kg_type.upper()}")
        
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
        """Vérifie le Knowledge Graph Dense"""
        print("\n🔍 VÉRIFICATION KG DENSE")
        print("-" * 50)
        
        try:
            driver = self.get_connection("dense")
            
            with driver.session() as session:
                # Test connexion
                result = session.run("RETURN 'Dense KG Connected!' as message")
                message = result.single()["message"]
                print(f"✅ Connexion: {message}")
                
                # Statistiques globales
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
                
                print(f"📊 Symptômes: {stats['symptoms']}")
                print(f"📊 Causes: {stats['causes']}")
                print(f"📊 Remèdes: {stats['remedies']}")
                print(f"📊 Relations CAUSES: {stats['causes_relations']}")
                print(f"📊 Relations TREATED_BY: {stats['treated_by_relations']}")
                
                # Vérification equipment
                eq_query = """
                MATCH (s:Symptom)
                WHERE s.equipment IS NOT NULL
                RETURN s.equipment as equipment, count(s) as count
                ORDER BY count DESC LIMIT 5
                """
                
                result = session.run(eq_query)
                print("🏭 Top équipements:")
                for record in result:
                    print(f"   • {record['equipment']}: {record['count']}")
                
                # Test densification (vérification qu'un symptôme a plusieurs causes)
                density_query = """
                MATCH (s:Symptom)-[:CAUSES]->(c:Cause)
                WITH s, count(c) as cause_count
                WHERE cause_count > 1
                RETURN count(s) as dense_symptoms, avg(cause_count) as avg_causes
                """
                
                result = session.run(density_query)
                density = result.single()
                
                if density['dense_symptoms'] > 0:
                    print(f"✅ Densification détectée: {density['dense_symptoms']} symptômes avec propagation")
                    print(f"📈 Moyenne causes par symptôme dense: {density['avg_causes']:.2f}")
                else:
                    print("⚠️  Aucune densification détectée (structure 1:1:1)")
                
                return True
                
        except Exception as e:
            print(f"❌ Erreur KG Dense: {str(e)}")
            return False
    
    def verify_sparse_kg(self):
        """Vérifie le Knowledge Graph Sparse"""
        print("\n🔍 VÉRIFICATION KG SPARSE")
        print("-" * 50)
        
        try:
            driver = self.get_connection("sparse")
            
            with driver.session() as session:
                # Test connexion
                result = session.run("RETURN 'Sparse KG Connected!' as message")
                message = result.single()["message"]
                print(f"✅ Connexion: {message}")
                
                # Statistiques globales
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
                
                print(f"📊 Symptômes: {stats['symptoms']}")
                print(f"📊 Causes: {stats['causes']}")
                print(f"📊 Remèdes: {stats['remedies']}")
                print(f"📊 Relations CAUSES: {stats['causes_relations']}")
                print(f"📊 Relations TREATED_BY: {stats['treated_by_relations']}")
                
                # Vérification structure 1:1:1
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
                
                print(f"📈 Ratio moyen causes/symptôme: {ratios['avg_causes']:.2f}")
                print(f"📈 Ratio moyen remèdes/symptôme: {ratios['avg_remedies']:.2f}")
                
                if abs(ratios['avg_causes'] - 1.0) < 0.1 and abs(ratios['avg_remedies'] - 1.0) < 0.1:
                    print("✅ Structure 1:1:1 confirmée")
                else:
                    print("⚠️  Structure non-1:1:1 détectée")
                
                # Vérification triplet_id
                triplet_query = """
                MATCH (s:Symptom)
                WHERE s.triplet_id IS NOT NULL
                RETURN count(s) as symptoms_with_id
                """
                
                result = session.run(triplet_query)
                id_count = result.single()["symptoms_with_id"]
                
                if id_count > 0:
                    print(f"✅ Triplet IDs présents: {id_count} symptômes")
                else:
                    print("⚠️  Aucun triplet_id trouvé")
                
                return True
                
        except Exception as e:
            print(f"❌ Erreur KG Sparse: {str(e)}")
            return False
    
    def verify_dense_sc_kg(self):
        """Vérifie le Knowledge Graph Dense S&C"""
        print("\n🔍 VÉRIFICATION KG DENSE S&C")
        print("-" * 50)
        
        try:
            driver = self.get_connection("dense_sc")
            
            with driver.session() as session:
                # Test connexion
                result = session.run("RETURN 'Dense S&C KG Connected!' as message")
                message = result.single()["message"]
                print(f"✅ Connexion: {message}")
                
                # Statistiques globales
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
                
                print(f"📊 Symptômes: {stats['symptoms']}")
                print(f"📊 Causes: {stats['causes']}")
                print(f"📊 Remèdes: {stats['remedies']}")
                print(f"📊 Relations CAUSES: {stats['causes_relations']}")
                print(f"📊 Relations TREATED_BY: {stats['treated_by_relations']}")
                
                # Vérification textes combinés S&C
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
                    print(f"✅ Textes combinés S&C: {record['symptoms_with_combined']} symptômes")
                    print(f"📝 Exemple: {record['example_text'][:100]}...")
                else:
                    print("⚠️  Aucun texte combiné S&C trouvé")
                
                # Test densification S&C
                density_query = """
                MATCH (s:Symptom)-[:CAUSES]->(c:Cause)
                WITH s, count(c) as cause_count
                WHERE cause_count > 1
                RETURN count(s) as dense_symptoms, max(cause_count) as max_causes
                """
                
                result = session.run(density_query)
                density = result.single()
                
                if density['dense_symptoms'] > 0:
                    print(f"✅ Densification S&C détectée: {density['dense_symptoms']} symptômes")
                    print(f"📈 Max causes pour un symptôme: {density['max_causes']}")
                else:
                    print("⚠️  Aucune densification S&C détectée")
                
                return True
                
        except Exception as e:
            print(f"❌ Erreur KG Dense S&C: {str(e)}")
            return False
    
    def close_connections(self):
        """Ferme toutes les connexions"""
        for driver in self.connections.values():
            driver.close()

def main():
    """Fonction principale de vérification"""
    print("🚀 VÉRIFICATION DES 3 KNOWLEDGE GRAPHS CLOUD")
    print("=" * 70)
    
    verifier = CloudKGVerifier()
    
    try:
        # Vérification des 3 KG
        results = {
            'dense': verifier.verify_dense_kg(),
            'sparse': verifier.verify_sparse_kg(),
            'dense_sc': verifier.verify_dense_sc_kg()
        }
        
        # Résumé final
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ DE LA VÉRIFICATION")
        print("=" * 70)
        
        for kg_type, success in results.items():
            status = "✅ OPÉRATIONNEL" if success else "❌ PROBLÈME"
            kg_name = {
                'dense': 'KG Dense (métrique hybride)',
                'sparse': 'KG Sparse (structure 1:1:1)',
                'dense_sc': 'KG Dense S&C (symptôme + cause)'
            }[kg_type]
            
            print(f"{status} {kg_name}")
        
        # Bilan global
        all_success = all(results.values())
        
        if all_success:
            print("\n🎉 SUCCÈS COMPLET ! Tous les Knowledge Graphs Cloud sont opérationnels !")
            print("🔄 Vous pouvez passer à la Phase 3 : Construction des Index")
        else:
            failed_kgs = [kg for kg, success in results.items() if not success]
            print(f"\n⚠️  PROBLÈMES DÉTECTÉS sur : {', '.join(failed_kgs)}")
            print("🔧 Vérifiez la construction de ces KG avant de continuer")
        
    except Exception as e:
        print(f"\n❌ ERREUR GLOBALE: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        verifier.close_connections()

if __name__ == "__main__":
    main()