"""
Script simple pour lancer tous les tests query_processing
"""

import subprocess
import sys
import os

def run_tests():
    """Lance tous les tests du module query_processing"""
    
    # Chemin vers le dossier des tests
    test_dir = os.path.dirname(__file__)
    
    # Liste des fichiers de test
    test_files = [
        "test_llm_client.py",
        "test_response_parser.py", 
        "test_unified_processor.py",
        "test_enhanced_retrieval.py",
        "test_integration.py"
    ]
    
    print("🧪 Lancement des tests query_processing")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    failed_files = []
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        
        if not os.path.exists(test_path):
            print(f"⚠️ Fichier manquant: {test_file}")
            continue
        
        print(f"\n📋 {test_file}")
        print("-" * 30)
        
        try:
            # Lancer pytest pour ce fichier
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_path, "-v"
            ], capture_output=True, text=True, cwd=os.path.join(test_dir, "..", ".."))
            
            if result.returncode == 0:
                print("✅ RÉUSSI")
                # Compter les tests (approximatif)
                test_count = result.stdout.count("PASSED")
                total_tests += test_count
                passed_tests += test_count
            else:
                print("❌ ÉCHEC")
                failed_files.append(test_file)
                print(f"Erreur: {result.stderr}")
                
        except Exception as e:
            print(f"❌ ERREUR: {e}")
            failed_files.append(test_file)
    
    # Résumé final
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ FINAL")
    print("=" * 50)
    print(f"✅ Tests réussis: {passed_tests}")
    print(f"📁 Fichiers testés: {len(test_files) - len(failed_files)}/{len(test_files)}")
    
    if failed_files:
        print(f"❌ Fichiers échoués: {', '.join(failed_files)}")
        return False
    else:
        print("🎉 TOUS LES TESTS SONT PASSÉS !")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)