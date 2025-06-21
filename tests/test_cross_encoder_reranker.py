#!/usr/bin/env python3
"""
Test du système de re-ranking CrossEncoder
Teste l'initialisation, le re-ranking et les performances du CrossEncoder

IMPORTANT: Pour run le test:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 poetry run python tests/test_cross_encoder.py > tests/test_reports/test_cross_encoder.txt 2>&1
"""

import sys
import time
from pathlib import Path
import yaml
from typing import Dict, Any, List
import numpy as np

# Ajouter le répertoire racine au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

# Import optionnel du reranker
try:
    from core.reranking_engine.cross_encoder_reranker import CrossEncoderReranker
    RERANKER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CrossEncoderReranker non disponible: {e}")
    RERANKER_AVAILABLE = False
    CrossEncoderReranker = None


def print_separator(title: str, char: str = "="):
    """Affiche un séparateur avec titre"""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """Charge la configuration depuis le fichier YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️ Erreur chargement config: {e}")
        return {
            "models": {
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
            }
        }


def test_reranker_availability():
    """Test de disponibilité du module reranker"""
    print_separator("TEST DE DISPONIBILITÉ")
    
    if RERANKER_AVAILABLE:
        print("✅ Module CrossEncoderReranker disponible")
        return True
    else:
        print("❌ Module CrossEncoderReranker non disponible")
        print("💡 Vérifiez l'installation: poetry add sentence-transformers torch")
        return False


def test_reranker_initialization():
    """Test d'initialisation du CrossEncoder"""
    print_separator("TEST D'INITIALISATION")
    
    if not RERANKER_AVAILABLE:
        print("⏭️ Skip - module non disponible")
        return False
    
    # Test avec modèle par défaut
    print("🔄 Test initialisation modèle par défaut...")
    try:
        start_time = time.time()
        reranker = CrossEncoderReranker()
        init_time = time.time() - start_time
        
        print(f"✅ Initialisation réussie ({init_time:.2f}s)")
        
        # Vérifier les infos du modèle
        model_info = reranker.get_model_info()
        print(f"📊 Infos modèle:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        return True, reranker
        
    except Exception as e:
        print(f"❌ Erreur initialisation: {e}")
        return False, None


def test_reranker_with_custom_model():
    """Test d'initialisation avec modèle custom"""
    print_separator("TEST MODÈLE CUSTOM")
    
    if not RERANKER_AVAILABLE:
        print("⏭️ Skip - module non disponible")
        return False
    
    # Test avec modèle léger pour rapidité
    print("🔄 Test avec modèle léger...")
    try:
        settings = load_settings()
        model_name = settings["models"]["reranker_model"]
        
        start_time = time.time()
        reranker = CrossEncoderReranker(model_name=model_name)
        init_time = time.time() - start_time
        
        print(f"✅ Modèle {model_name} chargé ({init_time:.2f}s)")
        return True, reranker
        
    except Exception as e:
        print(f"❌ Erreur modèle custom: {e}")
        return False, None


def create_test_candidates() -> List[Dict[str, Any]]:
    """Crée des candidats de test simulant les résultats de fusion"""
    return [
        {
            "document_id": "fanuc_manual",
            "chunk_id": "error_001", 
            "text": "ACAL-006 TPE operation error occurs when the teach pendant encounters a communication issue with the controller. Check the cable connections.",
            "fused_score": 0.85,
            "bm25_score": 0.90,
            "faiss_score": 0.80
        },
        {
            "document_id": "fanuc_manual",
            "chunk_id": "error_002",
            "text": "Robot calibration procedures must be followed exactly. Improper calibration can lead to positioning errors and operational failures.",
            "fused_score": 0.75,
            "bm25_score": 0.70,
            "faiss_score": 0.80
        },
        {
            "document_id": "technical_guide",
            "chunk_id": "troubleshoot_001",
            "text": "When troubleshooting FANUC robots, first check the error code display on the teach pendant. Common errors include communication and calibration issues.",
            "fused_score": 0.70,
            "bm25_score": 0.75,
            "faiss_score": 0.65
        },
        {
            "document_id": "safety_manual", 
            "chunk_id": "safety_001",
            "text": "Safety procedures require proper shutdown before maintenance. Always disconnect power and follow lockout procedures.",
            "fused_score": 0.45,
            "bm25_score": 0.40,
            "faiss_score": 0.50
        },
        {
            "document_id": "installation_guide",
            "chunk_id": "install_001",
            "text": "Installation of robotic systems requires careful planning and adherence to manufacturer specifications for optimal performance.",
            "fused_score": 0.35,
            "bm25_score": 0.30,
            "faiss_score": 0.40
        }
    ]


def test_basic_reranking():
    """Test de re-ranking basique"""
    print_separator("TEST RE-RANKING BASIQUE")
    
    if not RERANKER_AVAILABLE:
        print("⏭️ Skip - module non disponible")
        return False
    
    try:
        # Initialisation
        reranker = CrossEncoderReranker()
        print("✅ Reranker initialisé")
        
        # Données de test
        query = "ACAL-006 error on FANUC robot teach pendant"
        candidates = create_test_candidates()
        
        print(f"🔍 Requête: \"{query}\"")
        print(f"📊 Candidats: {len(candidates)}")
        
        # Affichage avant re-ranking
        print(f"\n📋 AVANT RE-RANKING (tri par score fusion):")
        for i, candidate in enumerate(candidates):
            print(f"   {i+1}. Score: {candidate['fused_score']:.3f} | {candidate['document_id']}|{candidate['chunk_id']}")
            print(f"      {candidate['text'][:80]}...")
        
        # Re-ranking
        print(f"\n🎯 Re-ranking avec CrossEncoder...")
        start_time = time.time()
        reranked = reranker.rerank(query, candidates, top_k=len(candidates))
        rerank_time = time.time() - start_time
        
        print(f"✅ Re-ranking terminé ({rerank_time:.2f}s)")
        
        # Affichage après re-ranking
        print(f"\n🏆 APRÈS RE-RANKING (tri par CrossEncoder):")
        for i, result in enumerate(reranked):
            cross_score = result.get('cross_encoder_score', 0)
            fusion_score = result.get('fused_score', 0)
            original_rank = result.get('original_rank', '?')
            
            print(f"   {i+1}. CrossEncoder: {cross_score:.3f} | Fusion: {fusion_score:.3f} | Rang orig: #{original_rank}")
            print(f"      {result['document_id']}|{result['chunk_id']}")
            print(f"      {result['text'][:80]}...")
        
        # Analyse des changements
        print(f"\n📊 ANALYSE DES CHANGEMENTS:")
        
        # Ordre original vs re-ranké
        original_order = [(c['document_id'], c['chunk_id']) for c in candidates]
        reranked_order = [(r['document_id'], r['chunk_id']) for r in reranked]
        
        changes = 0
        for i, (orig, rerank) in enumerate(zip(original_order, reranked_order)):
            if orig != rerank:
                changes += 1
        
        print(f"   🔄 Positions changées: {changes}/{len(candidates)}")
        print(f"   ⏱️ Temps par document: {(rerank_time/len(candidates)*1000):.1f}ms")
        
        # Vérifier que le premier résultat est plus pertinent
        if reranked:
            best_result = reranked[0]
            if "ACAL-006" in best_result['text'] and "TPE" in best_result['text']:
                print(f"   ✅ Le meilleur résultat contient les termes clés de la requête")
            else:
                print(f"   ⚠️ Le meilleur résultat ne semble pas optimal")
        
        return True, reranked
        
    except Exception as e:
        print(f"❌ Erreur re-ranking basique: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_edge_cases():
    """Test des cas limites"""
    print_separator("TEST CAS LIMITES")
    
    if not RERANKER_AVAILABLE:
        print("⏭️ Skip - module non disponible")
        return False
    
    try:
        reranker = CrossEncoderReranker()
        
        # Test 1: Liste vide
        print("📊 Test 1: Liste de candidats vide")
        empty_result = reranker.rerank("test query", [], top_k=5)
        print(f"   Résultat: {len(empty_result)} documents (attendu: 0)")
        
        # Test 2: Requête vide
        print("\n📊 Test 2: Requête vide")
        candidates = create_test_candidates()[:2]
        empty_query_result = reranker.rerank("", candidates, top_k=5)
        print(f"   Résultat: {len(empty_query_result)} documents")
        print(f"   Ordre conservé: {len(empty_query_result) == len(candidates)}")
        
        # Test 3: Textes vides ou invalides
        print("\n📊 Test 3: Candidats avec textes vides")
        invalid_candidates = [
            {"document_id": "doc1", "chunk_id": "1", "text": "", "fused_score": 0.8},
            {"document_id": "doc2", "chunk_id": "2", "text": "Valid text here", "fused_score": 0.7},
            {"document_id": "doc3", "chunk_id": "3", "text": "   ", "fused_score": 0.6}  # Espaces
        ]
        
        invalid_result = reranker.rerank("test query", invalid_candidates, top_k=5)
        print(f"   Candidats originaux: {len(invalid_candidates)}")
        print(f"   Candidats re-rankés: {len(invalid_result)}")
        
        # Test 4: top_k plus grand que candidats disponibles
        print("\n📊 Test 4: top_k > nombre de candidats")
        small_candidates = create_test_candidates()[:2]
        large_k_result = reranker.rerank("test", small_candidates, top_k=10)
        print(f"   Candidats: {len(small_candidates)}, top_k: 10, résultat: {len(large_k_result)}")
        
        # Test 5: Très long texte
        print("\n📊 Test 5: Texte très long")
        long_text = "This is a very long text. " * 100  # ~2500 caractères
        long_candidates = [{
            "document_id": "long_doc",
            "chunk_id": "1", 
            "text": long_text,
            "fused_score": 0.8
        }]
        
        long_result = reranker.rerank("test query", long_candidates, top_k=1)
        print(f"   Texte original: {len(long_text)} chars")
        print(f"   Traitement réussi: {len(long_result) > 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur tests cas limites: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_pairs():
    """Test de la fonction score_pairs"""
    print_separator("TEST SCORE PAIRS")
    
    if not RERANKER_AVAILABLE:
        print("⏭️ Skip - module non disponible")
        return False
    
    try:
        reranker = CrossEncoderReranker()
        
        # Test pairs simples
        pairs = [
            ("robot error", "The robot displays an error message on screen"),
            ("robot error", "Installing new software on the computer"),
            ("calibration procedure", "Follow calibration steps carefully for accuracy"),
            ("calibration procedure", "The weather is nice today")
        ]
        
        print(f"🔍 Test de {len(pairs)} paires query-document")
        
        start_time = time.time()
        scores = reranker.score_pairs(pairs)
        score_time = time.time() - start_time
        
        print(f"✅ Scoring terminé ({score_time:.2f}s)")
        
        print(f"\n📊 RÉSULTATS DES SCORES:")
        for i, (query, doc, score) in enumerate(zip([p[0] for p in pairs], [p[1] for p in pairs], scores)):
            print(f"   {i+1}. Score: {score:.4f}")
            print(f"      Query: \"{query}\"")
            print(f"      Doc: \"{doc[:60]}...\"")
            print()
        
        # Vérification logique des scores
        if len(scores) >= 4:
            # Les paires pertinentes devraient avoir des scores plus élevés
            relevant_scores = [scores[0], scores[2]]  # robot-robot, calibration-calibration
            irrelevant_scores = [scores[1], scores[3]]  # robot-software, calibration-weather
            
            avg_relevant = np.mean(relevant_scores)
            avg_irrelevant = np.mean(irrelevant_scores)
            
            print(f"📈 Score moyen pertinent: {avg_relevant:.4f}")
            print(f"📉 Score moyen non-pertinent: {avg_irrelevant:.4f}")
            
            if avg_relevant > avg_irrelevant:
                print("✅ Logique des scores correcte (pertinent > non-pertinent)")
            else:
                print("⚠️ Logique des scores questionnable")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test score pairs: {e}")
        return False


def test_performance_benchmark():
    """Test de performance du reranker"""
    print_separator("TEST PERFORMANCE")
    
    if not RERANKER_AVAILABLE:
        print("⏭️ Skip - module non disponible")
        return False
    
    try:
        reranker = CrossEncoderReranker()
        
        # Test avec différentes tailles de candidats
        test_sizes = [5, 10, 20]
        query = "FANUC robot error ACAL-006"
        
        print(f"🚀 Benchmark avec requête: \"{query}\"")
        
        for size in test_sizes:
            print(f"\n📊 Test avec {size} candidats:")
            
            # Créer des candidats de test
            base_candidates = create_test_candidates()
            test_candidates = []
            
            for i in range(size):
                candidate = base_candidates[i % len(base_candidates)].copy()
                candidate['chunk_id'] = f"chunk_{i}"
                candidate['text'] = f"Document {i}: " + candidate['text']
                test_candidates.append(candidate)
            
            # Mesurer le temps
            start_time = time.time()
            reranked = reranker.rerank(query, test_candidates, top_k=min(5, size))
            rerank_time = time.time() - start_time
            
            # Calculer les métriques
            docs_per_second = size / rerank_time if rerank_time > 0 else float('inf')
            ms_per_doc = (rerank_time / size) * 1000 if size > 0 else 0
            
            print(f"   ⏱️ Temps total: {rerank_time:.3f}s")
            print(f"   📈 Documents/seconde: {docs_per_second:.1f}")
            print(f"   📊 ms par document: {ms_per_doc:.1f}ms")
            print(f"   ✅ Résultats: {len(reranked)}")
        
        # Test benchmark intégré
        print(f"\n🧪 Test benchmark intégré:")
        test_docs = [candidate['text'] for candidate in create_test_candidates()]
        benchmark_results = reranker.benchmark_speed(query, test_docs, num_runs=3)
        
        print(f"📊 Résultats benchmark:")
        for key, value in benchmark_results.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test performance: {e}")
        return False


def test_with_fusion_data():
    """Test avec des données simulant vraiment le pipeline de fusion"""
    print_separator("TEST AVEC DONNÉES FUSION RÉALISTES")
    
    if not RERANKER_AVAILABLE:
        print("⏭️ Skip - module non disponible")
        return False
    
    try:
        reranker = CrossEncoderReranker()
        
        # Simuler des résultats de fusion réalistes
        fusion_candidates = [
            {
                "document_id": "fanuc_troubleshooting_guide",
                "chunk_id": "section_4_2",
                "text": "ACAL-006 Teach Pendant Error: This error indicates a communication failure between the teach pendant and the robot controller. First, check all cable connections. Ensure the teach pendant cable is securely connected to the controller. If connections are secure, restart the controller and teach pendant. If error persists, the teach pendant may need replacement.",
                "fused_score": 0.8245,
                "bm25_score": 0.9100,
                "faiss_score": 0.7390,
                "original_rank": 1
            },
            {
                "document_id": "fanuc_error_codes_manual",
                "chunk_id": "acal_errors",
                "text": "ACAL series errors are related to calibration and positioning systems. ACAL-006 specifically refers to TPE (Teach Pendant) operation errors. These errors occur when the robot controller cannot properly communicate with the teach pendant device.",
                "fused_score": 0.7891,
                "bm25_score": 0.8200,
                "faiss_score": 0.7582,
                "original_rank": 2
            },
            {
                "document_id": "maintenance_procedures",
                "chunk_id": "communication_troubleshoot",
                "text": "When troubleshooting communication errors between robot components, always start with physical connections. Check for loose cables, damaged connectors, or interference from other electrical equipment. Communication errors can also be caused by software configuration issues.",
                "fused_score": 0.7234,
                "bm25_score": 0.6890,
                "faiss_score": 0.7578,
                "original_rank": 3
            },
            {
                "document_id": "safety_procedures",
                "chunk_id": "error_response",
                "text": "When any error occurs on the robot system, immediately stop all operations and assess the situation. Do not attempt to bypass safety systems or ignore error messages. Follow proper shutdown procedures and consult the appropriate technical documentation.",
                "fused_score": 0.5467,
                "bm25_score": 0.5100,
                "faiss_score": 0.5834,
                "original_rank": 4
            },
            {
                "document_id": "installation_guide",
                "chunk_id": "initial_setup",
                "text": "During initial robot installation, ensure all communication cables are properly routed and secured. Use only manufacturer-approved cables and connectors. Test all communication links before beginning operation.",
                "fused_score": 0.4123,
                "bm25_score": 0.3890,
                "faiss_score": 0.4356,
                "original_rank": 5
            }
        ]
        
        query = "I got ACAL-006 error on my FANUC teach pendant, what should I do?"
        
        print(f"🔍 Requête réaliste: \"{query}\"")
        print(f"📊 {len(fusion_candidates)} candidats de fusion")
        
        # Afficher l'ordre initial
        print(f"\n📋 ORDRE INITIAL (par score fusion):")
        for i, candidate in enumerate(fusion_candidates):
            print(f"   {i+1}. Fusion: {candidate['fused_score']:.4f} | BM25: {candidate['bm25_score']:.4f} | FAISS: {candidate['faiss_score']:.4f}")
            print(f"      {candidate['document_id']}")
            print(f"      {candidate['text'][:100]}...")
            print()
        
        # Re-ranking
        print(f"🎯 Re-ranking avec CrossEncoder...")
        start_time = time.time()
        reranked_results = reranker.rerank(
            query=query,
            candidates=fusion_candidates,
            top_k=5,
            return_scores=True
        )
        rerank_time = time.time() - start_time
        
        print(f"✅ Re-ranking terminé ({rerank_time:.3f}s)")
        
        # Afficher les résultats re-rankés
        print(f"\n🏆 ORDRE APRÈS RE-RANKING:")
        for i, result in enumerate(reranked_results):
            cross_score = result['cross_encoder_score']
            fusion_score = result['fused_score']
            original_rank = result['original_rank']
            
            print(f"   {i+1}. CrossEncoder: {cross_score:.4f} | Fusion: {fusion_score:.4f} | Rang orig: #{original_rank}")
            print(f"      {result['document_id']}")
            print(f"      {result['text'][:100]}...")
            
            # Analyser si c'est un bon match
            text_lower = result['text'].lower()
            if 'acal-006' in text_lower and any(term in text_lower for term in ['teach pendant', 'tpe']):
                print(f"      ✅ Excellent match (contient ACAL-006 + teach pendant)")
            elif 'acal-006' in text_lower:
                print(f"      ✅ Bon match (contient ACAL-006)")
            elif any(term in text_lower for term in ['teach pendant', 'tpe', 'communication']):
                print(f"      ⚠️ Match partiel")
            else:
                print(f"      ❌ Match faible")
            print()
        
        # Analyse des améliorations
        print(f"📊 ANALYSE DE L'AMÉLIORATION:")
        
        # Comparer les positions
        position_changes = 0
        for i, result in enumerate(reranked_results):
            original_pos = result['original_rank'] - 1  # Convert to 0-based
            new_pos = i
            if original_pos != new_pos:
                position_changes += 1
                print(f"   🔄 {result['document_id']}: position {original_pos+1} → {new_pos+1}")
        
        print(f"   📈 Changements de position: {position_changes}/{len(reranked_results)}")
        
        # Vérifier si le résultat le plus pertinent est en tête
        best_result = reranked_results[0] if reranked_results else None
        if best_result and 'acal-006' in best_result['text'].lower():
            print(f"   ✅ Le résultat #1 contient ACAL-006 (très pertinent)")
        else:
            print(f"   ⚠️ Le résultat #1 ne contient pas ACAL-006")
        
        return True, reranked_results
        
    except Exception as e:
        print(f"❌ Erreur test données fusion: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Fonction principale du test"""
    print_separator("🎯 TEST DU SYSTÈME DE RE-RANKING CROSSENCODER 🎯")
    
    total_tests = 0
    passed_tests = 0
    
    # Liste des tests
    tests = [
        ("Disponibilité", test_reranker_availability),
        ("Initialisation", test_reranker_initialization),
        ("Modèle custom", test_reranker_with_custom_model),
        ("Re-ranking basique", test_basic_reranking),
        ("Cas limites", test_edge_cases),
        ("Score pairs", test_score_pairs),
        ("Performance", test_performance_benchmark),
        ("Données fusion", test_with_fusion_data)
    ]
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        print_separator(f"TEST: {test_name.upper()}", "-")
        total_tests += 1
        
        try:
            # Certains tests retournent des tuples, d'autres des booléens
            result = test_func()
            if isinstance(result, tuple):
                success = result[0]
            else:
                success = result
            
            if success:
                passed_tests += 1
                print(f"✅ {test_name}: RÉUSSI")
            else:
                print(f"❌ {test_name}: ÉCHEC")
                
        except Exception as e:
            print(f"💥 {test_name}: EXCEPTION - {e}")
    
    # Rapport final
    total_time = time.time() - start_time
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print_separator("📊 RAPPORT FINAL")
    print(f"⏱️ Temps total: {total_time:.1f}s")
    print(f"✅ Tests réussis: {passed_tests}/{total_tests}")
    print(f"📈 Taux de réussite: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 CROSSENCODER FONCTIONNE CORRECTEMENT!")
        print("🏅 Le système de re-ranking est opérationnel")
        if passed_tests == total_tests:
            print("💯 TOUS LES TESTS RÉUSSIS!")
    elif success_rate >= 60:
        print("\n⚠️ CrossEncoder partiellement fonctionnel")
        print("🔧 Quelques problèmes à résoudre")
    else:
        print("\n❌ PROBLÈMES MAJEURS DÉTECTÉS")
        print("🛠️ Révision nécessaire du système de re-ranking")
        
        if not RERANKER_AVAILABLE:
            print("\n💡 SOLUTION PROBABLE:")
            print("   poetry add sentence-transformers torch")
            print("   Puis relancer le test")
    
    print_separator("✅ TEST CROSSENCODER TERMINÉ")


if __name__ == "__main__":
    main()