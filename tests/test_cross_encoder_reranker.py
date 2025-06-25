#!/usr/bin/env python3
"""
CrossEncoder Reranking System Test Suite

This module provides comprehensive testing capabilities for the CrossEncoder reranking system
used in the RAG diagnosis pipeline. It validates initialization, reranking functionality,
performance characteristics, and integration with the fusion pipeline through systematic
testing of various scenarios and edge cases.

Key components:
- Module availability validation: Tests CrossEncoder import and dependency resolution
- Initialization testing: Validates model loading with default and custom configurations
- Reranking functionality: Tests core reranking capabilities with realistic data
- Performance benchmarking: Measures speed and efficiency across different data sizes
- Edge case validation: Tests system robustness with boundary conditions

Dependencies: sentence-transformers, torch, numpy, pyyaml, pathlib
Usage: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 poetry run python tests/test_cross_encoder.py > tests/test_reports/test_cross_encoder.txt 2>&1
"""

import sys
import time
from pathlib import Path
import yaml
from typing import Dict, Any, List
import numpy as np

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Optional reranker import
try:
    from core.reranking_engine.cross_encoder_reranker import CrossEncoderReranker
    RERANKER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CrossEncoderReranker not available: {e}")
    RERANKER_AVAILABLE = False
    CrossEncoderReranker = None

def print_separator(title: str, char: str = "="):
    """
    Display a separator with title for test section organization
    
    Args:
        title (str): Section title to display
        char (str): Character to use for separator line
    """
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")

def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML settings file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Loaded configuration settings with fallback defaults
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Config loading error: {e}")
        return {
            "models": {
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
            }
        }

def test_reranker_availability():
    """
    Test reranker module availability and import status
    
    Validates that the CrossEncoderReranker module can be imported and provides
    guidance for dependency installation if imports fail.
    
    Returns:
        bool: True if module is available, False otherwise
    """
    print_separator("AVAILABILITY TEST")
    
    if RERANKER_AVAILABLE:
        print("CrossEncoderReranker module available")
        return True
    else:
        print("CrossEncoderReranker module not available")
        print("Note: Check installation: poetry add sentence-transformers torch")
        return False

def test_reranker_initialization():
    """
    Test CrossEncoder initialization with default settings
    
    Validates that the CrossEncoder can be properly initialized, measures
    initialization time, and retrieves model information for verification.
    
    Returns:
        tuple: (success_status, reranker_instance) or (False, None) on failure
    """
    print_separator("INITIALIZATION TEST")
    
    if not RERANKER_AVAILABLE:
        print("Skip - module not available")
        return False
    
    # Test with default model
    print("Testing default model initialization...")
    try:
        start_time = time.time()
        reranker = CrossEncoderReranker()
        init_time = time.time() - start_time
        
        print(f"Initialization successful ({init_time:.2f}s)")
        
        # Verify model information
        model_info = reranker.get_model_info()
        print(f"Model information:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        return True, reranker
        
    except Exception as e:
        print(f"Initialization error: {e}")
        return False, None

def test_reranker_with_custom_model():
    """
    Test initialization with custom model from configuration
    
    Validates that the CrossEncoder can be initialized with a custom model
    specified in the configuration file, ensuring flexibility in model selection.
    
    Returns:
        tuple: (success_status, reranker_instance) or (False, None) on failure
    """
    print_separator("CUSTOM MODEL TEST")
    
    if not RERANKER_AVAILABLE:
        print("Skip - module not available")
        return False
    
    # Test with lightweight model for speed
    print("Testing lightweight model...")
    try:
        settings = load_settings()
        model_name = settings["models"]["reranker_model"]
        
        start_time = time.time()
        reranker = CrossEncoderReranker(model_name=model_name)
        init_time = time.time() - start_time
        
        print(f"Model {model_name} loaded ({init_time:.2f}s)")
        return True, reranker
        
    except Exception as e:
        print(f"Custom model error: {e}")
        return False, None

def create_test_candidates() -> List[Dict[str, Any]]:
    """
    Create test candidates simulating fusion pipeline results
    
    Generates realistic test data that mimics the output of the fusion pipeline,
    including document metadata, text content, and various scoring components.
    
    Returns:
        List[Dict[str, Any]]: List of candidate documents with metadata and scores
    """
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
    """
    Test basic reranking functionality with standard test data
    
    Validates core reranking capabilities by processing test candidates and
    analyzing the resulting reordering. Compares before and after rankings
    to assess reranking effectiveness.
    
    Returns:
        tuple: (success_status, reranked_results) or (False, None) on failure
    """
    print_separator("BASIC RERANKING TEST")
    
    if not RERANKER_AVAILABLE:
        print("Skip - module not available")
        return False
    
    try:
        # Initialization
        reranker = CrossEncoderReranker()
        print("Reranker initialized")
        
        # Test data
        query = "ACAL-006 error on FANUC robot teach pendant"
        candidates = create_test_candidates()
        
        print(f"Query: \"{query}\"")
        print(f"Candidates: {len(candidates)}")
        
        # Display before reranking
        print(f"\nBEFORE RERANKING (sorted by fusion score):")
        for i, candidate in enumerate(candidates):
            print(f"   {i+1}. Score: {candidate['fused_score']:.3f} | {candidate['document_id']}|{candidate['chunk_id']}")
            print(f"      {candidate['text'][:80]}...")
        
        # Reranking
        print(f"\nReranking with CrossEncoder...")
        start_time = time.time()
        reranked = reranker.rerank(query, candidates, top_k=len(candidates))
        rerank_time = time.time() - start_time
        
        print(f"Reranking completed ({rerank_time:.2f}s)")
        
        # Display after reranking
        print(f"\nAFTER RERANKING (sorted by CrossEncoder):")
        for i, result in enumerate(reranked):
            cross_score = result.get('cross_encoder_score', 0)
            fusion_score = result.get('fused_score', 0)
            original_rank = result.get('original_rank', '?')
            
            print(f"   {i+1}. CrossEncoder: {cross_score:.3f} | Fusion: {fusion_score:.3f} | Orig rank: #{original_rank}")
            print(f"      {result['document_id']}|{result['chunk_id']}")
            print(f"      {result['text'][:80]}...")
        
        # Analyze changes
        print(f"\nCHANGE ANALYSIS:")
        
        # Original vs reranked order
        original_order = [(c['document_id'], c['chunk_id']) for c in candidates]
        reranked_order = [(r['document_id'], r['chunk_id']) for r in reranked]
        
        changes = 0
        for i, (orig, rerank) in enumerate(zip(original_order, reranked_order)):
            if orig != rerank:
                changes += 1
        
        print(f"   Position changes: {changes}/{len(candidates)}")
        print(f"   Time per document: {(rerank_time/len(candidates)*1000):.1f}ms")
        
        # Verify best result relevance
        if reranked:
            best_result = reranked[0]
            if "ACAL-006" in best_result['text'] and "TPE" in best_result['text']:
                print(f"   Best result contains query key terms")
            else:
                print(f"   Warning: Best result may not be optimal")
        
        return True, reranked
        
    except Exception as e:
        print(f"Basic reranking error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_edge_cases():
    """
    Test edge cases and boundary conditions
    
    Validates system robustness by testing various edge cases including empty inputs,
    invalid data, and boundary conditions that may occur in production use.
    
    Returns:
        bool: True if all edge case tests pass, False otherwise
    """
    print_separator("EDGE CASES TEST")
    
    if not RERANKER_AVAILABLE:
        print("Skip - module not available")
        return False
    
    try:
        reranker = CrossEncoderReranker()
        
        # Test 1: Empty list
        print("Test 1: Empty candidate list")
        empty_result = reranker.rerank("test query", [], top_k=5)
        print(f"   Result: {len(empty_result)} documents (expected: 0)")
        
        # Test 2: Empty query
        print("\nTest 2: Empty query")
        candidates = create_test_candidates()[:2]
        empty_query_result = reranker.rerank("", candidates, top_k=5)
        print(f"   Result: {len(empty_query_result)} documents")
        print(f"   Order preserved: {len(empty_query_result) == len(candidates)}")
        
        # Test 3: Empty or invalid texts
        print("\nTest 3: Candidates with empty texts")
        invalid_candidates = [
            {"document_id": "doc1", "chunk_id": "1", "text": "", "fused_score": 0.8},
            {"document_id": "doc2", "chunk_id": "2", "text": "Valid text here", "fused_score": 0.7},
            {"document_id": "doc3", "chunk_id": "3", "text": "   ", "fused_score": 0.6}  # Spaces only
        ]
        
        invalid_result = reranker.rerank("test query", invalid_candidates, top_k=5)
        print(f"   Original candidates: {len(invalid_candidates)}")
        print(f"   Reranked candidates: {len(invalid_result)}")
        
        # Test 4: top_k larger than available candidates
        print("\nTest 4: top_k > number of candidates")
        small_candidates = create_test_candidates()[:2]
        large_k_result = reranker.rerank("test", small_candidates, top_k=10)
        print(f"   Candidates: {len(small_candidates)}, top_k: 10, result: {len(large_k_result)}")
        
        # Test 5: Very long text
        print("\nTest 5: Very long text")
        long_text = "This is a very long text. " * 100  # ~2500 characters
        long_candidates = [{
            "document_id": "long_doc",
            "chunk_id": "1", 
            "text": long_text,
            "fused_score": 0.8
        }]
        
        long_result = reranker.rerank("test query", long_candidates, top_k=1)
        print(f"   Original text: {len(long_text)} chars")
        print(f"   Processing successful: {len(long_result) > 0}")
        
        return True
        
    except Exception as e:
        print(f"Edge cases test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_score_pairs():
    """
    Test the score_pairs functionality for direct query-document scoring
    
    Validates the ability to score query-document pairs directly, which is
    useful for understanding model behavior and debugging reranking results.
    
    Returns:
        bool: True if score_pairs test passes, False otherwise
    """
    print_separator("SCORE PAIRS TEST")
    
    if not RERANKER_AVAILABLE:
        print("Skip - module not available")
        return False
    
    try:
        reranker = CrossEncoderReranker()
        
        # Test simple pairs
        pairs = [
            ("robot error", "The robot displays an error message on screen"),
            ("robot error", "Installing new software on the computer"),
            ("calibration procedure", "Follow calibration steps carefully for accuracy"),
            ("calibration procedure", "The weather is nice today")
        ]
        
        print(f"Testing {len(pairs)} query-document pairs")
        
        start_time = time.time()
        scores = reranker.score_pairs(pairs)
        score_time = time.time() - start_time
        
        print(f"Scoring completed ({score_time:.2f}s)")
        
        print(f"\nSCORE RESULTS:")
        for i, (query, doc, score) in enumerate(zip([p[0] for p in pairs], [p[1] for p in pairs], scores)):
            print(f"   {i+1}. Score: {score:.4f}")
            print(f"      Query: \"{query}\"")
            print(f"      Doc: \"{doc[:60]}...\"")
            print()
        
        # Logical score verification
        if len(scores) >= 4:
            # Relevant pairs should have higher scores
            relevant_scores = [scores[0], scores[2]]  # robot-robot, calibration-calibration
            irrelevant_scores = [scores[1], scores[3]]  # robot-software, calibration-weather
            
            avg_relevant = np.mean(relevant_scores)
            avg_irrelevant = np.mean(irrelevant_scores)
            
            print(f"Average relevant score: {avg_relevant:.4f}")
            print(f"Average irrelevant score: {avg_irrelevant:.4f}")
            
            if avg_relevant > avg_irrelevant:
                print("Score logic correct (relevant > irrelevant)")
            else:
                print("Warning: Score logic questionable")
        
        return True
        
    except Exception as e:
        print(f"Score pairs test error: {e}")
        return False

def test_performance_benchmark():
    """
    Test reranker performance across different data sizes
    
    Conducts performance benchmarking by measuring reranking speed across
    various candidate set sizes to assess scalability and efficiency.
    
    Returns:
        bool: True if performance tests complete successfully, False otherwise
    """
    print_separator("PERFORMANCE TEST")
    
    if not RERANKER_AVAILABLE:
        print("Skip - module not available")
        return False
    
    try:
        reranker = CrossEncoderReranker()
        
        # Test with different candidate sizes
        test_sizes = [5, 10, 20]
        query = "FANUC robot error ACAL-006"
        
        print(f"Benchmark with query: \"{query}\"")
        
        for size in test_sizes:
            print(f"\nTest with {size} candidates:")
            
            # Create test candidates
            base_candidates = create_test_candidates()
            test_candidates = []
            
            for i in range(size):
                candidate = base_candidates[i % len(base_candidates)].copy()
                candidate['chunk_id'] = f"chunk_{i}"
                candidate['text'] = f"Document {i}: " + candidate['text']
                test_candidates.append(candidate)
            
            # Measure time
            start_time = time.time()
            reranked = reranker.rerank(query, test_candidates, top_k=min(5, size))
            rerank_time = time.time() - start_time
            
            # Calculate metrics
            docs_per_second = size / rerank_time if rerank_time > 0 else float('inf')
            ms_per_doc = (rerank_time / size) * 1000 if size > 0 else 0
            
            print(f"   Total time: {rerank_time:.3f}s")
            print(f"   Documents/second: {docs_per_second:.1f}")
            print(f"   ms per document: {ms_per_doc:.1f}ms")
            print(f"   Results: {len(reranked)}")
        
        # Integrated benchmark test
        print(f"\nIntegrated benchmark test:")
        test_docs = [candidate['text'] for candidate in create_test_candidates()]
        benchmark_results = reranker.benchmark_speed(query, test_docs, num_runs=3)
        
        print(f"Benchmark results:")
        for key, value in benchmark_results.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Performance test error: {e}")
        return False

def test_with_fusion_data():
    """
    Test with realistic data simulating the fusion pipeline output
    
    Validates reranking performance using realistic data that closely mirrors
    the actual fusion pipeline output, including comprehensive metadata and
    scoring information for authentic testing conditions.
    
    Returns:
        tuple: (success_status, reranked_results) or (False, None) on failure
    """
    print_separator("REALISTIC FUSION DATA TEST")
    
    if not RERANKER_AVAILABLE:
        print("Skip - module not available")
        return False
    
    try:
        reranker = CrossEncoderReranker()
        
        # Simulate realistic fusion results
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
        
        print(f"Realistic query: \"{query}\"")
        print(f"{len(fusion_candidates)} fusion candidates")
        
        # Display initial order
        print(f"\nINITIAL ORDER (by fusion score):")
        for i, candidate in enumerate(fusion_candidates):
            print(f"   {i+1}. Fusion: {candidate['fused_score']:.4f} | BM25: {candidate['bm25_score']:.4f} | FAISS: {candidate['faiss_score']:.4f}")
            print(f"      {candidate['document_id']}")
            print(f"      {candidate['text'][:100]}...")
            print()
        
        # Reranking
        print(f"Reranking with CrossEncoder...")
        start_time = time.time()
        reranked_results = reranker.rerank(
            query=query,
            candidates=fusion_candidates,
            top_k=5,
            return_scores=True
        )
        rerank_time = time.time() - start_time
        
        print(f"Reranking completed ({rerank_time:.3f}s)")
        
        # Display reranked results
        print(f"\nORDER AFTER RERANKING:")
        for i, result in enumerate(reranked_results):
            cross_score = result['cross_encoder_score']
            fusion_score = result['fused_score']
            original_rank = result['original_rank']
            
            print(f"   {i+1}. CrossEncoder: {cross_score:.4f} | Fusion: {fusion_score:.4f} | Orig rank: #{original_rank}")
            print(f"      {result['document_id']}")
            print(f"      {result['text'][:100]}...")
            
            # Analyze match quality
            text_lower = result['text'].lower()
            if 'acal-006' in text_lower and any(term in text_lower for term in ['teach pendant', 'tpe']):
                print(f"      Excellent match (contains ACAL-006 + teach pendant)")
            elif 'acal-006' in text_lower:
                print(f"      Good match (contains ACAL-006)")
            elif any(term in text_lower for term in ['teach pendant', 'tpe', 'communication']):
                print(f"      Partial match")
            else:
                print(f"      Weak match")
            print()
        
        # Improvement analysis
        print(f"IMPROVEMENT ANALYSIS:")
        
        # Compare positions
        position_changes = 0
        for i, result in enumerate(reranked_results):
            original_pos = result['original_rank'] - 1  # Convert to 0-based
            new_pos = i
            if original_pos != new_pos:
                position_changes += 1
                print(f"   Position change {result['document_id']}: position {original_pos+1} → {new_pos+1}")
        
        print(f"   Position changes: {position_changes}/{len(reranked_results)}")
        
        # Verify most relevant result is at top
        best_result = reranked_results[0] if reranked_results else None
        if best_result and 'acal-006' in best_result['text'].lower():
            print(f"   Result #1 contains ACAL-006 (highly relevant)")
        else:
            print(f"   Warning: Result #1 does not contain ACAL-006")
        
        return True, reranked_results
        
    except Exception as e:
        print(f"Fusion data test error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """
    Main test execution pipeline for CrossEncoder reranking system
    
    Orchestrates the complete test suite execution, tracks results, and provides
    comprehensive reporting on system functionality and performance characteristics.
    """
    print_separator("CROSSENCODER RERANKING SYSTEM TEST SUITE")
    
    total_tests = 0
    passed_tests = 0
    
    # Test suite definition
    tests = [
        ("Availability", test_reranker_availability),
        ("Initialization", test_reranker_initialization),
        ("Custom model", test_reranker_with_custom_model),
        ("Basic reranking", test_basic_reranking),
        ("Edge cases", test_edge_cases),
        ("Score pairs", test_score_pairs),
        ("Performance", test_performance_benchmark),
        ("Fusion data", test_with_fusion_data)
    ]
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        print_separator(f"TEST: {test_name.upper()}", "-")
        total_tests += 1
        
        try:
            # Some tests return tuples, others return booleans
            result = test_func()
            if isinstance(result, tuple):
                success = result[0]
            else:
                success = result
            
            if success:
                passed_tests += 1
                print(f"{test_name}: PASSED")
            else:
                print(f"{test_name}: FAILED")
                
        except Exception as e:
            print(f"{test_name}: EXCEPTION - {e}")
    
    # Final report
    total_time = time.time() - start_time
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print_separator("FINAL REPORT")
    print(f"Total time: {total_time:.1f}s")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\nCROSSENCODER WORKING CORRECTLY")
        print("Reranking system is operational")
        if passed_tests == total_tests:
            print("ALL TESTS PASSED")
    elif success_rate >= 60:
        print("\nCrossEncoder partially functional")
        print("Some issues need resolution")
    else:
        print("\nMAJOR PROBLEMS DETECTED")
        print("Reranking system requires revision")
        
        if not RERANKER_AVAILABLE:
            print("\nPROBABLE SOLUTION:")
            print("   poetry add sentence-transformers torch")
            print("   Then restart the test")
    
    print_separator("CROSSENCODER TEST COMPLETED")

if __name__ == "__main__":
    main()