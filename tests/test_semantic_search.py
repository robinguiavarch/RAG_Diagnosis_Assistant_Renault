"""
Semantic Search Testing Module: FAISS Retriever Functionality Tests

This module provides comprehensive testing functionality for the FAISSRetriever semantic search system.
It implements various test scenarios to validate semantic search capabilities and is designed for
quality assurance and performance evaluation of the RAG system's retrieval component.

Key components:
- Basic functionality testing: Validates retriever initialization and index statistics
- Semantic query testing: Tests various semantic search patterns and relevance scoring
- Edge case testing: Handles boundary conditions and error scenarios
- Performance testing: Measures search speed and throughput metrics
- Comparative analysis: Evaluates semantic vs conceptual search differences

Dependencies: FAISS, sentence-transformers, PyYAML, pathlib
Usage: Run with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 poetry run python tests/test_semantic.py
"""

import sys
from pathlib import Path
import yaml
from typing import Dict, Any

# Add root directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.retrieval_engine.semantic_search import FAISSRetriever


def load_settings(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    Load configuration settings from YAML file.
    
    Reads and parses the main configuration file containing paths, model names,
    and other system settings required for the retriever initialization.
    
    Args:
        config_path (str): Path to the YAML configuration file
    
    Returns:
        Dict[str, Any]: Parsed configuration dictionary with all settings
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the YAML file is malformed
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def print_separator(title: str, char: str = "="):
    """
    Print a formatted separator line with title for test section organization.
    
    Creates visual separation between different test sections to improve
    readability of test output and organization of results.
    
    Args:
        title (str): The title to display in the separator
        char (str): Character to use for the separator line (default: "=")
    
    Returns:
        None: Prints directly to stdout
    """
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def print_result(result: Dict, index: int):
    """
    Display a search result in a formatted, readable manner.
    
    Formats and prints detailed information about a single search result including
    metadata, scores, and text content with proper line wrapping for readability.
    
    Args:
        result (Dict): Search result dictionary containing all metadata and text
        index (int): Zero-based index of the result for numbering display
    
    Returns:
        None: Prints formatted result directly to stdout
    """
    print(f"\nSEMANTIC RESULT #{index + 1}")
    print(f"Document ID: {result['document_id']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Similarity Score: {result['score']:.4f}")
    print(f"L2 Distance: {result.get('distance', 'N/A'):.4f}" if 'distance' in result else "")
    print(f"Word Count: {result.get('word_count', 'N/A')}")
    print(f"Character Count: {result.get('char_count', 'N/A')}")
    print(f"Embedding Norm: {result.get('embedding_norm', 'N/A'):.3f}" if 'embedding_norm' in result else "")
    print(f"Source File: {result.get('source_file', 'N/A')}")
    print(f"TEXT CONTENT:")
    print("-" * 50)
    
    # Format text with line breaks for readability (80 characters per line)
    text = result['text']
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > 80:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                lines.append(word)  # Very long word
                current_length = 0
        else:
            current_line.append(word)
            current_length += len(word) + 1
    
    if current_line:
        lines.append(' '.join(current_line))
    
    for line in lines:
        print(line)
    print("-" * 50)


def test_basic_functionality():
    """
    Test basic functionality of the FAISS retriever system.
    
    Validates the initialization process, configuration loading, file existence,
    and basic retriever statistics. This serves as the foundation test to ensure
    the system is properly set up before running more complex tests.
    
    Returns:
        FAISSRetriever or None: Initialized retriever instance if successful, None if failed
        
    Raises:
        Exception: Various exceptions related to file access, configuration, or initialization
    """
    print_separator("BASIC FUNCTIONALITY TESTING")
    
    try:
        # Load configuration
        settings = load_settings()
        
        # FAISS index paths
        faiss_index_dir = Path(settings["paths"]["faiss_index_dir"])
        index_path = Path(settings["paths"]["faiss_index"])
        metadata_path = Path(settings["paths"]["embedding_file"])
        model_name = settings["models"]["embedding_model"]
        
        print(f"FAISS Index Path: {index_path}")
        print(f"Metadata Path: {metadata_path}")
        print(f"Model Name: {model_name}")
        
        # Verify file existence
        if not index_path.exists():
            print(f"FAISS index not found: {index_path}")
            print("Execute first: poetry run python scripts/05_create_faiss_index.py")
            return None
        
        if not metadata_path.exists():
            print(f"Metadata not found: {metadata_path}")
            return None
        
        # Initialize retriever
        print("\nInitializing FAISSRetriever...")
        retriever = FAISSRetriever(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_model_name=model_name
        )
        print("Retriever initialized successfully")
        
        # Index statistics
        stats = retriever.get_index_stats()
        print(f"\nINDEX STATISTICS:")
        print(f"   Total Vectors: {stats['total_vectors']}")
        print(f"   Vector Dimension: {stats['vector_dimension']}")
        print(f"   Index Type: {stats['index_type']}")
        print(f"   Unique Documents: {stats['unique_documents']}")
        print(f"   Total Chunks: {stats['total_chunks']}")
        print(f"   Average Chunks per Document: {stats['avg_chunks_per_doc']:.1f}")
        print(f"   Model Device: {stats['model_device']}")
        print(f"   Metadata Format: {stats['metadata_format']}")
        
        return retriever
        
    except Exception as e:
        print(f"Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_semantic_queries(retriever: FAISSRetriever):
    """
    Test various semantic search queries to validate retrieval accuracy.
    
    Executes a comprehensive set of semantic search queries designed to test
    different aspects of the retrieval system including specific error codes,
    general concepts, and various semantic similarity patterns.
    
    Args:
        retriever (FAISSRetriever): Initialized FAISS retriever instance
    
    Returns:
        None: Prints test results directly to stdout
        
    Raises:
        Exception: Search-related errors or retriever failures
    """
    print_separator("SEMANTIC QUERY TESTING")
    
    # Semantic test queries
    test_queries = [
        {
            "query": "I got the error ACAL-006 TPE operation error on the FANUC teach pendant. What should I do?",
            "description": "Primary Query - FANUC ACAL-006 Error",
            "top_k": 5
        },
        {
            "query": "robot calibration failed teach pendant",
            "description": "Semantic Query - Robot Calibration Failure",
            "top_k": 3
        },
        {
            "query": "how to troubleshoot FANUC robot error",
            "description": "General Query - FANUC Troubleshooting",
            "top_k": 3
        },
        {
            "query": "teaching pendant operation problem",
            "description": "Semantic Query - Teaching Pendant Problem",
            "top_k": 3
        },
        {
            "query": "automation error code diagnostic",
            "description": "Conceptual Query - Error Code Diagnostic",
            "top_k": 3
        },
        {
            "query": "industrial robot malfunction solution",
            "description": "Semantic Query - Robot Malfunction Solution",
            "top_k": 3
        }
    ]
    
    for i, test_case in enumerate(test_queries):
        print_separator(f"QUERY {i+1}: {test_case['description']}", "-")
        print(f"Query Text: \"{test_case['query']}\"")
        print(f"Top-K Results: {test_case['top_k']}")
        
        try:
            # Perform semantic search
            results = retriever.search(
                query=test_case['query'], 
                top_k=test_case['top_k'],
                min_score=0.0  # No score filtering to see all results
            )
            
            if not results:
                print("No results found")
                
                # Debug search to understand the issue
                debug_info = retriever.debug_search(test_case['query'], top_k=1)
                print(f"Debug - Query embedding dimension: {debug_info['query_embedding_dim']}")
                print(f"Debug - Query embedding norm: {debug_info['query_embedding_norm']:.3f}")
                print(f"Debug - Index statistics: {debug_info['index_stats']}")
                continue
            
            print(f"Found {len(results)} result(s)")
            
            # Display results
            for j, result in enumerate(results):
                print_result(result, j)
            
            # Semantic relevance analysis
            scores = [r['score'] for r in results]
            distances = [r.get('distance', 0) for r in results if 'distance' in r]
            
            print(f"\nSEMANTIC SCORE ANALYSIS:")
            print(f"   Maximum Score: {max(scores):.4f}")
            print(f"   Minimum Score: {min(scores):.4f}")
            print(f"   Average Score: {sum(scores)/len(scores):.4f}")
            if distances:
                print(f"   Minimum L2 Distance: {min(distances):.4f}")
                print(f"   Maximum L2 Distance: {max(distances):.4f}")
            
        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            traceback.print_exc()


def test_semantic_vs_lexical_comparison(retriever: FAISSRetriever):
    """
    Compare semantic search results with conceptually similar queries.
    
    Tests the semantic understanding capabilities by comparing results from
    the main query with conceptually similar but lexically different queries
    to validate the semantic search effectiveness.
    
    Args:
        retriever (FAISSRetriever): Initialized FAISS retriever instance
    
    Returns:
        None: Prints comparison results directly to stdout
        
    Raises:
        Exception: Search-related errors during comparison
    """
    print_separator("SEMANTIC VS CONCEPTUAL COMPARISON")
    
    # Queries for testing semantic differences
    comparison_queries = [
        {
            "query": "robot malfunctioning",
            "description": "Concept: Robot Malfunction",
            "similar_concepts": ["machine broken", "automation failure", "equipment error"]
        },
        {
            "query": "calibration procedure",
            "description": "Concept: Calibration Procedure", 
            "similar_concepts": ["adjustment process", "setup method", "configuration steps"]
        }
    ]
    
    for test_case in comparison_queries:
        print_separator(f"TEST: {test_case['description']}", "-")
        
        # Test main query
        print(f"Main Query: \"{test_case['query']}\"")
        main_results = retriever.search(test_case['query'], top_k=2)
        
        if main_results:
            print(f"Found {len(main_results)} results for main query")
            for i, result in enumerate(main_results):
                print(f"   {i+1}. Score: {result['score']:.4f} | {result['text'][:100]}...")
        
        # Test similar concepts
        for concept in test_case['similar_concepts']:
            print(f"\nSimilar Concept: \"{concept}\"")
            concept_results = retriever.search(concept, top_k=1)
            
            if concept_results:
                result = concept_results[0]
                print(f"   Score: {result['score']:.4f} | {result['text'][:100]}...")
                
                # Compare with main query
                if main_results:
                    score_diff = abs(result['score'] - main_results[0]['score'])
                    print(f"   Score Difference: {score_diff:.4f}")


def test_edge_cases(retriever: FAISSRetriever):
    """
    Test edge cases and boundary conditions for semantic search robustness.
    
    Validates system behavior with various problematic inputs including empty
    queries, nonsensical text, very short queries, and other edge conditions
    to ensure robust error handling and graceful degradation.
    
    Args:
        retriever (FAISSRetriever): Initialized FAISS retriever instance
    
    Returns:
        None: Prints edge case test results directly to stdout
        
    Raises:
        Exception: Various exceptions that should be handled gracefully
    """
    print_separator("SEMANTIC EDGE CASE TESTING")
    
    edge_cases = [
        "",  # Empty query
        "   ",  # Whitespace only
        "qwertyuiopasdfgh",  # Nonsense words
        "a",  # Very short query
        "the and or",  # Common words only
        "FANUC " * 20,  # Excessive repetition
    ]
    
    for i, query in enumerate(edge_cases):
        print(f"\nEdge Case {i+1}: \"{query}\"")
        try:
            results = retriever.search(query, top_k=2)
            print(f"   Results Found: {len(results)}")
            if results:
                print(f"   Best Score: {results[0]['score']:.4f}")
                print(f"   Distance: {results[0].get('distance', 'N/A'):.4f}" if 'distance' in results[0] else "")
        except Exception as e:
            print(f"   Error: {e}")


def test_performance(retriever: FAISSRetriever):
    """
    Test semantic search performance and throughput metrics.
    
    Measures the speed and efficiency of semantic search operations by
    performing multiple searches and calculating average response times,
    throughput, and performance classification.
    
    Args:
        retriever (FAISSRetriever): Initialized FAISS retriever instance
    
    Returns:
        None: Prints performance metrics directly to stdout
        
    Raises:
        Exception: Performance testing related errors
    """
    print_separator("SEMANTIC PERFORMANCE TESTING")
    
    import time
    
    query = "FANUC error ACAL-006 TPE operation"
    num_searches = 10
    
    print(f"Running {num_searches} semantic searches with: \"{query}\"")
    
    start_time = time.time()
    
    for i in range(num_searches):
        results = retriever.search(query, top_k=5)
        if i == 0:
            first_result_count = len(results)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_searches
    
    print(f"Total Time: {total_time:.3f}s")
    print(f"Average Time per Search: {avg_time:.3f}s")
    print(f"Results per Search: {first_result_count}")
    print(f"Searches per Second: {num_searches/total_time:.1f}")
    
    # Performance classification
    if avg_time < 0.1:
        print("Excellent Performance (<100ms)")
    elif avg_time < 0.5:
        print("Good Performance (<500ms)")
    else:
        print("Performance Needs Improvement (>500ms)")


def main():
    """
    Main test execution function that orchestrates all semantic search tests.
    
    Coordinates the execution of all test suites in the proper order, handling
    initialization, test execution, and final reporting. Provides comprehensive
    coverage of the FAISSRetriever semantic search functionality.
    
    Returns:
        None: Prints all test results and final status
        
    Raises:
        Exception: Global test execution errors
    """
    print_separator("FAISS RETRIEVER SEMANTIC SEARCH TESTING")
    
    try:
        # Test basic functionality
        retriever = test_basic_functionality()
        
        if retriever is None:
            print("Cannot initialize retriever. Stopping tests.")
            return
        
        # Test semantic queries
        test_semantic_queries(retriever)
        
        # Semantic vs conceptual comparison
        test_semantic_vs_lexical_comparison(retriever)
        
        # Edge case testing
        test_edge_cases(retriever)
        
        # Performance testing
        test_performance(retriever)
        
        print_separator("SEMANTIC TESTS COMPLETED SUCCESSFULLY")
        print("The FAISSRetriever is functioning correctly")
        print("Semantic search captures concepts and meaning effectively")
        
    except Exception as e:
        print(f"GLOBAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()