"""
Hybrid Symptom Matcher: Advanced Multi-Modal Symptom Search System

This module provides sophisticated hybrid search capabilities combining BM25 lexical search,
FAISS semantic search, and Levenshtein distance matching for optimal symptom retrieval.
It implements weighted fusion algorithms and specialized error code matching to enhance
diagnostic accuracy in the RAG system with comprehensive fallback mechanisms.

Key components:
- Multi-modal search combining BM25, FAISS, and Levenshtein distance algorithms
- Weighted fusion with configurable parameters for optimal result combination
- Specialized error code extraction and matching using regex patterns
- Lazy loading of search components for efficient resource management
- Comprehensive normalization and scoring mechanisms for result quality

Dependencies: sentence-transformers, faiss, whoosh, Levenshtein, numpy, yaml
Usage: Import HybridSymptomMatcher for advanced multi-modal symptom search operations
"""

import os
import re
import yaml
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from whoosh import scoring
from Levenshtein import distance as levenshtein_distance

class HybridSymptomMatcher:
    """
    Hybrid matcher for symptoms combining BM25, FAISS, and Levenshtein distance.
    
    Implements sophisticated multi-modal search strategy with weighted fusion
    for optimal symptom matching across lexical, semantic, and similarity-based
    search modalities with comprehensive configuration and fallback support.
    """
    
    def __init__(self):
        """
        Initialize hybrid symptom matcher with configuration loading and lazy component setup.
        
        Sets up the matcher with configurable weights, search modalities, and deferred
        loading of search components for optimal resource utilization and performance.
        """
        self.config = self._load_config()
        self.weights = self.config["hybrid_symptom_search"]["weights"]
        self.enabled = self.config["hybrid_symptom_search"]["enabled"]
        
        # Components (lazy loading)
        self.bm25_index = None
        self.faiss_index = None
        self.faiss_metadata = None
        self.embedding_model = None
        
        print(f"HybridSymptomMatcher initialized (enabled: {self.enabled})")
        print(f"Weights: BM25={self.weights['bm25_alpha']}, FAISS={self.weights['faiss_beta']}, Levenshtein={self.weights['levenshtein_gamma']}")
    
    def _load_config(self):
        """
        Load configuration from settings file with proper path resolution.
        
        Reads hybrid search configuration including weights, thresholds, and
        component settings from YAML configuration file with error handling.
        
        Returns:
            dict: Configuration dictionary with hybrid search parameters
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "..", "config", "settings.yaml")
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def _get_bm25_index(self):
        """
        Load BM25 symptom index with lazy initialization and error handling.
        
        Initializes the Whoosh BM25 index for lexical symptom search on first use
        with path validation and comprehensive error handling for robust operation.
        
        Returns:
            whoosh.index: BM25 index instance or None if unavailable
        """
        if self.bm25_index is None:
            bm25_path = self.config["hybrid_symptom_search"]["bm25_index_path"]
            if os.path.exists(bm25_path):
                self.bm25_index = open_dir(bm25_path)
                print("BM25 symptom index loaded")
            else:
                print(f"Warning: BM25 index not found: {bm25_path}")
        return self.bm25_index
    
    def _get_faiss_components(self):
        """
        Load FAISS index and metadata with corrected path resolution.
        
        Initializes FAISS semantic search components including index and metadata
        with proper path construction and comprehensive error handling for reliability.
        
        Returns:
            tuple: (FAISS index, metadata dict) or (None, None) if unavailable
        """
        if self.faiss_index is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Corrected path: symptom_embeddings_dense → symptom_embedding_dense
            embedding_dir = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "symptom_embedding_dense")
            
            index_path = os.path.join(embedding_dir, "index.faiss")
            metadata_path = os.path.join(embedding_dir, "symptom_embedding_dense.pkl")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.faiss_index = faiss.read_index(index_path)
                with open(metadata_path, "rb") as f:
                    self.faiss_metadata = pickle.load(f)
                print("FAISS symptom index loaded")
            else:
                print(f"Warning: FAISS index not found: {embedding_dir}")
        
        return self.faiss_index, self.faiss_metadata
    
    def _get_embedding_model(self):
        """
        Load SentenceTransformer embedding model with lazy initialization.
        
        Initializes the embedding model for semantic search on first use with
        configuration-based model selection for consistent embedding generation.
        
        Returns:
            SentenceTransformer: Embedding model instance for semantic search
        """
        if self.embedding_model is None:
            model_name = self.config["models"]["embedding_model"]
            self.embedding_model = SentenceTransformer(model_name)
        return self.embedding_model
    
    def _search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform BM25 lexical search in symptom index.
        
        Executes lexical search using Whoosh BM25 scoring with query cleaning
        and comprehensive result formatting for integration with hybrid fusion.
        
        Args:
            query (str): Search query for BM25 lexical matching
            top_k (int): Maximum number of results to return
            
        Returns:
            List[Dict]: BM25 search results with scores and metadata
        """
        bm25_index = self._get_bm25_index()
        if not bm25_index:
            return []
        
        try:
            with bm25_index.searcher(weighting=scoring.BM25F()) as searcher:
                parser = QueryParser("symptom_text", schema=bm25_index.schema)
                
                # Query cleaning
                cleaned_query = re.sub(r'[^\w\s-]', ' ', query)
                parsed_query = parser.parse(cleaned_query)
                
                results = searcher.search(parsed_query, limit=top_k)
                
                bm25_results = []
                for hit in results:
                    bm25_results.append({
                        'symptom_text': hit['symptom_text'],
                        'symptom_id': hit['symptom_id'],
                        'equipment': hit.get('equipment', 'unknown'),
                        'bm25_score': float(hit.score),
                        'source': 'BM25'
                    })
                
                return bm25_results
                
        except Exception as e:
            print(f"BM25 error: {e}")
            return []
    
    def _search_faiss(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform FAISS semantic search in symptom embeddings.
        
        Executes semantic search using FAISS vector similarity with normalized
        embeddings and comprehensive result formatting for hybrid integration.
        
        Args:
            query (str): Search query for semantic matching
            top_k (int): Maximum number of results to return
            
        Returns:
            List[Dict]: FAISS search results with similarity scores and metadata
        """
        faiss_index, metadata = self._get_faiss_components()
        if not faiss_index or not metadata:
            return []
        
        try:
            model = self._get_embedding_model()
            symptom_names = metadata['symptom_names']
            
            # Query embedding
            query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
            scores, indices = faiss_index.search(query_vec, top_k)
            
            faiss_results = []
            for i, score in zip(indices[0], scores[0]):
                if i < len(symptom_names):
                    faiss_results.append({
                        'symptom_text': symptom_names[i],
                        'symptom_id': str(i),
                        'equipment': 'unknown',  # Equipment not in FAISS metadata
                        'faiss_score': float(score),
                        'source': 'FAISS'
                    })
            
            return faiss_results
            
        except Exception as e:
            print(f"FAISS error: {e}")
            return []
    
    def _extract_error_codes(self, text: str) -> List[str]:
        """
        Extract error codes from text using regex pattern matching.
        
        Identifies and extracts standardized error codes (e.g., ACAL-006, SYST-001)
        from input text for specialized Levenshtein distance matching of diagnostic codes.
        
        Args:
            text (str): Input text to search for error codes
            
        Returns:
            List[str]: List of extracted error codes in uppercase format
        """
        pattern = r'\b[A-Z]{3,5}-\d{3,4}\b'
        return re.findall(pattern, text.upper())
    
    def _search_levenshtein(self, query: str, all_symptoms: List[str], top_k: int = 10) -> List[Dict]:
        """
        Perform Levenshtein distance search for error code matching.
        
        Executes specialized string similarity search focusing on error code matching
        with configurable distance thresholds and inverse scoring for optimal results.
        
        Args:
            query (str): Search query containing potential error codes
            all_symptoms (List[str]): Complete list of symptoms to search
            top_k (int): Maximum number of results to return
            
        Returns:
            List[Dict]: Levenshtein search results with distance scores and matched codes
        """
        query_codes = self._extract_error_codes(query)
        if not query_codes:
            return []
        
        threshold = self.config["hybrid_symptom_search"]["levenshtein_threshold"]
        
        levenshtein_results = []
        for symptom in all_symptoms:
            symptom_codes = self._extract_error_codes(symptom)
            
            if symptom_codes:
                # Minimum distance between query codes and symptom codes
                min_distance = float('inf')
                for q_code in query_codes:
                    for s_code in symptom_codes:
                        dist = levenshtein_distance(q_code, s_code)
                        min_distance = min(min_distance, dist)
                
                # Inverse score: low distance = high score
                if min_distance <= threshold:
                    score = 1.0 - (min_distance / threshold)
                    levenshtein_results.append({
                        'symptom_text': symptom,
                        'symptom_id': 'lev_' + str(len(levenshtein_results)),
                        'equipment': 'unknown',
                        'levenshtein_score': score,
                        'source': 'Levenshtein',
                        'matched_codes': (query_codes, symptom_codes)
                    })
        
        # Sort by descending score
        levenshtein_results.sort(key=lambda x: x['levenshtein_score'], reverse=True)
        return levenshtein_results[:top_k]
    
    def search_hybrid(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Main hybrid search combining BM25, FAISS, and Levenshtein with weighted fusion.
        
        Implements comprehensive multi-modal search strategy with normalized scoring,
        weighted combination, and intelligent fallback mechanisms for optimal symptom
        matching across different search modalities and result quality optimization.
        
        Args:
            query (str): Search query for hybrid symptom matching
            top_k (int): Maximum number of results to return
            
        Returns:
            List[Dict]: Ranked hybrid search results with combined scores
        """
        if not self.enabled:
            print("Hybrid search disabled, falling back to FAISS")
            return self._search_faiss(query, top_k)
        
        print(f"Hybrid search for: '{query}'")
        
        # 1. Component searches
        bm25_results = self._search_bm25(query, top_k * 2)
        faiss_results = self._search_faiss(query, top_k * 2)
        
        # For Levenshtein, we need all symptoms
        all_symptoms = []
        if self.faiss_metadata:
            all_symptoms = self.faiss_metadata.get('symptom_names', [])
        
        levenshtein_results = self._search_levenshtein(query, all_symptoms, top_k)
        
        print(f"Raw results: BM25={len(bm25_results)}, FAISS={len(faiss_results)}, Levenshtein={len(levenshtein_results)}")
        
        # 2. Weighted fusion
        combined_scores = {}
        
        # Normalization and BM25 fusion
        if bm25_results:
            max_bm25 = max(r['bm25_score'] for r in bm25_results)
            for result in bm25_results:
                symptom = result['symptom_text']
                norm_score = result['bm25_score'] / max_bm25 if max_bm25 > 0 else 0
                combined_scores[symptom] = combined_scores.get(symptom, 0) + self.weights['bm25_alpha'] * norm_score
        
        # Normalization and FAISS fusion
        if faiss_results:
            max_faiss = max(r['faiss_score'] for r in faiss_results)
            for result in faiss_results:
                symptom = result['symptom_text']
                norm_score = result['faiss_score'] / max_faiss if max_faiss > 0 else 0
                combined_scores[symptom] = combined_scores.get(symptom, 0) + self.weights['faiss_beta'] * norm_score
        
        # Levenshtein fusion (already normalized 0-1)
        for result in levenshtein_results:
            symptom = result['symptom_text']
            combined_scores[symptom] = combined_scores.get(symptom, 0) + self.weights['levenshtein_gamma'] * result['levenshtein_score']
        
        # 3. Final ranking and formatting
        final_results = []
        for symptom, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            final_results.append({
                'symptom_text': symptom,
                'hybrid_score': score,
                'source': 'Hybrid'
            })
        
        print(f"Top {len(final_results)} hybrid symptoms selected")
        return final_results

# Utility Functions

def create_hybrid_symptom_matcher() -> HybridSymptomMatcher:
    """
    Create hybrid symptom matcher instance for multi-modal search operations.
    
    Factory function for creating configured hybrid matcher instances with
    proper initialization and component loading for optimal search performance.
    
    Returns:
        HybridSymptomMatcher: Configured hybrid matcher instance
    """
    return HybridSymptomMatcher()

def search_symptoms_hybrid(query: str, top_k: int = 5) -> List[Dict]:
    """
    Utility function for quick hybrid symptom search operations.
    
    Convenience function for one-off hybrid search operations without requiring
    explicit matcher instance management and configuration handling.
    
    Args:
        query (str): Search query for symptom matching
        top_k (int): Maximum number of results to return
        
    Returns:
        List[Dict]: Hybrid search results with combined scores
    """
    matcher = create_hybrid_symptom_matcher()
    return matcher.search_hybrid(query, top_k)

# CLI Testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Test hybrid search: {query}")
        print("-" * 50)
        
        matcher = create_hybrid_symptom_matcher()
        results = matcher.search_hybrid(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['symptom_text']} (score: {result['hybrid_score']:.3f})")
    else:
        print("Usage: python hybrid_symptom_matcher.py 'your query'")