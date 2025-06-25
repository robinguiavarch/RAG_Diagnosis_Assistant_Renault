"""
Cross Encoder Reranker: Advanced Document Reranking with Neural Relevance Scoring

This module provides sophisticated document reranking capabilities using Cross Encoder
models for query-document relevance assessment. It enhances hybrid search results by
applying neural scoring with comprehensive batch processing, device optimization, and
robust error handling mechanisms.

Key components:
- Cross Encoder model integration with automatic device selection and fallback
- Batch processing for efficient large-scale reranking operations
- Comprehensive score tracking including fusion, BM25, and FAISS scores
- Performance benchmarking and speed optimization capabilities
- Robust error handling with graceful fallback to original ranking

Dependencies: sentence-transformers, torch, numpy, time
Usage: Import CrossEncoderReranker for neural reranking of hybrid search results
"""

from typing import List, Dict, Union, Any, Optional
from sentence_transformers import CrossEncoder
import torch
import numpy as np
import time


class CrossEncoderReranker:
    """
    Advanced reranker using CrossEncoder models for query-document relevance assessment.
    
    Provides neural reranking capabilities to refine hybrid search results by evaluating
    semantic relevance between queries and documents with comprehensive error handling
    and performance optimization.
    """
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize CrossEncoder reranker with automatic device optimization.
        
        Sets up the Cross Encoder model with automatic device selection, fallback
        mechanisms, and configurable sequence length limits for optimal performance.
        
        Args:
            model_name (str): Name of the CrossEncoder model to use
            device (Optional[str]): Target device ('cuda', 'cpu', or None for auto-detection)
            max_length (int): Maximum sequence length for input processing
            
        Raises:
            RuntimeError: If model cannot be loaded on any available device
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Device determination
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading CrossEncoder: {model_name}")
        print(f"Target device: {device}")
        
        try:
            self.model = CrossEncoder(
                model_name, 
                device=device,
                max_length=max_length
            )
            self.device = device
            print(f"CrossEncoder loaded successfully on {device}")
            
        except Exception as e:
            print(f"Warning: Failed loading on {device}: {e}")
            if device == "cuda":
                print("Attempting CPU fallback...")
                try:
                    self.model = CrossEncoder(
                        model_name, 
                        device="cpu",
                        max_length=max_length
                    )
                    self.device = "cpu"
                    print("CrossEncoder loaded on CPU (fallback)")
                except Exception as e2:
                    raise RuntimeError(f"Unable to load model: {e2}")
            else:
                raise RuntimeError(f"Model loading error: {e}")

    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 5,
        batch_size: int = 32,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates based on query relevance using Cross Encoder scoring.
        
        Performs comprehensive reranking of candidate documents using neural relevance
        scoring with batch processing, score preservation, and robust error handling.
        
        Args:
            query (str): User query for relevance assessment
            candidates (List[Dict[str, Any]]): List of candidate documents to rerank
            top_k (int): Number of top results to return
            batch_size (int): Batch size for efficient processing
            return_scores (bool): Whether to include all scores in results
            
        Returns:
            List[Dict[str, Any]]: Reranked candidates with Cross Encoder scores
        """
        if not candidates:
            print("Warning: No candidates to rerank")
            return []
        
        if not query.strip():
            print("Warning: Empty query for reranking, returning original results")
            return candidates[:top_k]
        
        print(f"Reranking {len(candidates)} candidates with CrossEncoder")
        
        # Candidate validation and cleaning
        valid_candidates = []
        for i, candidate in enumerate(candidates):
            text = candidate.get("text", "").strip()
            if text:
                # Add original index for traceability
                candidate_copy = candidate.copy()
                candidate_copy["original_rank"] = i + 1
                valid_candidates.append(candidate_copy)
            else:
                print(f"Warning: Candidate {i+1} ignored (empty text)")
        
        if not valid_candidates:
            print("Error: No valid candidates for reranking")
            return []
        
        try:
            # Preparation of query-document pairs
            pairs = []
            for candidate in valid_candidates:
                text = candidate["text"]
                # Truncation if necessary to respect max_length
                if len(query) + len(text) > self.max_length - 10:  # Margin for special tokens
                    # Keep beginning of text which is often most important
                    max_text_length = self.max_length - len(query) - 10
                    text = text[:max_text_length]
                
                pairs.append((query, text))
            
            print(f"Scoring {len(pairs)} query-document pairs...")
            
            # Cross Encoder scoring with batch processing
            all_scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.model.predict(batch_pairs)
                all_scores.extend(batch_scores)
            
            # Score enrichment with Cross Encoder results
            for candidate, cross_score in zip(valid_candidates, all_scores):
                candidate["cross_encoder_score"] = float(cross_score)
                
                # Optional: preserve all scores for analysis
                if return_scores and "fused_score" in candidate:
                    candidate["all_scores"] = {
                        "cross_encoder": float(cross_score),
                        "fusion": candidate["fused_score"],
                        "bm25": candidate.get("bm25_score", 0.0),
                        "faiss": candidate.get("faiss_score", 0.0)
                    }
            
            # Sort by Cross Encoder score (descending)
            reranked = sorted(
                valid_candidates,
                key=lambda x: x["cross_encoder_score"],
                reverse=True
            )
            
            print(f"Reranking completed")
            print(f"Top-1 score: {reranked[0]['cross_encoder_score']:.4f}")
            if len(reranked) > 1:
                print(f"Top-2 score: {reranked[1]['cross_encoder_score']:.4f}")
            
            return reranked[:top_k]
            
        except Exception as e:
            print(f"Error during reranking: {e}")
            print("Fallback: returning original fusion results")
            
            # Fallback: return candidates sorted by fusion score
            try:
                fallback_results = sorted(
                    valid_candidates,
                    key=lambda x: x.get("fused_score", x.get("score", 0)),
                    reverse=True
                )
                return fallback_results[:top_k]
            except Exception as e2:
                print(f"Fallback error: {e2}")
                return valid_candidates[:top_k]

    def score_pairs(self, pairs: List[tuple]) -> List[float]:
        """
        Score a list of query-document pairs for relevance assessment.
        
        Provides direct access to Cross Encoder scoring for custom applications
        with comprehensive error handling.
        
        Args:
            pairs (List[tuple]): List of (query, document) tuples to score
            
        Returns:
            List[float]: Relevance scores for each pair
        """
        if not pairs:
            return []
        
        try:
            scores = self.model.predict(pairs)
            return [float(score) for score in scores]
        except Exception as e:
            print(f"Scoring error: {e}")
            return [0.0] * len(pairs)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return comprehensive model information and configuration.
        
        Provides detailed information about the loaded model for debugging,
        monitoring, and configuration validation purposes.
        
        Returns:
            Dict[str, Any]: Complete model configuration and status information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "model_type": "CrossEncoder",
            "framework": "sentence-transformers"
        }

    def benchmark_speed(self, query: str, documents: List[str], num_runs: int = 3) -> Dict[str, float]:
        """
        Benchmark reranking performance across multiple runs.
        
        Provides comprehensive performance analysis including timing statistics,
        throughput metrics, and device-specific performance characteristics.
        
        Args:
            query (str): Test query for benchmarking
            documents (List[str]): List of test documents
            num_runs (int): Number of execution runs for averaging
            
        Returns:
            Dict[str, float]: Comprehensive performance statistics
        """
        if not documents:
            return {"error": "No documents for benchmarking"}
        
        pairs = [(query, doc) for doc in documents]
        times = []
        
        print(f"CrossEncoder benchmark: {len(documents)} documents, {num_runs} runs")
        
        for run in range(num_runs):
            start_time = time.time()
            try:
                _ = self.model.predict(pairs)
                execution_time = time.time() - start_time
                times.append(execution_time)
                print(f"   Run {run+1}: {execution_time:.3f}s")
            except Exception as e:
                print(f"   Run {run+1}: ERROR - {e}")
                continue
        
        if not times:
            return {"error": "All runs failed"}
        
        return {
            "avg_time_seconds": np.mean(times),
            "min_time_seconds": np.min(times),
            "max_time_seconds": np.max(times),
            "std_time_seconds": np.std(times),
            "documents_per_second": len(documents) / np.mean(times),
            "device": self.device,
            "num_documents": len(documents),
            "num_runs": len(times)
        }