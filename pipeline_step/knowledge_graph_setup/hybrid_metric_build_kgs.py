"""
Autonomous Hybrid Metric for Knowledge Graph Densification

This module implements an autonomous hybrid similarity metric that combines Cosine,
Jaccard, and Levenshtein similarities for Knowledge Graph densification. The metric
operates without external dependencies beyond standard ML libraries and provides
comprehensive text similarity analysis for diagnostic symptoms and error codes.

Key components:
- AutonomousHybridMetric: Main class implementing the hybrid similarity calculation
- Multi-component similarity: Combines semantic (cosine), lexical (Jaccard), and error code (Levenshtein) similarities
- Autonomous operation: Self-contained implementation with lazy loading and efficient computation

Dependencies: numpy, sentence-transformers, scikit-learn, python-Levenshtein
Usage: Primary component for Knowledge Graph construction and symptom similarity assessment
"""

import numpy as np
import re
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance

class AutonomousHybridMetric:
    """Autonomous hybrid metric for Knowledge Graph densification and similarity assessment"""
    
    def __init__(self, weights=None):
        """
        Initialize the autonomous hybrid metric with configurable component weights
        
        Args:
            weights (dict, optional): Dictionary containing weight parameters for the three components.
                Expected keys: 'cosine_alpha', 'jaccard_beta', 'levenshtein_gamma'.
                Defaults to balanced weights if not provided.
        """
        self.weights = weights or {
            'cosine_alpha': 0.4,
            'jaccard_beta': 0.4,
            'levenshtein_gamma': 0.2
        }
        self.model = None
        print(f"Autonomous Hybrid Metric initialized")
        print(f"Weights: Cosine={self.weights['cosine_alpha']}, Jaccard={self.weights['jaccard_beta']}, Levenshtein={self.weights['levenshtein_gamma']}")
    
    def _get_model(self):
        """
        Lazy loading of the sentence embedding model
        
        Implements lazy initialization to avoid loading the model until needed,
        reducing memory usage and startup time.
        
        Returns:
            SentenceTransformer: Initialized embedding model
        """
        if self.model is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.model
    
    def _tokenize(self, text: str) -> set:
        """
        Simple tokenization for Jaccard similarity calculation
        
        Performs text preprocessing including lowercasing, punctuation removal,
        and token extraction while preserving alphanumeric characters and hyphens.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            set: Set of unique tokens extracted from the text
        """
        # Cleaning and tokenization
        text = text.lower()
        # Remove punctuation, keep letters, numbers, spaces and hyphens
        text = re.sub(r'[^\w\s-]', ' ', text)
        # Split on spaces and hyphens
        tokens = set(text.split())
        # Remove empty tokens
        return {token for token in tokens if token.strip()}
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts
        
        Computes the Jaccard index as the ratio of intersection to union
        of token sets, providing a lexical similarity measure.
        
        Args:
            text1 (str): First text for comparison
            text2 (str): Second text for comparison
            
        Returns:
            float: Jaccard similarity score between 0 and 1
        """
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        
        if not tokens1 and not tokens2:
            return 1.0  # Two empty texts are identical
        
        if not tokens1 or not tokens2:
            return 0.0  # One empty and one non-empty text
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_error_codes(self, text: str) -> List[str]:
        """
        Extract error codes from text using pattern matching
        
        Identifies standardized error codes (e.g., ACAL-006, SYST-001) commonly
        found in industrial diagnostic systems and equipment manuals.
        
        Args:
            text (str): Input text potentially containing error codes
            
        Returns:
            List[str]: List of extracted error codes
        """
        pattern = r'\b[A-Z]{3,5}-\d{3,4}\b'
        return re.findall(pattern, text.upper())
    
    def _levenshtein_similarity(self, text1: str, text2: str, threshold: int = 3) -> float:
        """
        Calculate Levenshtein similarity for error codes
        
        Computes edit distance-based similarity specifically for error codes,
        useful for identifying related diagnostic codes that may have minor variations.
        
        Args:
            text1 (str): First text containing potential error codes
            text2 (str): Second text containing potential error codes
            threshold (int): Maximum edit distance to consider as similar
            
        Returns:
            float: Similarity score between 0 and 1 (1 = very similar)
        """
        codes1 = self._extract_error_codes(text1)
        codes2 = self._extract_error_codes(text2)
        
        if not codes1 or not codes2:
            return 0.0  # No error codes found
        
        # Minimum distance between all code pairs
        min_distance = float('inf')
        for code1 in codes1:
            for code2 in codes2:
                dist = levenshtein_distance(code1, code2)
                min_distance = min(min_distance, dist)
        
        # Convert distance to similarity
        if min_distance <= threshold:
            return 1.0 - (min_distance / threshold)
        else:
            return 0.0
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Main function: Compute hybrid similarity matrix for a list of texts
        
        Calculates a comprehensive similarity matrix by combining three different
        similarity metrics: semantic (cosine), lexical (Jaccard), and structural
        (Levenshtein for error codes). The final matrix represents weighted
        similarity scores suitable for Knowledge Graph construction.
        
        Args:
            texts (List[str]): List of texts (symptoms or symptom+cause combinations)
            
        Returns:
            np.ndarray: Hybrid similarity matrix with shape (n_texts, n_texts)
        """
        n_texts = len(texts)
        print(f"Computing autonomous hybrid metric for {n_texts} texts...")
        
        # 1. Cosine Similarity Component
        print("Computing cosine similarity...")
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=True)
        cosine_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(cosine_matrix, 0)  # Avoid self-similarity
        
        # 2. Jaccard Similarity Component
        print("Computing Jaccard similarity...")
        jaccard_matrix = np.zeros((n_texts, n_texts))
        
        for i in range(n_texts):
            if (i + 1) % 50 == 0:
                print(f"   Jaccard: {i + 1}/{n_texts} texts processed...")
            
            for j in range(i + 1, n_texts):  # Symmetric matrix
                jaccard_score = self._jaccard_similarity(texts[i], texts[j])
                jaccard_matrix[i][j] = jaccard_score
                jaccard_matrix[j][i] = jaccard_score
        
        # 3. Levenshtein Similarity Component
        print("Computing Levenshtein similarity...")
        levenshtein_matrix = np.zeros((n_texts, n_texts))
        
        for i in range(n_texts):
            if (i + 1) % 100 == 0:
                print(f"   Levenshtein: {i + 1}/{n_texts} texts processed...")
            
            for j in range(i + 1, n_texts):  # Symmetric matrix
                lev_score = self._levenshtein_similarity(texts[i], texts[j])
                levenshtein_matrix[i][j] = lev_score
                levenshtein_matrix[j][i] = lev_score
        
        # 4. Weighted Fusion
        print("Performing weighted fusion of metrics...")
        
        # Optional normalization (all metrics are already 0-1)
        hybrid_matrix = (
            self.weights['cosine_alpha'] * cosine_matrix +
            self.weights['jaccard_beta'] * jaccard_matrix +
            self.weights['levenshtein_gamma'] * levenshtein_matrix
        )
        
        print(f"Hybrid matrix computed: {hybrid_matrix.shape}")
        
        # Statistics
        non_zero = np.count_nonzero(hybrid_matrix)
        total_possible = n_texts * (n_texts - 1)
        print(f"Non-zero connections: {non_zero}/{total_possible} ({non_zero/total_possible*100:.1f}%)")
        print(f"Average score: {hybrid_matrix.mean():.3f}")
        print(f"Maximum score: {hybrid_matrix.max():.3f}")
        
        return hybrid_matrix

def create_autonomous_hybrid_metric(weights=None):
    """
    Factory function to create an autonomous hybrid metric instance
    
    Args:
        weights (dict, optional): Component weights for the hybrid metric
        
    Returns:
        AutonomousHybridMetric: Configured hybrid metric instance
    """
    return AutonomousHybridMetric(weights)

def compute_hybrid_similarity_matrix(texts: List[str], weights=None) -> np.ndarray:
    """
    Utility function for rapid similarity matrix computation
    
    Provides a streamlined interface for computing hybrid similarity matrices
    without requiring explicit metric instantiation.
    
    Args:
        texts (List[str]): Input texts for similarity analysis
        weights (dict, optional): Component weights for the hybrid metric
        
    Returns:
        np.ndarray: Computed hybrid similarity matrix
    """
    metric = create_autonomous_hybrid_metric(weights)
    return metric.compute_similarity_matrix(texts)

if __name__ == "__main__":
    # Simple test case
    test_texts = [
        "motor overheating FANUC R-30iB error ACAL-006",
        "motor temperature high FANUC robot ACAL-007",
        "servo motor problem KUKA KR C4",
        "hydraulic pump failure ABB IRC5"
    ]
    
    print("Testing autonomous hybrid metric")
    print("-" * 50)
    
    metric = create_autonomous_hybrid_metric()
    similarity_matrix = metric.compute_similarity_matrix(test_texts)
    
    print("\nSimilarity matrix results:")
    for i, text1 in enumerate(test_texts):
        for j, text2 in enumerate(test_texts):
            if i < j:  # Display only upper triangle
                score = similarity_matrix[i][j]
                print(f"{i+1}-{j+1}: {score:.3f}")
                print(f"   '{text1[:30]}...' <-> '{text2[:30]}...'")
                print()