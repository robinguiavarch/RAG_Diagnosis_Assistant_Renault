"""
Métrique Hybride Autonome pour densification KG
Combine Cosine + Jaccard + Levenshtein sans dépendances externes
"""

import numpy as np
import re
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance

class AutonomousHybridMetric:
    """Métrique hybride autonome pour densification des Knowledge Graphs"""
    
    def __init__(self, weights=None):
        """
        Args:
            weights: dict avec clés 'cosine_alpha', 'jaccard_beta', 'levenshtein_gamma'
        """
        self.weights = weights or {
            'cosine_alpha': 0.4,
            'jaccard_beta': 0.4,
            'levenshtein_gamma': 0.2
        }
        self.model = None
        print(f"🔧 Métrique Hybride Autonome initialisée")
        print(f"⚖️ Poids: Cosine={self.weights['cosine_alpha']}, Jaccard={self.weights['jaccard_beta']}, Levenshtein={self.weights['levenshtein_gamma']}")
    
    def _get_model(self):
        """Lazy loading du modèle d'embedding"""
        if self.model is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self.model
    
    def _tokenize(self, text: str) -> set:
        """Tokenization simple pour Jaccard"""
        # Nettoyage et tokenization
        text = text.lower()
        # Supprime ponctuation, garde lettres, chiffres, espaces et tirets
        text = re.sub(r'[^\w\s-]', ' ', text)
        # Split sur espaces et tirets
        tokens = set(text.split())
        # Enlève tokens vides
        return {token for token in tokens if token.strip()}
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité de Jaccard entre deux textes"""
        tokens1 = self._tokenize(text1)
        tokens2 = self._tokenize(text2)
        
        if not tokens1 and not tokens2:
            return 1.0  # Deux textes vides sont identiques
        
        if not tokens1 or not tokens2:
            return 0.0  # Un texte vide et un non-vide
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_error_codes(self, text: str) -> List[str]:
        """Extrait les codes d'erreur (ex: ACAL-006, SYST-001)"""
        pattern = r'\b[A-Z]{3,5}-\d{3,4}\b'
        return re.findall(pattern, text.upper())
    
    def _levenshtein_similarity(self, text1: str, text2: str, threshold: int = 3) -> float:
        """
        Calcule la similarité Levenshtein pour codes d'erreur
        
        Args:
            text1, text2: Textes à comparer
            threshold: Distance max pour considérer comme similaire
            
        Returns:
            float: Score 0-1 (1 = très similaire)
        """
        codes1 = self._extract_error_codes(text1)
        codes2 = self._extract_error_codes(text2)
        
        if not codes1 or not codes2:
            return 0.0  # Pas de codes d'erreur
        
        # Distance minimale entre tous les codes
        min_distance = float('inf')
        for code1 in codes1:
            for code2 in codes2:
                dist = levenshtein_distance(code1, code2)
                min_distance = min(min_distance, dist)
        
        # Conversion distance → similarité
        if min_distance <= threshold:
            return 1.0 - (min_distance / threshold)
        else:
            return 0.0
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        🎯 FONCTION PRINCIPALE
        Calcule la matrice de similarité hybride pour une liste de textes
        
        Args:
            texts: Liste des textes (symptômes ou symptômes+causes)
            
        Returns:
            np.ndarray: Matrice de similarité hybride
        """
        n_texts = len(texts)
        print(f"🧠 Calcul métrique hybride autonome pour {n_texts} textes...")
        
        # 1. Composante Cosine Similarity
        print("🔄 Calcul cosine similarity...")
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=True)
        cosine_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(cosine_matrix, 0)  # Éviter auto-similarité
        
        # 2. Composante Jaccard Similarity
        print("🔄 Calcul Jaccard similarity...")
        jaccard_matrix = np.zeros((n_texts, n_texts))
        
        for i in range(n_texts):
            if (i + 1) % 50 == 0:
                print(f"   • Jaccard: {i + 1}/{n_texts} textes traités...")
            
            for j in range(i + 1, n_texts):  # Matrice symétrique
                jaccard_score = self._jaccard_similarity(texts[i], texts[j])
                jaccard_matrix[i][j] = jaccard_score
                jaccard_matrix[j][i] = jaccard_score
        
        # 3. Composante Levenshtein Similarity
        print("🔄 Calcul Levenshtein similarity...")
        levenshtein_matrix = np.zeros((n_texts, n_texts))
        
        for i in range(n_texts):
            if (i + 1) % 100 == 0:
                print(f"   • Levenshtein: {i + 1}/{n_texts} textes traités...")
            
            for j in range(i + 1, n_texts):  # Matrice symétrique
                lev_score = self._levenshtein_similarity(texts[i], texts[j])
                levenshtein_matrix[i][j] = lev_score
                levenshtein_matrix[j][i] = lev_score
        
        # 4. Fusion pondérée
        print("🔄 Fusion pondérée des métriques...")
        
        # Normalisation optionnelle (toutes les métriques sont déjà 0-1)
        hybrid_matrix = (
            self.weights['cosine_alpha'] * cosine_matrix +
            self.weights['jaccard_beta'] * jaccard_matrix +
            self.weights['levenshtein_gamma'] * levenshtein_matrix
        )
        
        print(f"✅ Matrice hybride calculée : {hybrid_matrix.shape}")
        
        # Statistiques
        non_zero = np.count_nonzero(hybrid_matrix)
        total_possible = n_texts * (n_texts - 1)
        print(f"📊 Connexions non-nulles: {non_zero}/{total_possible} ({non_zero/total_possible*100:.1f}%)")
        print(f"📊 Score moyen: {hybrid_matrix.mean():.3f}")
        print(f"📊 Score max: {hybrid_matrix.max():.3f}")
        
        return hybrid_matrix

# === FONCTIONS UTILITAIRES ===

def create_autonomous_hybrid_metric(weights=None):
    """Crée une instance de métrique hybride autonome"""
    return AutonomousHybridMetric(weights)

def compute_hybrid_similarity_matrix(texts: List[str], weights=None) -> np.ndarray:
    """Fonction utilitaire pour calcul rapide"""
    metric = create_autonomous_hybrid_metric(weights)
    return metric.compute_similarity_matrix(texts)

# === TEST ===
if __name__ == "__main__":
    # Test simple
    test_texts = [
        "motor overheating FANUC R-30iB error ACAL-006",
        "motor temperature high FANUC robot ACAL-007",
        "servo motor problem KUKA KR C4",
        "hydraulic pump failure ABB IRC5"
    ]
    
    print("🧪 Test métrique hybride autonome")
    print("-" * 50)
    
    metric = create_autonomous_hybrid_metric()
    similarity_matrix = metric.compute_similarity_matrix(test_texts)
    
    print("\n📊 Matrice de similarité :")
    for i, text1 in enumerate(test_texts):
        for j, text2 in enumerate(test_texts):
            if i < j:  # Afficher seulement triangle supérieur
                score = similarity_matrix[i][j]
                print(f"{i+1}-{j+1}: {score:.3f}")
                print(f"   '{text1[:30]}...' ↔ '{text2[:30]}...'")
                print()