"""
Hybrid Symptom Matcher - Version Simple CORRIGÉE
Combine BM25 + FAISS + Levenshtein pour recherche optimisée des symptômes
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
    """Matcher hybride pour symptômes: BM25 + FAISS + Levenshtein"""
    
    def __init__(self):
        self.config = self._load_config()
        self.weights = self.config["hybrid_symptom_search"]["weights"]
        self.enabled = self.config["hybrid_symptom_search"]["enabled"]
        
        # Composants (chargés de façon paresseuse)
        self.bm25_index = None
        self.faiss_index = None
        self.faiss_metadata = None
        self.embedding_model = None
        
        print(f"🔧 HybridSymptomMatcher initialisé (enabled: {self.enabled})")
        print(f"⚖️ Poids: BM25={self.weights['bm25_alpha']}, FAISS={self.weights['faiss_beta']}, Levenshtein={self.weights['levenshtein_gamma']}")
    
    def _load_config(self):
        """Charge la configuration"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "..", "..", "config", "settings.yaml")
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def _get_bm25_index(self):
        """Charge l'index BM25 des symptômes"""
        if self.bm25_index is None:
            bm25_path = self.config["hybrid_symptom_search"]["bm25_index_path"]
            if os.path.exists(bm25_path):
                self.bm25_index = open_dir(bm25_path)
                print("✅ Index BM25 symptômes chargé")
            else:
                print(f"⚠️ Index BM25 non trouvé: {bm25_path}")
        return self.bm25_index
    
    def _get_faiss_components(self):
        """Charge l'index FAISS et métadonnées"""
        if self.faiss_index is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 🔧 CORRECTION DU CHEMIN : symptom_embeddings_dense → symptom_embedding_dense
            embedding_dir = os.path.join(script_dir, "..", "..", "data", "knowledge_base", "symptom_embedding_dense")
            
            index_path = os.path.join(embedding_dir, "index.faiss")
            metadata_path = os.path.join(embedding_dir, "symptom_embedding_dense.pkl")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self.faiss_index = faiss.read_index(index_path)
                with open(metadata_path, "rb") as f:
                    self.faiss_metadata = pickle.load(f)
                print("✅ Index FAISS symptômes chargé")
            else:
                print(f"⚠️ Index FAISS non trouvé: {embedding_dir}")
        
        return self.faiss_index, self.faiss_metadata
    
    def _get_embedding_model(self):
        """Charge le modèle d'embedding"""
        if self.embedding_model is None:
            model_name = self.config["models"]["embedding_model"]
            self.embedding_model = SentenceTransformer(model_name)
        return self.embedding_model
    
    def _search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        """Recherche BM25 dans les symptômes"""
        bm25_index = self._get_bm25_index()
        if not bm25_index:
            return []
        
        try:
            with bm25_index.searcher(weighting=scoring.BM25F()) as searcher:
                parser = QueryParser("symptom_text", schema=bm25_index.schema)
                
                # Nettoyage requête
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
            print(f"❌ Erreur BM25: {e}")
            return []
    
    def _search_faiss(self, query: str, top_k: int = 10) -> List[Dict]:
        """Recherche FAISS dans les symptômes"""
        faiss_index, metadata = self._get_faiss_components()
        if not faiss_index or not metadata:
            return []
        
        try:
            model = self._get_embedding_model()
            symptom_names = metadata['symptom_names']
            
            # Embedding de la requête
            query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
            scores, indices = faiss_index.search(query_vec, top_k)
            
            faiss_results = []
            for i, score in zip(indices[0], scores[0]):
                if i < len(symptom_names):
                    faiss_results.append({
                        'symptom_text': symptom_names[i],
                        'symptom_id': str(i),
                        'equipment': 'unknown',  # Equipment pas dans metadata FAISS
                        'faiss_score': float(score),
                        'source': 'FAISS'
                    })
            
            return faiss_results
            
        except Exception as e:
            print(f"❌ Erreur FAISS: {e}")
            return []
    
    def _extract_error_codes(self, text: str) -> List[str]:
        """Extrait les codes d'erreur du type ACAL-006, SYST-001, etc."""
        pattern = r'\b[A-Z]{3,5}-\d{3,4}\b'
        return re.findall(pattern, text.upper())
    
    def _search_levenshtein(self, query: str, all_symptoms: List[str], top_k: int = 10) -> List[Dict]:
        """Recherche par distance Levenshtein pour codes d'erreur"""
        query_codes = self._extract_error_codes(query)
        if not query_codes:
            return []
        
        threshold = self.config["hybrid_symptom_search"]["levenshtein_threshold"]
        
        levenshtein_results = []
        for symptom in all_symptoms:
            symptom_codes = self._extract_error_codes(symptom)
            
            if symptom_codes:
                # Distance minimale entre codes de la requête et du symptôme
                min_distance = float('inf')
                for q_code in query_codes:
                    for s_code in symptom_codes:
                        dist = levenshtein_distance(q_code, s_code)
                        min_distance = min(min_distance, dist)
                
                # Score inversé: distance faible = score élevé
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
        
        # Tri par score décroissant
        levenshtein_results.sort(key=lambda x: x['levenshtein_score'], reverse=True)
        return levenshtein_results[:top_k]
    
    def search_hybrid(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        🎯 RECHERCHE HYBRIDE PRINCIPALE
        Combine BM25 + FAISS + Levenshtein avec fusion pondérée
        """
        if not self.enabled:
            print("🔄 Recherche hybride désactivée, fallback FAISS")
            return self._search_faiss(query, top_k)
        
        print(f"🔍 Recherche hybride pour: '{query}'")
        
        # 1. Recherches par composante
        bm25_results = self._search_bm25(query, top_k * 2)
        faiss_results = self._search_faiss(query, top_k * 2)
        
        # Pour Levenshtein, on a besoin de tous les symptômes
        all_symptoms = []
        if self.faiss_metadata:
            all_symptoms = self.faiss_metadata.get('symptom_names', [])
        
        levenshtein_results = self._search_levenshtein(query, all_symptoms, top_k)
        
        print(f"📊 Résultats bruts: BM25={len(bm25_results)}, FAISS={len(faiss_results)}, Levenshtein={len(levenshtein_results)}")
        
        # 2. Fusion pondérée
        combined_scores = {}
        
        # Normalisation et fusion BM25
        if bm25_results:
            max_bm25 = max(r['bm25_score'] for r in bm25_results)
            for result in bm25_results:
                symptom = result['symptom_text']
                norm_score = result['bm25_score'] / max_bm25 if max_bm25 > 0 else 0
                combined_scores[symptom] = combined_scores.get(symptom, 0) + self.weights['bm25_alpha'] * norm_score
        
        # Normalisation et fusion FAISS
        if faiss_results:
            max_faiss = max(r['faiss_score'] for r in faiss_results)
            for result in faiss_results:
                symptom = result['symptom_text']
                norm_score = result['faiss_score'] / max_faiss if max_faiss > 0 else 0
                combined_scores[symptom] = combined_scores.get(symptom, 0) + self.weights['faiss_beta'] * norm_score
        
        # Fusion Levenshtein (déjà normalisé 0-1)
        for result in levenshtein_results:
            symptom = result['symptom_text']
            combined_scores[symptom] = combined_scores.get(symptom, 0) + self.weights['levenshtein_gamma'] * result['levenshtein_score']
        
        # 3. Tri final et formatage
        final_results = []
        for symptom, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            final_results.append({
                'symptom_text': symptom,
                'hybrid_score': score,
                'source': 'Hybrid'
            })
        
        print(f"✅ Top {len(final_results)} symptômes hybrides sélectionnés")
        return final_results

# === FONCTION UTILITAIRE ===

def create_hybrid_symptom_matcher() -> HybridSymptomMatcher:
    """Crée un matcher hybride"""
    return HybridSymptomMatcher()

def search_symptoms_hybrid(query: str, top_k: int = 5) -> List[Dict]:
    """Fonction utilitaire pour recherche rapide"""
    matcher = create_hybrid_symptom_matcher()
    return matcher.search_hybrid(query, top_k)

# === TEST ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Test recherche hybride: {query}")
        print("-" * 50)
        
        matcher = create_hybrid_symptom_matcher()
        results = matcher.search_hybrid(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['symptom_text']} (score: {result['hybrid_score']:.3f})")
    else:
        print("Usage: python hybrid_symptom_matcher.py 'votre requête'")