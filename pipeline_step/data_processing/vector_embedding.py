import os
import json
import yaml
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from sentence_transformers import SentenceTransformer

# === FONCTION POUR CHARGER LE YAML ===
def load_settings(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis le fichier YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# === CHARGEMENT DES DOCUMENTS CHUNKÉS ===
def load_chunked_documents(processed_dir: Path) -> List[Dict[str, Any]]:
    """Charge tous les chunks depuis les fichiers JSON"""
    print(f"🔍 Chargement des chunks depuis: {processed_dir}")
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Répertoire des chunks introuvable: {processed_dir}")
    
    documents = []
    chunk_files = list(processed_dir.glob("*_chunks.json"))
    
    if not chunk_files:
        raise FileNotFoundError(f"Aucun fichier chunks trouvé dans: {processed_dir}")
    
    print(f"📄 {len(chunk_files)} fichier(s) chunks trouvé(s)")
    
    for json_file in chunk_files:
        print(f"   📝 Lecture: {json_file.name}")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Validation de la structure
            if "chunks" not in data:
                print(f"   ⚠️ Champ 'chunks' manquant dans {json_file.name}")
                continue
            
            # Chargement des chunks
            doc_chunks = 0
            for chunk in data["chunks"]:
                # Validation des champs requis
                required_fields = ["chunk_id", "text"]
                if not all(field in chunk for field in required_fields):
                    print(f"   ⚠️ Chunk invalide dans {json_file.name}")
                    continue
                
                documents.append({
                    "document_id": data.get("document_id", json_file.stem),
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "word_count": chunk.get("word_count", len(chunk["text"].split())),
                    "char_count": chunk.get("char_count", len(chunk["text"])),
                    "source_file": json_file.name
                })
                doc_chunks += 1
            
            print(f"   ✅ {doc_chunks} chunks chargés")
            
        except Exception as e:
            print(f"   ❌ Erreur lecture {json_file.name}: {e}")
            continue
    
    print(f"📊 Total: {len(documents)} chunks chargés")
    return documents

# === GÉNÉRATION DES EMBEDDINGS AVEC SentenceTransformer ===
def generate_embeddings(documents: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    """Génère les embeddings pour tous les chunks"""
    print(f"\n🤖 Génération des embeddings avec: {model_name}")
    
    if not documents:
        raise ValueError("Aucun document à traiter")
    
    # Chargement du modèle
    print("📥 Chargement du modèle...")
    try:
        model = SentenceTransformer(model_name)
        print(f"✅ Modèle chargé: {model_name}")
        print(f"📏 Dimension des embeddings: {model.get_sentence_embedding_dimension()}")
    except Exception as e:
        raise RuntimeError(f"Erreur chargement modèle {model_name}: {e}")
    
    # Préparation des textes
    texts = [doc["text"] for doc in documents]
    print(f"📝 {len(texts)} textes à encoder...")
    
    # Statistiques des textes
    text_lengths = [len(text.split()) for text in texts]
    print(f"📊 Longueur mots - Min: {min(text_lengths)}, Max: {max(text_lengths)}, Moy: {np.mean(text_lengths):.1f}")
    
    # Génération des embeddings
    print("🔄 Génération des embeddings en cours...")
    try:
        vectors = model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=32,
            convert_to_tensor=False,  # Return numpy arrays
            normalize_embeddings=True  # Normaliser pour le cosine similarity
        )
        print(f"✅ {len(vectors)} embeddings générés")
        print(f"📏 Shape des embeddings: {vectors.shape}")
        
    except Exception as e:
        raise RuntimeError(f"Erreur génération embeddings: {e}")
    
    # Enrichissement des documents avec les embeddings
    print("🔗 Association embeddings -> documents...")
    for doc, vec in zip(documents, vectors):
        doc["embedding"] = vec.tolist()  # Convert numpy to list for serialization
        doc["embedding_dim"] = len(vec)
        doc["embedding_norm"] = float(np.linalg.norm(vec))
    
    return documents

# === SAUVEGARDE DES EMBEDDINGS ===
def save_embeddings(embeddings: List[Dict[str, Any]], path: Path):
    """Sauvegarde les embeddings avec métadonnées"""
    print(f"\n💾 Sauvegarde des embeddings...")
    
    # Créer le répertoire parent si nécessaire
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Préparation des métadonnées
    metadata = {
        "created_at": datetime.now().isoformat(),
        "num_documents": len(embeddings),
        "embedding_dimension": embeddings[0]["embedding_dim"] if embeddings else 0,
        "total_chunks": len(embeddings),
        "documents_info": {}
    }
    
    # Statistiques par document
    doc_stats = {}
    for emb in embeddings:
        doc_id = emb["document_id"]
        if doc_id not in doc_stats:
            doc_stats[doc_id] = {"chunks": 0, "total_words": 0}
        doc_stats[doc_id]["chunks"] += 1
        doc_stats[doc_id]["total_words"] += emb.get("word_count", 0)
    
    metadata["documents_info"] = doc_stats
    
    # Structure finale à sauvegarder
    final_data = {
        "metadata": metadata,
        "embeddings": embeddings
    }
    
    # Sauvegarde
    try:
        with open(path, "wb") as f:
            pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✅ Embeddings sauvegardés: {path}")
        print(f"📁 Taille fichier: {file_size_mb:.1f} MB")
        print(f"📊 {len(embeddings)} embeddings sauvegardés")
        
    except Exception as e:
        raise RuntimeError(f"Erreur sauvegarde embeddings: {e}")

def analyze_embeddings_quality(embeddings: List[Dict[str, Any]]):
    """Analyse la qualité des embeddings générés"""
    print(f"\n📊 ANALYSE DE QUALITÉ DES EMBEDDINGS")
    print("=" * 50)
    
    if not embeddings:
        print("❌ Aucun embedding à analyser")
        return
    
    # Statistiques générales
    embedding_norms = [emb["embedding_norm"] for emb in embeddings]
    text_lengths = [emb.get("word_count", 0) for emb in embeddings]
    
    print(f"📏 Normes des embeddings:")
    print(f"   Min: {min(embedding_norms):.3f}")
    print(f"   Max: {max(embedding_norms):.3f}")
    print(f"   Moyenne: {np.mean(embedding_norms):.3f}")
    print(f"   Std: {np.std(embedding_norms):.3f}")
    
    print(f"\n📝 Longueurs des textes:")
    print(f"   Min: {min(text_lengths)} mots")
    print(f"   Max: {max(text_lengths)} mots")
    print(f"   Moyenne: {np.mean(text_lengths):.1f} mots")
    
    # Détection d'anomalies
    zero_embeddings = sum(1 for emb in embeddings if emb["embedding_norm"] < 0.001)
    if zero_embeddings > 0:
        print(f"⚠️ {zero_embeddings} embeddings proches de zéro détectés!")
    
    empty_texts = sum(1 for emb in embeddings if len(emb["text"].strip()) == 0)
    if empty_texts > 0:
        print(f"⚠️ {empty_texts} textes vides détectés!")
    
    if zero_embeddings == 0 and empty_texts == 0:
        print("✅ Qualité des embeddings: OK")

# === MAIN ===
def main():
    print("🚀 GÉNÉRATION DES EMBEDDINGS")
    print("=" * 50)
    
    try:
        # Chargement de la configuration
        settings = load_settings("config/settings.yaml")
        processed_dir = Path(settings["paths"]["chunk_documents"])
        embedding_path = Path(settings["paths"]["embedding_file"])
        model_name = settings["models"]["embedding_model"]
        
        print(f"🔧 Configuration:")
        print(f"   📁 Chunks: {processed_dir}")
        print(f"   💾 Embeddings: {embedding_path}")
        print(f"   🤖 Modèle: {model_name}")
        
        # Traitement
        documents = load_chunked_documents(processed_dir)
        
        if not documents:
            print("❌ Aucun document à traiter!")
            return
        
        enriched_docs = generate_embeddings(documents, model_name)
        analyze_embeddings_quality(enriched_docs)
        save_embeddings(enriched_docs, embedding_path)
        
        print(f"\n🎉 SUCCÈS!")
        print(f"✅ {len(enriched_docs)} embeddings générés et sauvegardés")
        print(f"📁 Fichier: {embedding_path}")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()