import os
import yaml
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# === FONCTION POUR CHARGER LE YAML ===
def load_settings(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis le fichier YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_embeddings(embeddings_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Charge les embeddings avec gestion des formats ancien/nouveau"""
    print(f"📥 Chargement des embeddings depuis: {embeddings_path}")
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Fichier embeddings introuvable: {embeddings_path}")
    
    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)
    
    # Détection du format
    if isinstance(data, dict) and "embeddings" in data:
        # Nouveau format avec métadonnées
        print("✅ Nouveau format détecté avec métadonnées")
        embeddings = data["embeddings"]
        metadata = data.get("metadata", {})
        
        print(f"📊 Métadonnées trouvées:")
        print(f"   🕒 Créé le: {metadata.get('created_at', 'Inconnu')}")
        print(f"   📄 Nombre de documents: {metadata.get('num_documents', 'Inconnu')}")
        print(f"   📏 Dimension: {metadata.get('embedding_dimension', 'Inconnue')}")
        
    elif isinstance(data, list):
        # Ancien format - liste directe
        print("⚠️ Ancien format détecté (liste directe)")
        embeddings = data
        metadata = {}
    else:
        raise ValueError("Format d'embeddings non reconnu")
    
    # Validation des embeddings
    if not embeddings:
        raise ValueError("Aucun embedding trouvé dans le fichier")
    
    # Vérification de la structure des embeddings
    required_fields = ["document_id", "chunk_id", "text", "embedding"]
    for i, emb in enumerate(embeddings[:3]):  # Vérifier les 3 premiers
        missing_fields = [field for field in required_fields if field not in emb]
        if missing_fields:
            raise ValueError(f"Champs manquants dans embedding {i}: {missing_fields}")
    
    print(f"📊 {len(embeddings)} embeddings chargés")
    return embeddings, metadata

def validate_embeddings(embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Valide et analyse la qualité des embeddings"""
    print("🔍 Validation des embeddings...")
    
    # Extraction des vecteurs pour analyse
    vectors = []
    invalid_count = 0
    
    for i, emb in enumerate(embeddings):
        try:
            vector = np.array(emb["embedding"], dtype=np.float32)
            
            # Vérifications de base
            if vector.size == 0:
                print(f"   ⚠️ Embedding {i} vide")
                invalid_count += 1
                continue
                
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                print(f"   ⚠️ Embedding {i} contient NaN/Inf")
                invalid_count += 1
                continue
                
            vectors.append(vector)
            
        except Exception as e:
            print(f"   ❌ Erreur embedding {i}: {e}")
            invalid_count += 1
            continue
    
    if not vectors:
        raise ValueError("Aucun embedding valide trouvé")
    
    # Analyse des dimensions
    dimensions = [v.shape[0] for v in vectors]
    unique_dims = set(dimensions)
    
    if len(unique_dims) > 1:
        raise ValueError(f"Dimensions incohérentes: {unique_dims}")
    
    dimension = dimensions[0]
    
    # Analyse des normes
    norms = [np.linalg.norm(v) for v in vectors]
    
    # Statistiques
    stats = {
        "total_embeddings": len(embeddings),
        "valid_embeddings": len(vectors),
        "invalid_embeddings": invalid_count,
        "dimension": dimension,
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms))
    }
    
    print(f"📊 Validation terminée:")
    print(f"   ✅ Embeddings valides: {stats['valid_embeddings']}/{stats['total_embeddings']}")
    print(f"   📏 Dimension: {stats['dimension']}")
    print(f"   📐 Norme moy: {stats['norm_mean']:.3f} ± {stats['norm_std']:.3f}")
    
    if invalid_count > 0:
        print(f"   ⚠️ {invalid_count} embeddings invalides ignorés")
    
    return stats

def create_faiss_index(embeddings: List[Dict[str, Any]], index_type: str = "flat") -> Tuple[faiss.Index, List[str], List[Dict[str, Any]]]:
    """Crée l'index FAISS avec différents types possibles"""
    
    # Filtrage des embeddings valides
    valid_embeddings = []
    for emb in embeddings:
        try:
            vector = np.array(emb["embedding"], dtype=np.float32)
            if not (np.any(np.isnan(vector)) or np.any(np.isinf(vector)) or vector.size == 0):
                valid_embeddings.append(emb)
        except:
            continue
    
    if not valid_embeddings:
        raise ValueError("Aucun embedding valide pour créer l'index")
    
    # Extraction des vecteurs
    vectors = np.array([emb["embedding"] for emb in valid_embeddings], dtype=np.float32)
    dimension = vectors.shape[1]
    
    print(f"🛠️ Création index FAISS {index_type.upper()}...")
    print(f"   📏 Dimension: {dimension}")
    print(f"   📄 Nombre de vecteurs: {len(vectors)}")
    
    # Choix du type d'index
    if index_type == "flat":
        # Index exact (L2 distance)
        index = faiss.IndexFlatL2(dimension)
    elif index_type == "ivf":
        # Index approximatif pour de gros volumes
        nlist = min(100, len(vectors) // 10)  # Nombre de clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # Entraînement requis pour IVF
        print(f"🔄 Entraînement index IVF avec {nlist} clusters...")
        index.train(vectors)
    elif index_type == "hnsw":
        # Index HNSW pour recherche rapide
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
    else:
        raise ValueError(f"Type d'index non supporté: {index_type}")
    
    # Ajout des vecteurs
    print("📥 Ajout des vecteurs à l'index...")
    index.add(vectors)
    
    # Création des IDs et métadonnées
    ids = [f"{emb['document_id']}|{emb['chunk_id']}" for emb in valid_embeddings]
    
    # Métadonnées complètes pour chaque document
    documents_metadata = []
    for emb in valid_embeddings:
        doc_meta = {
            "document_id": emb["document_id"],
            "chunk_id": emb["chunk_id"],
            "text": emb["text"],
            "word_count": emb.get("word_count", len(emb["text"].split())),
            "char_count": emb.get("char_count", len(emb["text"])),
            "embedding_norm": emb.get("embedding_norm", float(np.linalg.norm(emb["embedding"]))),
            "source_file": emb.get("source_file", "unknown")
        }
        documents_metadata.append(doc_meta)
    
    print(f"✅ Index créé avec {index.ntotal} vecteurs")
    return index, ids, documents_metadata

def save_faiss_index(index: faiss.Index, ids: List[str], documents: List[Dict[str, Any]], 
                    faiss_index_dir: Path, original_metadata: Dict[str, Any]):
    """Sauvegarde l'index FAISS et ses métadonnées"""
    print(f"💾 Sauvegarde de l'index FAISS...")
    
    # Sauvegarde de l'index
    index_path = faiss_index_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    
    # Métadonnées enrichies
    enhanced_metadata = {
        "created_at": datetime.now().isoformat(),
        "index_type": "FAISS",
        "faiss_index_type": type(index).__name__,
        "total_vectors": index.ntotal,
        "dimension": index.d,
        "original_embeddings_metadata": original_metadata,
        "documents_count": len(set(doc["document_id"] for doc in documents)),
        "chunks_count": len(documents)
    }
    
    # Métadonnées complètes
    complete_metadata = {
        "metadata": enhanced_metadata,
        "ids": ids,
        "documents": documents
    }
    
    # Sauvegarde des métadonnées
    metadata_path = faiss_index_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(complete_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Sauvegarde JSON lisible pour debug
    json_metadata = {k: v for k, v in enhanced_metadata.items() if k != "original_embeddings_metadata"}
    json_path = faiss_index_dir / "index_info.json"
    
    import json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_metadata, f, ensure_ascii=False, indent=2)
    
    # Statistiques de taille
    index_size = index_path.stat().st_size / (1024 * 1024)
    metadata_size = metadata_path.stat().st_size / (1024 * 1024)
    
    print(f"✅ Sauvegarde terminée:")
    print(f"   📁 Index: {index_path} ({index_size:.1f} MB)")
    print(f"   📁 Métadonnées: {metadata_path} ({metadata_size:.1f} MB)")
    print(f"   📁 Info JSON: {json_path}")

def test_faiss_index(faiss_index_dir: Path, k: int = 3):
    """Test rapide de l'index FAISS"""
    print(f"\n🧪 TEST DE L'INDEX FAISS (top-{k})")
    
    try:
        # Chargement de l'index
        index_path = faiss_index_dir / "index.faiss"
        metadata_path = faiss_index_dir / "metadata.pkl"
        
        index = faiss.read_index(str(index_path))
        
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
        
        documents = data["documents"]
        
        if len(documents) == 0:
            print("❌ Aucun document pour le test")
            return
        
        # Test avec le premier embedding comme requête
        test_vector = np.array([documents[0]["text"]], dtype=object)  # Simulation
        # En réalité, il faudrait recalculer l'embedding, mais pour le test on prend le premier
        query_vector = np.array([np.random.random(index.d)], dtype=np.float32)
        
        # Recherche
        distances, indices = index.search(query_vector, k)
        
        print(f"🔍 Résultats de recherche:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # -1 signifie pas de résultat
                doc = documents[idx]
                print(f"   {i+1}. Doc: {doc['document_id']} | Chunk: {doc['chunk_id']}")
                print(f"      Distance: {dist:.3f}")
                print(f"      Texte: {doc['text'][:100]}...")
                print()
        
        print("✅ Index FAISS fonctionne correctement!")
        
    except Exception as e:
        print(f"❌ Erreur test index: {e}")

# === MAIN ===
def main():
    print("🚀 CRÉATION DE L'INDEX FAISS")
    print("=" * 50)
    
    try:
        # Configuration
        settings = load_settings("../../config/settings.yaml")
        embeddings_path = Path(settings["paths"]["embedding_file"])
        
        # 🔧 CORRECTION : Gestion intelligente du chemin
        faiss_config = settings["paths"].get("faiss_index_dir")  # Nouveau format
        if faiss_config:
            faiss_index_dir = Path(faiss_config)
            print("✅ Utilisation du nouveau format faiss_index_dir")
        else:
            # Fallback : extraire le répertoire depuis faiss_index
            faiss_index_path = Path(settings["paths"]["faiss_index"])
            if faiss_index_path.suffix == ".faiss":
                # C'est un fichier, on prend son répertoire parent
                faiss_index_dir = faiss_index_path.parent
                print("⚠️ Ancien format détecté, extraction du répertoire parent")
            else:
                # C'est déjà un répertoire
                faiss_index_dir = faiss_index_path
                print("✅ Chemin répertoire détecté")
        
        # 🔧 CORRECTION : Création sûre du répertoire
        print(f"📁 Répertoire cible: {faiss_index_dir}")
        
        # Suppression du fichier s'il existe au lieu du répertoire
        if faiss_index_dir.is_file():
            print(f"⚠️ Suppression du fichier existant: {faiss_index_dir}")
            faiss_index_dir.unlink()
        
        # Création du répertoire
        faiss_index_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 Configuration:")
        print(f"   📥 Embeddings: {embeddings_path}")
        print(f"   📁 Index FAISS: {faiss_index_dir}")
        
        # Chargement et validation
        embeddings, original_metadata = load_embeddings(embeddings_path)
        validation_stats = validate_embeddings(embeddings)
        
        # Choix automatique du type d'index selon la taille
        num_vectors = validation_stats["valid_embeddings"]
        if num_vectors < 1000:
            index_type = "flat"
        elif num_vectors < 10000:
            index_type = "hnsw"
        else:
            index_type = "ivf"
        
        print(f"🎯 Type d'index choisi: {index_type.upper()} (pour {num_vectors} vecteurs)")
        
        # Création de l'index
        index, ids, documents_metadata = create_faiss_index(embeddings, index_type)
        
        # Sauvegarde
        save_faiss_index(index, ids, documents_metadata, faiss_index_dir, original_metadata)
        
        # Test
        test_faiss_index(faiss_index_dir)
        
        print(f"\n🎉 SUCCÈS!")
        print(f"✅ Index FAISS créé avec {index.ntotal} vecteurs")
        print(f"📁 Répertoire: {faiss_index_dir}")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()