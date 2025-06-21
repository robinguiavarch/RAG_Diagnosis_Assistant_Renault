import os
import shutil
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.index import create_in
from whoosh.analysis import StandardAnalyzer

# === FONCTION POUR CHARGER LE YAML ===
def load_settings(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis le fichier YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# === CHARGEMENT DES CHUNKS ===
def load_documents(chunk_dir: Path) -> List[Dict[str, Any]]:
    """Charge tous les chunks depuis les fichiers JSON avec validation"""
    print(f"🔍 Chargement des chunks depuis: {chunk_dir}")
    
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Répertoire des chunks introuvable: {chunk_dir}")
    
    documents = []
    chunk_files = list(chunk_dir.glob("*_chunks.json"))
    
    if not chunk_files:
        raise FileNotFoundError(f"Aucun fichier chunks trouvé dans: {chunk_dir}")
    
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
            
            # Vérification si c'est un nouveau format avec métadonnées
            has_metadata = "source_stats" in data
            chunking_method = data.get("chunking_config", {}).get("method", "unknown")
            
            if has_metadata:
                print(f"   ✅ Nouveau format détecté (méthode: {chunking_method})")
            else:
                print(f"   ⚠️ Ancien format détecté")
            
            # Chargement des chunks
            doc_chunks = 0
            for chunk in data["chunks"]:
                # Validation des champs requis
                required_fields = ["chunk_id", "text"]
                if not all(field in chunk for field in required_fields):
                    print(f"   ⚠️ Chunk invalide dans {json_file.name}")
                    continue
                
                # Vérification que le texte n'est pas vide
                text = chunk["text"].strip()
                if not text:
                    print(f"   ⚠️ Chunk vide ignoré (ID: {chunk['chunk_id']})")
                    continue
                
                # Détection de mots collés (diagnostic de qualité)
                words = text.split()
                long_words = [w for w in words if len(w) > 15]
                
                documents.append({
                    "document_id": data.get("document_id", json_file.stem.replace("_chunks", "")),
                    "chunk_id": chunk["chunk_id"],
                    "text": text,
                    "word_count": chunk.get("word_count", len(words)),
                    "char_count": chunk.get("char_count", len(text)),
                    "source_file": json_file.name,
                    "has_metadata": has_metadata,
                    "chunking_method": chunking_method,
                    "quality_score": max(0, 1.0 - (len(long_words) / len(words))) if words else 0.0
                })
                doc_chunks += 1
            
            print(f"   ✅ {doc_chunks} chunks chargés")
            
        except Exception as e:
            print(f"   ❌ Erreur lecture {json_file.name}: {e}")
            continue
    
    # Analyse de qualité globale
    if documents:
        quality_scores = [doc["quality_score"] for doc in documents]
        avg_quality = sum(quality_scores) / len(quality_scores)
        low_quality_count = sum(1 for score in quality_scores if score < 0.8)
        
        print(f"\n📊 ANALYSE DE QUALITÉ:")
        print(f"   📄 Total chunks: {len(documents)}")
        print(f"   📈 Qualité moyenne: {avg_quality:.2%}")
        if low_quality_count > 0:
            print(f"   ⚠️ Chunks qualité faible: {low_quality_count}")
        else:
            print(f"   ✅ Tous les chunks semblent de bonne qualité")
    
    print(f"📊 Total: {len(documents)} chunks chargés")
    return documents

# === INDEXATION BM25 AVEC WHOOSH ===
def create_bm25_index(index_dir: Path, documents: List[Dict[str, Any]]):
    """Crée l'index BM25 avec Whoosh"""
    print(f"\n🛠️ Création de l'index BM25 dans: {index_dir}")
    
    # Nettoyage de l'index existant
    if index_dir.exists():
        print("🗑️ Suppression de l'ancien index...")
        shutil.rmtree(index_dir)
    
    index_dir.mkdir(parents=True, exist_ok=True)

    # Schéma enrichi avec plus de métadonnées
    schema = Schema(
        document_id=ID(stored=True),
        chunk_id=ID(stored=True),
        content=TEXT(stored=True, analyzer=StandardAnalyzer()),
        word_count=NUMERIC(stored=True),
        char_count=NUMERIC(stored=True),
        quality_score=NUMERIC(stored=True),
        source_file=ID(stored=True),
        chunking_method=ID(stored=True)
    )

    print("📝 Création du schéma d'index...")
    index = create_in(index_dir, schema)
    writer = index.writer()

    print("🔄 Indexation des documents...")
    indexed_count = 0
    errors_count = 0
    
    for i, doc in enumerate(documents):
        try:
            writer.add_document(
                document_id=str(doc["document_id"]),
                chunk_id=str(doc["chunk_id"]),
                content=doc["text"],
                word_count=doc["word_count"],
                char_count=doc["char_count"],
                quality_score=doc["quality_score"],
                source_file=doc["source_file"],
                chunking_method=doc["chunking_method"]
            )
            indexed_count += 1
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"   📄 {i + 1}/{len(documents)} documents indexés...")
                
        except Exception as e:
            print(f"   ❌ Erreur indexation document {doc['document_id']}-{doc['chunk_id']}: {e}")
            errors_count += 1
            continue

    print("💾 Validation de l'index...")
    writer.commit()
    
    # Statistiques finales
    print(f"\n✅ INDEX BM25 CRÉÉ AVEC SUCCÈS!")
    print(f"📊 Statistiques:")
    print(f"   📄 Documents indexés: {indexed_count}")
    print(f"   ❌ Erreurs: {errors_count}")
    print(f"   📁 Répertoire: {index_dir}")
    
    # Taille de l'index
    index_size = sum(f.stat().st_size for f in index_dir.rglob('*') if f.is_file())
    index_size_mb = index_size / (1024 * 1024)
    print(f"   💾 Taille index: {index_size_mb:.1f} MB")

def test_index(index_dir: Path, sample_query: str = "error robot"):
    """Test rapide de l'index créé"""
    print(f"\n🧪 TEST DE L'INDEX avec requête: '{sample_query}'")
    
    try:
        from whoosh.index import open_dir
        from whoosh.qparser import QueryParser
        
        index = open_dir(index_dir)
        searcher = index.searcher()
        parser = QueryParser("content", index.schema)
        query = parser.parse(sample_query)
        
        results = searcher.search(query, limit=3)
        
        print(f"🔍 {len(results)} résultat(s) trouvé(s):")
        for i, result in enumerate(results):
            print(f"   {i+1}. Doc: {result['document_id']} | Chunk: {result['chunk_id']}")
            print(f"      Score: {result.score:.3f}")
            print(f"      Extrait: {result['content'][:100]}...")
            print()
        
        searcher.close()
        print("✅ Index fonctionne correctement!")
        
    except Exception as e:
        print(f"❌ Erreur test index: {e}")

def save_index_metadata(index_dir: Path, documents: List[Dict[str, Any]]):
    """Sauvegarde les métadonnées de l'index"""
    metadata = {
        "created_at": datetime.now().isoformat(),
        "num_documents": len(documents),
        "index_type": "BM25_Whoosh",
        "schema_fields": ["document_id", "chunk_id", "content", "word_count", "char_count", "quality_score"],
        "documents_info": {}
    }
    
    # Statistiques par document source
    doc_stats = {}
    for doc in documents:
        doc_id = doc["document_id"]
        if doc_id not in doc_stats:
            doc_stats[doc_id] = {
                "chunks": 0, 
                "total_words": 0,
                "avg_quality": 0.0,
                "chunking_method": doc["chunking_method"]
            }
        doc_stats[doc_id]["chunks"] += 1
        doc_stats[doc_id]["total_words"] += doc["word_count"]
    
    # Calcul de la qualité moyenne par document
    for doc_id in doc_stats:
        doc_chunks = [d for d in documents if d["document_id"] == doc_id]
        avg_quality = sum(d["quality_score"] for d in doc_chunks) / len(doc_chunks)
        doc_stats[doc_id]["avg_quality"] = avg_quality
    
    metadata["documents_info"] = doc_stats
    
    # Sauvegarde
    metadata_file = index_dir / "index_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"📝 Métadonnées sauvées: {metadata_file}")

# === MAIN ===
def main():
    print("🚀 CRÉATION DE L'INDEX BM25")
    print("=" * 50)
    
    try:
        # Chargement de la configuration
        settings = load_settings("../../config/settings.yaml")
        chunk_dir = Path(settings["paths"]["chunk_documents"])
        index_dir = Path(settings["paths"]["bm25_index"])
        
        print(f"🔧 Configuration:")
        print(f"   📁 Chunks: {chunk_dir}")
        print(f"   🗂️ Index: {index_dir}")
        
        # Chargement des documents
        documents = load_documents(chunk_dir)
        
        if not documents:
            print("❌ Aucun document à indexer!")
            return
        
        # Création de l'index
        create_bm25_index(index_dir, documents)
        
        # Sauvegarde des métadonnées
        save_index_metadata(index_dir, documents)
        
        # Test de l'index
        test_index(index_dir)
        
        print(f"\n🎉 SUCCÈS!")
        print(f"✅ Index BM25 créé avec {len(documents)} documents")
        print(f"📁 Répertoire: {index_dir}")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()