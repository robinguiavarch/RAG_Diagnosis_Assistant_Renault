import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import nltk

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# S'assurer que les tokenizers de phrases sont dispo
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

# === FONCTION POUR CHARGER LE YAML ===
def load_settings(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis le fichier YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# === CHUNKING PAR MOTS ===
def chunk_by_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Découpe le texte en chunks de taille fixe avec overlap"""
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

# === CHUNKING PAR GROUPES DE PHRASES ===
def chunk_by_sentence_grouping(text: str, max_words: int, overlap_words: int) -> List[str]:
    """Découpe le texte en groupes de phrases respectant la taille max et l'overlap"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        word_count = len(sent.split())
        
        # Si on peut ajouter cette phrase sans dépasser la limite
        if current_len + word_count <= max_words:
            current_chunk.append(sent)
            current_len += word_count
        else:
            # Finaliser le chunk actuel s'il n'est pas vide
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
                # Gestion de l'overlap : garder les dernières phrases
                overlap_sentences = []
                total_overlap_words = 0
                
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if total_overlap_words + sent_words <= overlap_words:
                        overlap_sentences.insert(0, sent)
                        total_overlap_words += sent_words
                    else:
                        break
                
                # Commencer le nouveau chunk avec l'overlap + la phrase actuelle
                current_chunk = overlap_sentences + [sent]
                current_len = sum(len(s.split()) for s in current_chunk)
            else:
                # Phrase trop longue seule - la mettre dans un chunk à elle seule
                chunks.append(sent)
                current_chunk = []
                current_len = 0

    # Ajouter le dernier chunk s'il reste du contenu
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def validate_json_structure(data: Dict[str, Any], file_path: Path) -> bool:
    """Valide la structure du fichier JSON"""
    required_fields = ["document_id", "num_pages", "pages"]
    
    for field in required_fields:
        if field not in data:
            print(f"   ⚠️ Champ manquant '{field}' dans {file_path.name}")
            return False
    
    if not isinstance(data["pages"], list):
        print(f"   ⚠️ Le champ 'pages' n'est pas une liste dans {file_path.name}")
        return False
    
    for i, page in enumerate(data["pages"]):
        if "text" not in page:
            print(f"   ⚠️ Champ 'text' manquant page {i+1} dans {file_path.name}")
            return False
    
    return True

def analyze_chunk_quality(chunks: List[str], method: str) -> Dict[str, Any]:
    """Analyse la qualité des chunks produits"""
    if not chunks:
        return {"error": "Aucun chunk produit"}
    
    word_counts = [len(chunk.split()) for chunk in chunks]
    char_counts = [len(chunk) for chunk in chunks]
    
    return {
        "num_chunks": len(chunks),
        "avg_words": sum(word_counts) / len(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "avg_chars": sum(char_counts) / len(char_counts),
        "method": method,
        "chunks_under_50_words": len([c for c in word_counts if c < 50]),
        "chunks_over_500_words": len([c for c in word_counts if c > 500])
    }

# === MAIN ===
def main():
    print("🔄 Début du chunking des documents (texte déjà nettoyé)...")
    
    try:
        settings = load_settings("../../config/settings.yaml")
        json_dir = Path(settings["paths"]["json_documents"])
        chunk_dir = Path(settings["paths"]["chunk_documents"])
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_size = settings["chunking"]["chunk_size"]
        overlap = settings["chunking"]["overlap"]
        tokenize_by = settings["chunking"].get("tokenize_by", "word")

        # DEBUG: Afficher des informations détaillées
        print(f"🔍 DIAGNOSTIC PATHS:")
        print(f"   📁 Répertoire courant: {Path.cwd()}")
        print(f"   📁 JSON dir configuré: {json_dir}")
        print(f"   📁 JSON dir absolu: {json_dir.resolve()}")
        print(f"   📁 JSON dir existe: {json_dir.exists()}")
        
        if json_dir.exists():
            all_files = list(json_dir.glob("*"))
            json_files = list(json_dir.glob("*.json"))
            print(f"   📄 Tous fichiers: {len(all_files)} → {[f.name for f in all_files[:5]]}")
            print(f"   📄 Fichiers JSON: {len(json_files)} → {[f.name for f in json_files[:5]]}")
        else:
            print(f"   ❌ Le répertoire {json_dir} n'existe pas!")
            # Essayer de trouver où sont les fichiers
            possible_paths = [
                Path("data/json_documents/intelligent"),
                Path("./data/json_documents/intelligent"),
                Path("../data/json_documents/intelligent"),
            ]
            for path in possible_paths:
                if path.exists():
                    files = list(path.glob("*.json"))
                    print(f"   💡 Trouvé à la place: {path.resolve()} ({len(files)} JSON)")

        print(f"📊 Configuration: {chunk_size} mots, overlap {overlap}, méthode: {tokenize_by}")

        json_files = list(json_dir.glob("*.json"))
        
        if not json_files:
            print(f"❌ Aucun fichier JSON trouvé dans {json_dir}")
            print(f"❌ Chemin absolu vérifié: {json_dir.resolve()}")
            return

        print(f"📄 {len(json_files)} document(s) à traiter...")

        total_chunks = 0
        total_docs_processed = 0

        for json_file in json_files:
            print(f"📝 Traitement de {json_file.name}...")
            
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Validation de la structure
                if not validate_json_structure(data, json_file):
                    print(f"   ❌ Structure invalide, document ignoré")
                    continue

                # Concaténation du texte de toutes les pages (déjà nettoyé)
                raw_text = " ".join(page["text"] for page in data["pages"])
                
                # Statistiques du document source
                total_words = len(raw_text.split())
                total_chars = len(raw_text)
                print(f"   📄 Document: {total_words:,} mots, {total_chars:,} caractères")

                # Chunking selon la méthode choisie (SANS nettoyage supplémentaire)
                if tokenize_by == "sentence":
                    chunks = chunk_by_sentence_grouping(raw_text, max_words=chunk_size, overlap_words=overlap)
                else:
                    chunks = chunk_by_words(raw_text, chunk_size, overlap)

                # Analyse de la qualité des chunks
                quality = analyze_chunk_quality(chunks, tokenize_by)

                chunked_output = {
                    "document_id": data["document_id"],
                    "source_stats": {
                        "total_words": total_words,
                        "total_chars": total_chars,
                        "num_pages": data["num_pages"]
                    },
                    "chunking_config": {
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "method": tokenize_by
                    },
                    "chunk_quality": quality,
                    "num_chunks": len(chunks),
                    "chunks": [
                        {
                            "chunk_id": i, 
                            "text": chunk,
                            "word_count": len(chunk.split()),
                            "char_count": len(chunk)
                        }
                        for i, chunk in enumerate(chunks)
                    ]
                }

                output_path = chunk_dir / f"{data['document_id']}_chunks.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(chunked_output, f, ensure_ascii=False, indent=2)
                
                print(f"   ✅ {len(chunks)} chunks créés (moy: {quality['avg_words']:.1f} mots)")
                print(f"   💾 Sauvé: {output_path.name}")
                
                total_chunks += len(chunks)
                total_docs_processed += 1
                
            except Exception as e:
                print(f"   ❌ Erreur traitement {json_file.name}: {e}")
                continue

        print(f"\n🎉 Chunking terminé !")
        print(f"📊 Résumé: {total_docs_processed} documents → {total_chunks} chunks")
        print(f"📁 Fichiers sauvés dans: {chunk_dir}")
        
        if total_chunks > 0:
            avg_chunks_per_doc = total_chunks / total_docs_processed
            print(f"📈 Moyenne: {avg_chunks_per_doc:.1f} chunks par document")
        
    except Exception as e:
        print(f"❌ Erreur globale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()