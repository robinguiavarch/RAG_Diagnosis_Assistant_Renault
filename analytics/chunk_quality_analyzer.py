import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List
import sys
import yaml
sys.path.append(str(Path(__file__).parent.parent))

def load_settings() -> Dict[str, Any]:
    """Charge la configuration depuis settings.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_all_chunks(chunk_dir: Path) -> List[Dict[str, Any]]:
    """Charge tous les chunks de tous les documents"""
    all_chunks = []
    chunk_files = list(chunk_dir.glob("*_chunks.json"))
    
    print(f"🔍 Fichiers chunks trouvés: {len(chunk_files)}")
    for chunk_file in chunk_files:
        print(f"   📄 {chunk_file.name}")
    
    for chunk_file in chunk_files:
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # DIAGNOSTIC: Vérifier d'où viennent les chunks
        print(f"\n📊 Analyse de {chunk_file.name}:")
        if "source_stats" in data:
            print(f"   ✅ Fichier moderne avec métadonnées")
            print(f"   📄 Document source: {data.get('document_id', 'N/A')}")
            print(f"   🔧 Méthode chunking: {data.get('chunking_config', {}).get('method', 'N/A')}")
        else:
            print(f"   ⚠️ Fichier ancien sans métadonnées")
        
        # Analyser quelques chunks pour détecter les mots collés
        sample_chunks = data["chunks"][:2] if "chunks" in data else []
        for i, chunk in enumerate(sample_chunks):
            text = chunk.get("text", "")
            words = text.split()
            long_words = [w for w in words if len(w) > 15]
            if long_words:
                print(f"   ⚠️ Chunk {i}: mots suspects ({len(long_words)}): {long_words[:3]}")
            else:
                print(f"   ✅ Chunk {i}: texte semble propre")
        
        for chunk in data["chunks"]:
            chunk_info = {
                "document_id": data["document_id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "word_count": chunk.get("word_count", len(chunk["text"].split())),
                "char_count": chunk.get("char_count", len(chunk["text"])),
                "source_file": chunk_file.name,
                "has_metadata": "source_stats" in data,
                "extraction_method": data.get("chunking_config", {}).get("method", "unknown")
            }
            all_chunks.append(chunk_info)
    
    return all_chunks

def display_chunk(chunk: Dict[str, Any], index: int):
    """Affiche un chunk de manière formatée avec diagnostic"""
    print(f"{'='*80}")
    print(f"CHUNK #{index + 1}")
    print(f"{'='*80}")
    print(f"📄 Document: {chunk['document_id']}")
    print(f"🔢 Chunk ID: {chunk['chunk_id']}")
    print(f"📊 Stats: {chunk['word_count']} mots | {chunk['char_count']} caractères")
    print(f"📁 Fichier: {chunk['source_file']}")
    print(f"🔧 Métadonnées: {'✅ Oui' if chunk['has_metadata'] else '❌ Non'}")
    print(f"🔧 Méthode: {chunk['extraction_method']}")
    
    # Diagnostic qualité du texte
    text = chunk['text']
    words = text.split()
    long_words = [w for w in words if len(w) > 15]
    
    if long_words:
        print(f"⚠️ PROBLÈME: {len(long_words)} mots suspects détectés!")
        print(f"   Exemples: {long_words[:5]}")
    else:
        print(f"✅ QUALITÉ: Texte semble propre")
    
    print(f"{'─'*80}")
    print("📝 CONTENU:")
    print(f"{'─'*80}")
    
    # Afficher le texte avec retour à la ligne pour lisibilité
    text_lines = chunk['text'].split('. ')
    for line in text_lines:
        if line.strip():
            print(f"   {line.strip()}{'.' if not line.endswith('.') else ''}")
    
    print(f"{'─'*80}")
    print()

def display_chunks_statistics(all_chunks: List[Dict[str, Any]]):
    """Affiche des statistiques générales sur les chunks avec diagnostic"""
    if not all_chunks:
        print("❌ Aucun chunk trouvé!")
        return
    
    word_counts = [chunk['word_count'] for chunk in all_chunks]
    char_counts = [chunk['char_count'] for chunk in all_chunks]
    
    # Documents uniques
    unique_docs = set(chunk['document_id'] for chunk in all_chunks)
    
    # Diagnostic de qualité
    chunks_with_metadata = [c for c in all_chunks if c['has_metadata']]
    chunks_with_long_words = []
    
    for chunk in all_chunks:
        words = chunk['text'].split()
        long_words = [w for w in words if len(w) > 15]
        if long_words:
            chunks_with_long_words.append(chunk)
    
    print(f"📊 STATISTIQUES GÉNÉRALES")
    print(f"{'='*50}")
    print(f"📄 Nombre de documents: {len(unique_docs)}")
    print(f"🔢 Nombre total de chunks: {len(all_chunks)}")
    print(f"📏 Taille moyenne (mots): {sum(word_counts) / len(word_counts):.1f}")
    print(f"📏 Taille min/max (mots): {min(word_counts)} / {max(word_counts)}")
    print()
    
    # Diagnostic de qualité
    print(f"🔍 DIAGNOSTIC DE QUALITÉ")
    print(f"{'='*50}")
    print(f"✅ Chunks avec métadonnées: {len(chunks_with_metadata)}/{len(all_chunks)}")
    print(f"⚠️ Chunks avec mots suspects: {len(chunks_with_long_words)}/{len(all_chunks)}")
    
    if chunks_with_long_words:
        print(f"❌ PROBLÈME DÉTECTÉ: Des chunks contiennent des mots collés!")
        print(f"   Cela indique que l'extraction intelligente n'a pas été utilisée.")
        print(f"   Vérifiez le chemin json_documents dans settings.yaml")
    else:
        print(f"✅ QUALITÉ OK: Tous les chunks semblent propres")
    
    print()
    
    # Distribution par document
    doc_chunk_counts = {}
    for chunk in all_chunks:
        doc_id = chunk['document_id']
        doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
    
    print("📊 RÉPARTITION PAR DOCUMENT:")
    print(f"{'─'*50}")
    for doc_id, count in sorted(doc_chunk_counts.items()):
        print(f"   {doc_id}: {count} chunks")
    print()

def search_chunks_by_keyword(all_chunks: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
    """Recherche des chunks contenant un mot-clé (utile pour débugger)"""
    matching_chunks = []
    keyword_lower = keyword.lower()
    
    for chunk in all_chunks:
        if keyword_lower in chunk['text'].lower():
            matching_chunks.append(chunk)
    
    return matching_chunks

def main():
    try:
        # Charger la configuration
        settings = load_settings()
        chunk_dir = Path(settings["paths"]["chunk_documents"])
        json_dir = Path(settings["paths"]["json_documents"])
        
        # DIAGNOSTIC: Afficher les chemins utilisés
        print(f"🔍 DIAGNOSTIC DES CHEMINS:")
        print(f"{'='*50}")
        print(f"📁 JSON source: {json_dir}")
        print(f"📁 Chunks: {chunk_dir}")
        print(f"📁 JSON existe: {json_dir.exists()}")
        print(f"📁 Chunks existe: {chunk_dir.exists()}")
        
        if json_dir.exists():
            json_files = list(json_dir.glob("*.json"))
            print(f"📄 Fichiers JSON trouvés: {len(json_files)}")
            if json_files:
                print(f"📄 Premier fichier: {json_files[0].name}")
                
                # Vérifier la qualité du premier fichier JSON
                with open(json_files[0], "r", encoding="utf-8") as f:
                    sample_data = json.load(f)
                
                if "pages" in sample_data and sample_data["pages"]:
                    sample_text = sample_data["pages"][0]["text"][:200]
                    sample_words = sample_text.split()
                    long_words = [w for w in sample_words if len(w) > 15]
                    
                    print(f"🔍 Qualité du JSON source:")
                    if long_words:
                        print(f"   ⚠️ JSON contient des mots collés: {long_words[:3]}")
                        print(f"   💡 Utilisez l'extraction intelligente: scripts/01_extract_text_PyMuPDF_intelligent.py")
                    else:
                        print(f"   ✅ JSON semble propre")
        
        print()
        
        if not chunk_dir.exists():
            print(f"❌ Le répertoire {chunk_dir} n'existe pas!")
            return
        
        print("🔍 Chargement des chunks...")
        all_chunks = load_all_chunks(chunk_dir)
        
        if not all_chunks:
            print("❌ Aucun chunk trouvé dans le répertoire!")
            return
        
        # Afficher les statistiques avec diagnostic
        display_chunks_statistics(all_chunks)
        
        # Sélectionner 3 chunks aléatoires (réduit pour le diagnostic)
        num_samples = min(3, len(all_chunks))
        random_chunks = random.sample(all_chunks, num_samples)
        
        print(f"🎲 AFFICHAGE DE {num_samples} CHUNKS ALÉATOIRES")
        print("="*80)
        print()
        
        for i, chunk in enumerate(random_chunks):
            display_chunk(chunk, i)
        
        # Recommandations basées sur le diagnostic
        print("\n" + "="*80)
        print("💡 RECOMMANDATIONS")
        print("="*80)
        
        chunks_with_issues = [c for c in all_chunks if len([w for w in c['text'].split() if len(w) > 15]) > 0]
        
        if chunks_with_issues:
            print("⚠️ PROBLÈME DÉTECTÉ:")
            print("   Vos chunks contiennent des mots collés (texte mal extrait)")
            print()
            print("🔧 SOLUTION:")
            print("   1. Utilisez l'extraction intelligente:")
            print("      poetry run python scripts/01_extract_text_PyMuPDF.py")
            print()
            print("   2. Modifiez settings.yaml pour pointer vers les JSONs:")
            print("      json_documents: data/json_documents/")
            print()
            print("   3. Re-chunkez avec le texte propre:")
            print("      poetry run python scripts/02_chunk_documents.py")
            print()
            print("   4. Re-visualisez:")
            print("      poetry run python visualization/visualize_chunks.py")
        else:
            print("✅ QUALITÉ OK:")
            print("   Vos chunks semblent propres et prêts pour le RAG!")
        
        print("\n👋 Diagnostic terminé!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()