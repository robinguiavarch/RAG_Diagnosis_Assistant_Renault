import os
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List
import yaml

import sys
from pathlib import Path

# Ajouter la racine du projet au Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_settings() -> Dict[str, Any]:
    """Charge la configuration depuis settings.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_all_documents(json_dir: Path) -> List[Dict[str, Any]]:
    """Charge tous les documents JSON depuis le répertoire intelligent/"""
    documents = []
    json_files = list(json_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            full_text = " ".join(page["text"] for page in data["pages"])
            
            doc_info = {
                "document_id": data["document_id"],
                "num_pages": data["num_pages"],
                "full_text": full_text,
                "word_count": len(full_text.split()),
                "char_count": len(full_text),
                "source_file": json_file.name,
                "pages": data["pages"]
            }
            documents.append(doc_info)
            
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de {json_file}: {e}")
    
    return documents

def extract_scr_triplets_sequential(text: str) -> List[Dict[str, Any]]:
    """Extrait les triplets SCR avec la méthode séquentielle optimisée"""
    triplets = []
    error_codes = re.findall(r'[A-Z]+-\d+', text)
    
    for i, code in enumerate(error_codes):
        # Trouver la position de ce code
        code_pos = text.find(code)
        if code_pos == -1:
            continue
            
        # Définir la fenêtre de recherche (jusqu'au prochain code ou fin de texte)
        if i < len(error_codes) - 1:
            next_code = error_codes[i + 1]
            next_pos = text.find(next_code, code_pos + len(code))
            if next_pos != -1:
                window = text[code_pos:next_pos]
            else:
                window = text[code_pos:code_pos + 1500]  # Fenêtre plus large
        else:
            window = text[code_pos:code_pos + 1500]
        
        # Chercher cause et remedy dans cette fenêtre
        cause_match = re.search(r'Cause:\s*(.*?)(?=Remedy:|$)', window, re.DOTALL | re.IGNORECASE)
        remedy_match = re.search(r'Remedy:\s*(.*?)(?=$|\n\n|\d+\.\d+)', window, re.DOTALL | re.IGNORECASE)
        
        if cause_match and remedy_match:
            # Extraire le titre/symptôme (ligne avec le code)
            symptom_match = re.search(rf'({re.escape(code)}[^\n]*)', window)
            symptom = symptom_match.group(1) if symptom_match else code
            
            triplets.append({
                'error_code': code,
                'symptom': symptom.strip(),
                'cause': cause_match.group(1).strip(),
                'remedy': remedy_match.group(1).strip()
            })
    
    return triplets

def extract_random_sample(text: str, num_tokens: int = 1000) -> Dict[str, Any]:
    """Extrait un échantillon aléatoire de tokens depuis le texte"""
    words = text.split()
    total_words = len(words)
    
    if total_words <= num_tokens:
        return {
            "sample": text,
            "start_position": 0,
            "end_position": total_words,
            "coverage_percent": 100.0
        }
    
    max_start = total_words - num_tokens
    start_pos = random.randint(0, max_start)
    end_pos = start_pos + num_tokens
    
    sample_words = words[start_pos:end_pos]
    sample_text = " ".join(sample_words)
    
    return {
        "sample": sample_text,
        "start_position": start_pos,
        "end_position": end_pos,
        "coverage_percent": (len(sample_words) / total_words) * 100
    }

def display_document_stats(documents: List[Dict[str, Any]]):
    """Affiche les statistiques essentielles des documents"""
    if not documents:
        print("❌ Aucun document trouvé!")
        return
    
    total_pages = sum(doc['num_pages'] for doc in documents)
    total_words = sum(doc['word_count'] for doc in documents)
    
    print(f"📊 STATISTIQUES GÉNÉRALES")
    print(f"{'='*60}")
    print(f"📄 Documents: {len(documents)}")
    print(f"📖 Pages totales: {total_pages:,}")
    print(f"📝 Mots totaux: {total_words:,}")
    print(f"📏 Mots par page (moyenne): {total_words // total_pages:,}")
    print()

def display_scr_results(doc: Dict[str, Any], triplets: List[Dict[str, Any]]):
    """Affiche les résultats de l'extraction SCR"""
    total_error_codes = len(re.findall(r'[A-Z]+-\d+', doc['full_text']))
    detection_rate = (len(triplets) / total_error_codes * 100) if total_error_codes > 0 else 0
    
    print(f"🎯 EXTRACTION SCR (Méthode séquentielle)")
    print(f"{'='*60}")
    print(f"📄 Document analysé: {doc['document_id']}")
    print(f"🚨 Codes d'erreur totaux: {total_error_codes:,}")
    print(f"✅ Triplets SCR détectés: {len(triplets):,}")
    print(f"📊 Taux de détection: {detection_rate:.1f}%")
    print()
    
    # Afficher UN exemple de triplet
    if triplets:
        example = triplets[0]
        print(f"📋 EXEMPLE DE TRIPLET SCR:")
        print(f"{'─'*60}")
        print(f"🔸 Code d'erreur: {example['error_code']}")
        print(f"🔸 Symptôme: {example['symptom']}")
        print(f"🔸 Cause: {example['cause'][:150]}{'...' if len(example['cause']) > 150 else ''}")
        print(f"🔸 Remède: {example['remedy'][:150]}{'...' if len(example['remedy']) > 150 else ''}")
        print()

def display_text_sample(doc: Dict[str, Any], sample_info: Dict[str, Any]):
    """Affiche un échantillon du texte"""
    print(f"📝 APERÇU DU DOCUMENT JSON")
    print(f"{'='*60}")
    print(f"📄 Document: {doc['document_id']}")
    print(f"📍 Position: mots {sample_info['start_position']:,} à {sample_info['end_position']:,}")
    print(f"📏 Couverture: {sample_info['coverage_percent']:.1f}% du document")
    print(f"{'─'*60}")
    
    # Afficher le texte avec une mise en forme simple
    sample_text = sample_info['sample']
    lines = sample_text.replace('. ', '.\n').split('\n')
    
    # Prendre seulement les 15 premières lignes pour éviter le verbeux
    for line in lines[:15]:
        line = line.strip()
        if line and len(line) > 10:  # Ignorer les lignes trop courtes
            if len(line) > 100:
                print(f"   {line[:100]}...")
            else:
                print(f"   {line}")
    
    if len(lines) > 15:
        print(f"   ... ({len(lines) - 15} lignes supplémentaires)")
    print()

def main():
    try:
        print("🔍 Analyse des documents JSON nettoyés...")
        
        # Charger la configuration
        settings = load_settings()
        json_dir = Path(settings["paths"]["json_documents"])
        
        if not json_dir.exists():
            print(f"❌ Le répertoire {json_dir} n'existe pas!")
            print(f"💡 Exécutez d'abord: python 01_extract_text_PyMuPDF_intelligent.py")
            return
        
        # Charger les documents
        documents = load_all_documents(json_dir)
        if not documents:
            print("❌ Aucun document JSON trouvé!")
            return
        
        # Statistiques générales
        display_document_stats(documents)
        
        # Analyser le document le plus volumineux
        target_doc = max(documents, key=lambda x: x['word_count'])
        print(f"🎯 Analyse SCR sur le document principal: {target_doc['document_id']}")
        print()
        
        # Extraction SCR avec méthode séquentielle
        triplets = extract_scr_triplets_sequential(target_doc['full_text'])
        display_scr_results(target_doc, triplets)
        
        # Aperçu du document
        sample_info = extract_random_sample(target_doc['full_text'], num_tokens=800)
        display_text_sample(target_doc, sample_info)
        
        # Résumé final
        print(f"💡 RÉSUMÉ")
        print(f"{'='*40}")
        print(f"✅ Méthode séquentielle validée")
        print(f"📊 ~{len(triplets):,} triplets extractibles")
        print(f"🚀 Prêt pour réécrire 00_extract_scr_triplets.py")
        print()
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()