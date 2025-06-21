import os
import json
import yaml
import re
from pathlib import Path
from typing import Dict, Any, List, Set
import nltk

try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF n'est pas installé. Installez-le avec:")
    print("poetry add pymupdf")
    exit(1)

# Télécharger les ressources NLTK nécessaires
nltk.download('words', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import words

def load_settings(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

class IntelligentWordSplitter:
    """Classe pour séparer intelligemment les mots collés en utilisant un dictionnaire"""
    
    def __init__(self):
        # Charger le dictionnaire anglais de NLTK
        print("🔄 Chargement du dictionnaire anglais...")
        self.english_words = set(word.lower() for word in words.words())
        
        # Ajouter des mots techniques courants qui peuvent manquer dans NLTK
        technical_words = {
            'error', 'alarm', 'robot', 'controller', 'pendant', 'teach', 'emergency',
            'stop', 'reset', 'program', 'line', 'process', 'logger', 'internal',
            'system', 'software', 'hardware', 'config', 'configuration', 'setup',
            'operation', 'manual', 'technical', 'representative', 'contact',
            'information', 'parameters', 'equipment', 'number', 'indicated',
            'dispensing', 'outputs', 'operator', 'panel', 'status', 'signal',
            'unique', 'identifier', 'position', 'statement', 'standalone',
            'motion', 'equal', 'matching', 'found', 'events', 'leading',
            'document', 'refer', 'further', 'please', 'also', 'cycle', 'power',
            'enable', 'disable', 'assigned', 'executed', 'version', 'mismatch'
        }
        self.english_words.update(technical_words)
        
        # Mots très courts à ignorer (éviter la sur-segmentation)
        self.min_word_length = 3
        
        print(f"✅ Dictionnaire chargé: {len(self.english_words):,} mots")
    
    def is_valid_word(self, word: str) -> bool:
        """Vérifie si un mot est valide (dans le dictionnaire)"""
        return word.lower() in self.english_words and len(word) >= self.min_word_length
    
    def find_word_splits(self, merged_word: str) -> List[str]:
        """
        Trouve la meilleure façon de séparer un mot collé en mots valides
        Utilise la programmation dynamique pour trouver la segmentation optimale
        """
        word = merged_word.lower()
        n = len(word)
        
        # DP array: dp[i] = True si word[0:i] peut être segmenté en mots valides
        dp = [False] * (n + 1)
        dp[0] = True  # chaîne vide
        
        # Pour reconstruire la solution
        parent = [-1] * (n + 1)
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and self.is_valid_word(word[j:i]):
                    dp[i] = True
                    parent[i] = j
                    break
        
        # Reconstruire la segmentation si possible
        if dp[n]:
            result = []
            pos = n
            while pos > 0:
                start = parent[pos]
                # Préserver la casse originale
                original_segment = merged_word[start:pos]
                result.append(original_segment)
                pos = start
            return list(reversed(result))
        
        # Si aucune segmentation valide trouvée, essayer des heuristiques
        return self.heuristic_split(merged_word)
    
    def heuristic_split(self, merged_word: str) -> List[str]:
        """Heuristiques de segmentation quand la DP échoue"""
        # Essayer de diviser sur les changements de casse
        parts = re.findall(r'[A-Z][a-z]*|[a-z]+', merged_word)
        
        # Vérifier si les parties sont des mots valides
        valid_parts = []
        for part in parts:
            if self.is_valid_word(part) or len(part) <= 3:
                valid_parts.append(part)
            else:
                # Essayer de diviser plus finement
                subparts = self.try_common_prefixes_suffixes(part)
                valid_parts.extend(subparts)
        
        return valid_parts if len(valid_parts) > 1 else [merged_word]
    
    def try_common_prefixes_suffixes(self, word: str) -> List[str]:
        """Essaie de séparer avec des préfixes/suffixes courants"""
        common_prefixes = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under', 'out']
        common_suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness', 'ment']
        
        word_lower = word.lower()
        
        # Essayer les préfixes
        for prefix in common_prefixes:
            if word_lower.startswith(prefix) and len(word) > len(prefix) + 2:
                rest = word[len(prefix):]
                if self.is_valid_word(rest):
                    return [word[:len(prefix)], rest]
        
        # Essayer les suffixes
        for suffix in common_suffixes:
            if word_lower.endswith(suffix) and len(word) > len(suffix) + 2:
                base = word[:-len(suffix)]
                if self.is_valid_word(base):
                    return [base, word[-len(suffix):]]
        
        return [word]
    
    def split_long_word(self, word: str) -> str:
        """Sépare un mot long s'il semble être composé de plusieurs mots"""
        # Ignorer les mots courts ou ceux qui semblent être des codes
        if len(word) < 8 or re.match(r'^[A-Z]+-\d+$', word):
            return word
        
        # Ignorer les mots qui sont déjà valides
        if self.is_valid_word(word):
            return word
        
        # Essayer de séparer
        splits = self.find_word_splits(word)
        
        # Ne retourner la segmentation que si elle améliore vraiment
        if len(splits) > 1 and all(len(s) >= 2 for s in splits):
            return ' '.join(splits)
        
        return word

def clean_extracted_text_intelligent(text: str, word_splitter: IntelligentWordSplitter) -> str:
    """Nettoyage intelligent du texte avec séparation automatique des mots"""
    if not text:
        return ""
    
    # print("🔄 Nettoyage intelligent du texte...")
    
    # 1. Restaurer les espaces manquants entre minuscule et majuscule
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # 2. Restaurer les espaces après les points suivis d'une majuscule
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    
    # 3. Restaurer les espaces autour de la ponctuation collée
    text = re.sub(r'([a-zA-Z])([.,;:!?])([a-zA-Z])', r'\1\2 \3', text)
    
    # 4. Séparer les mots collés avec des chiffres
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    
    # 5. INTELLIGENCE ARTIFICIELLE: Séparer les mots longs collés
    # print("🧠 Séparation intelligente des mots collés...")
    words = text.split()
    fixed_words = []
    
    for word in words:
        # Nettoyer le mot de la ponctuation pour l'analyse
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word:
            # Essayer de séparer le mot
            split_result = word_splitter.split_long_word(clean_word)
            
            # Remettre la ponctuation si elle était présente
            if split_result != clean_word:
                # Le mot a été séparé
                punctuation = re.findall(r'[^\w]', word)
                if punctuation:
                    # Ajouter la ponctuation au dernier mot
                    split_words = split_result.split()
                    if split_words:
                        split_words[-1] += ''.join(punctuation)
                        fixed_words.extend(split_words)
                    else:
                        fixed_words.append(word)
                else:
                    fixed_words.extend(split_result.split())
            else:
                fixed_words.append(word)
        else:
            fixed_words.append(word)
    
    text = ' '.join(fixed_words)
    
    # 6. Nettoyer les numéros de page/section isolés
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\s+', '', text)
    text = re.sub(r'\b\d{1,2}\s*–\s*\d{3,4}\s+', '', text)
    text = re.sub(r'\b\d{1,2}\.\s*(?=\n|\s|$)', '', text)
    
    # 7. Restaurer la structure avec les codes d'erreur
    text = re.sub(r'([A-Z]{2,}-\d{3})', r'\n\n\1', text)
    
    # 8. Séparer Cause: et Remedy: sur de nouvelles lignes
    text = re.sub(r'(Cause:)', r'\n\1', text)
    text = re.sub(r'(Remedy:)', r'\n\1', text)
    
    # 9. Nettoyer les espaces en trop
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()

def extract_text_with_pymupdf_intelligent(pdf_path: Path, word_splitter: IntelligentWordSplitter) -> Dict[str, Any]:
    """Extraction PDF avec PyMuPDF et séparation intelligente"""
    content = {
        "document_id": pdf_path.stem,
        "num_pages": 0,
        "pages": []
    }
    
    try:
        doc = fitz.open(pdf_path)
        content["num_pages"] = len(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extraction avec différentes méthodes
            text_basic = page.get_text()
            
            try:
                text_layout = page.get_text("text", sort=True)
            except:
                text_layout = text_basic
            
            # Choisir la meilleure extraction
            best_text = text_layout if len(text_layout) > len(text_basic) else text_basic
            
            # Nettoyage intelligent
            cleaned_text = clean_extracted_text_intelligent(best_text, word_splitter)
            
            content["pages"].append({
                "page_number": page_num + 1,
                "text": cleaned_text,
                "raw_text": best_text,
                "extraction_method": "pymupdf_intelligent"
            })
        
        doc.close()
        
    except Exception as e:
        print(f"   ❌ Erreur PyMuPDF: {e}")
        content = {
            "document_id": pdf_path.stem,
            "num_pages": 0,
            "pages": [{"page_number": 1, "text": f"Erreur d'extraction: {e}", "extraction_method": "error"}]
        }
    
    return content

def analyze_improvements(old_text: str, new_text: str) -> Dict[str, Any]:
    """Analyse les améliorations apportées"""
    
    # Compter les mots suspects (très longs)
    old_long_words = [w for w in old_text.split() if len(w) > 15]
    new_long_words = [w for w in new_text.split() if len(w) > 15]
    
    # Compter les mots totaux
    old_total_words = len(old_text.split())
    new_total_words = len(new_text.split())
    
    return {
        "old_long_words_count": len(old_long_words),
        "new_long_words_count": len(new_long_words),
        "old_long_words_examples": old_long_words[:5],
        "new_long_words_examples": new_long_words[:5],
        "improvement": len(old_long_words) - len(new_long_words),
        "old_total_words": old_total_words,
        "new_total_words": new_total_words,
        "word_increase": new_total_words - old_total_words
    }

def main():
    print("🚀 Extraction PDF avec séparation intelligente des mots...")
    
    try:
        # Initialiser le séparateur de mots intelligent
        word_splitter = IntelligentWordSplitter()
        
        # Charger la config
        settings = load_settings("config/settings.yaml")
        raw_dir = Path(settings["paths"]["raw_documents"])
        json_dir = Path(settings["paths"]["json_documents"])
        
        # Créer un sous-dossier pour cette méthode
        json_dir_intelligent = json_dir / "intelligent"
        json_dir_intelligent.mkdir(parents=True, exist_ok=True)

        pdf_files = list(raw_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ Aucun PDF trouvé dans {raw_dir}")
            return

        print(f"📚 {len(pdf_files)} PDF(s) à traiter...")

        for pdf_path in pdf_files:
            print(f"📄 Traitement de {pdf_path.name}...")
            
            # Extraction intelligente
            data = extract_text_with_pymupdf_intelligent(pdf_path, word_splitter)
            
            # Sauvegarder
            output_path = json_dir_intelligent / f"{pdf_path.stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Statistiques
            total_chars = sum(len(page["text"]) for page in data["pages"])
            total_words = sum(len(page["text"].split()) for page in data["pages"])
            
            print(f"   ✅ {data['num_pages']} pages | {total_words:,} mots | {total_chars:,} caractères")
            
            # Analyser les améliorations vs version basic
            basic_file = json_dir / f"{pdf_path.stem}.json"
            if basic_file.exists():
                with open(basic_file, "r", encoding="utf-8") as f:
                    basic_data = json.load(f)
                
                old_text = " ".join(page["text"] for page in basic_data["pages"])
                new_text = " ".join(page["text"] for page in data["pages"])
                
                improvements = analyze_improvements(old_text, new_text)
                print(f"   📊 Mots longs (>15 chars): {improvements['old_long_words_count']} → {improvements['new_long_words_count']}")
                print(f"   📈 Augmentation mots totaux: +{improvements['word_increase']}")
                
                if improvements['old_long_words_examples']:
                    print(f"   🔧 Exemples corrigés: {improvements['old_long_words_examples'][:3]}")

        print(f"🎉 Extraction intelligente terminée !")
        print(f"📁 Fichiers sauvés dans: {json_dir_intelligent}")
        print(f"💡 Pour tester: modifiez settings.yaml:")
        print(f"   json_documents: data/json_documents/intelligent/")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()