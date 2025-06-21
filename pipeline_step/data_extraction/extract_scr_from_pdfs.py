import os
import json
import re
import csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml

def load_settings(config_path: str) -> Dict[str, Any]:
    """Charge la configuration depuis le fichier YAML"""
    # Gérer l'exécution depuis le dossier scripts/
    if not os.path.exists(config_path):
        config_path = os.path.join("..", config_path)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def extract_scr_triplets_with_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extrait les triplets SCR avec la méthode séquentielle optimisée
    Inclut le numéro de page pour chaque triplet
    """
    all_triplets = []
    
    # Construire un texte global avec marqueurs de pages
    full_text_with_markers = ""
    page_positions = {}  # position_char -> page_number
    current_pos = 0
    
    for page_data in pages:
        page_num = page_data["page_number"]
        page_text = page_data["text"]
        
        # Marquer le début de cette page
        page_positions[current_pos] = page_num
        full_text_with_markers += page_text + "\n"
        current_pos = len(full_text_with_markers)
    
    # Trouver tous les codes d'erreur avec leurs positions
    error_codes_with_pos = []
    for match in re.finditer(r'[A-Z]+-\d+', full_text_with_markers):
        error_codes_with_pos.append({
            'code': match.group(),
            'start_pos': match.start(),
            'end_pos': match.end()
        })
    
    # Extraire les triplets pour chaque code d'erreur
    for i, error_info in enumerate(error_codes_with_pos):
        code = error_info['code']
        code_pos = error_info['start_pos']
        
        # Déterminer le numéro de page pour ce code
        page_num = 1  # Par défaut
        for pos, page in page_positions.items():
            if pos <= code_pos:
                page_num = page
            else:
                break
        
        # Définir la fenêtre de recherche (jusqu'au prochain code ou fin de texte)
        if i < len(error_codes_with_pos) - 1:
            next_code_pos = error_codes_with_pos[i + 1]['start_pos']
            window = full_text_with_markers[code_pos:next_code_pos]
        else:
            # Dernière occurrence - prendre jusqu'à 2000 caractères
            window = full_text_with_markers[code_pos:code_pos + 2000]
        
        # Chercher cause et remedy dans cette fenêtre
        cause_match = re.search(r'Cause:\s*(.*?)(?=Remedy:|$)', window, re.DOTALL | re.IGNORECASE)
        remedy_match = re.search(r'Remedy:\s*(.*?)(?=$|\n\n|\d+\.\d+|[A-Z]+-\d+)', window, re.DOTALL | re.IGNORECASE)
        
        if cause_match and remedy_match:
            # Extraire le symptôme (ligne avec le code d'erreur)
            symptom_match = re.search(rf'({re.escape(code)}.*?)(?=\s*Cause:|$)', window, re.IGNORECASE)
            symptom = symptom_match.group(1) if symptom_match else code
            
            # Nettoyer les textes extraits
            cause_text = clean_extracted_text(cause_match.group(1))
            remedy_text = clean_extracted_text(remedy_match.group(1))
            symptom_text = clean_extracted_text(symptom)
            
            # Valider que nous avons des contenus significatifs
            if len(cause_text) > 10 and len(remedy_text) > 10:
                triplet = {
                    'error_code': code,
                    'page_number': page_num,
                    'symptom': symptom_text,
                    'cause': cause_text,
                    'remedy': remedy_text
                }
                all_triplets.append(triplet)
    
    return all_triplets

def clean_extracted_text(text: str) -> str:
    """Nettoie le texte extrait des triplets"""
    if not text:
        return ""
    
    # Supprimer les retours à la ligne multiples
    text = re.sub(r'\n+', ' ', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les caractères de contrôle indésirables
    text = re.sub(r'[^\x20-\x7E\xC0-\xFF]', '', text)
    
    # Nettoyer les débuts/fins
    text = text.strip()
    
    # Supprimer les artefacts OCR courants
    text = re.sub(r'\$\.\$', '', text)  # Artefacts comme $.$
    text = re.sub(r'\^[0-9]', '', text)  # Artefacts comme ^4, ^5
    
    return text

def get_equipment_name() -> str:
    """Demande à l'utilisateur de saisir le nom de l'équipement"""
    print("\n" + "="*60)
    print("🔧 CONFIGURATION DE L'ÉQUIPEMENT")
    print("="*60)
    print("📋 Veuillez spécifier le nom de l'équipement pour ce document.")
    print("💡 Exemples: 'Robot FANUC R-30iB', 'Machine CNC', 'Soudeuse Arc', etc.")
    print()
    
    while True:
        equipment = input("🏭 Nom de l'équipement: ").strip()
        if equipment and len(equipment) >= 2:
            return equipment
        else:
            print("⚠️ Veuillez entrer un nom d'équipement valide (au moins 2 caractères)")

def save_triplets_to_csv(triplets: List[Dict[str, Any]], pdf_filename: str, equipment: str, output_path: Path):
    """Sauvegarde les triplets au format CSV avec les colonnes demandées"""
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Écrire le CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['URL', 'equipment', 'page', 'symptom', 'cause', 'remedy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Écrire l'en-tête
        writer.writeheader()
        
        # Écrire les données
        for triplet in triplets:
            writer.writerow({
                'URL': pdf_filename,
                'equipment': equipment,
                'page': triplet['page_number'],
                'symptom': triplet['symptom'],
                'cause': triplet['cause'],
                'remedy': triplet['remedy']
            })

def display_extraction_summary(triplets: List[Dict[str, Any]], doc_id: str, equipment: str):
    """Affiche un résumé de l'extraction"""
    print(f"\n📊 RÉSUMÉ DE L'EXTRACTION")
    print(f"{'='*60}")
    print(f"📄 Document: {doc_id}")
    print(f"🏭 Équipement: {equipment}")
    print(f"✅ Triplets extraits: {len(triplets):,}")
    
    if triplets:
        # Statistiques par page
        pages_with_triplets = set(t['page_number'] for t in triplets)
        print(f"📖 Pages concernées: {len(pages_with_triplets)}")
        print(f"📊 Moyenne par page: {len(triplets) / len(pages_with_triplets):.1f} triplets")
        
        # Codes d'erreur uniques
        unique_codes = set(t['error_code'] for t in triplets)
        print(f"🚨 Codes d'erreur uniques: {len(unique_codes)}")
        
        # Exemple de triplet
        example = triplets[0]
        print(f"\n📋 PREMIER TRIPLET EXTRAIT:")
        print(f"{'─'*60}")
        print(f"🔸 Page: {example['page_number']}")
        print(f"🔸 Code: {example['error_code']}")
        print(f"🔸 Symptôme: {example['symptom'][:80]}{'...' if len(example['symptom']) > 80 else ''}")
        print(f"🔸 Cause: {example['cause'][:80]}{'...' if len(example['cause']) > 80 else ''}")
        print(f"🔸 Remède: {example['remedy'][:80]}{'...' if len(example['remedy']) > 80 else ''}")
    
    print()

def process_document(json_file: Path, output_dir: Path) -> bool:
    """Traite un document JSON et extrait les triplets SCR"""
    
    try:
        # Charger le document JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        doc_id = doc_data['document_id']
        pages = doc_data['pages']
        
        print(f"📄 Traitement de: {doc_id}")
        print(f"📖 Pages à analyser: {len(pages)}")
        
        # Demander le nom de l'équipement
        equipment = get_equipment_name()
        
        print(f"\n🔍 Extraction des triplets SCR en cours...")
        
        # Extraire les triplets
        triplets = extract_scr_triplets_with_pages(pages)
        
        if not triplets:
            print(f"⚠️ Aucun triplet SCR trouvé dans {doc_id}")
            return False
        
        # Nom du fichier PDF original (supposé être le même que le JSON)
        pdf_filename = f"{doc_id}.pdf"
        
        # Chemin de sortie CSV
        csv_filename = f"{doc_id}_scr_triplets.csv"
        csv_path = output_dir / csv_filename
        
        # Sauvegarder en CSV
        save_triplets_to_csv(triplets, pdf_filename, equipment, csv_path)
        
        # Afficher le résumé
        display_extraction_summary(triplets, doc_id, equipment)
        
        print(f"💾 Sauvegardé: {csv_path}")
        print(f"✅ Extraction terminée avec succès!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {json_file}: {e}")
        return False

def main():
    try:
        print("🚀 Extraction des triplets Symptôme-Cause-Remède")
        print("="*60)
        
        # Charger la configuration (gérer l'exécution depuis scripts/)
        config_path = "config/settings.yaml"
        if not os.path.exists(config_path):
            config_path = "../config/settings.yaml"
            
        settings = load_settings(config_path)
        
        # Chemins avec ajustement pour scripts/
        base_json_dir = Path(settings["paths"]["json_documents"])
        
        # Utiliser extract_scr du config ou chemin par défaut
        if "extract_scr" in settings["paths"]:
            base_output_dir = Path(settings["paths"]["extract_scr"])
        else:
            base_output_dir = Path("data/extract_scr")  # Chemin par défaut
        
        # Ajuster les chemins si on exécute depuis scripts/
        if not base_json_dir.exists():
            base_json_dir = Path("..") / base_json_dir
            base_output_dir = Path("..") / base_output_dir
        
        json_dir = base_json_dir / "intelligent"
        output_dir = base_output_dir
        
        # Vérifier que le répertoire JSON existe
        if not json_dir.exists():
            print(f"❌ Le répertoire {json_dir} n'existe pas!")
            print(f"💡 Exécutez d'abord: python 01_extract_text_PyMuPDF_intelligent.py")
            return
        
        # Créer le répertoire de sortie
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trouver les fichiers JSON
        json_files = list(json_dir.glob("*.json"))
        
        if not json_files:
            print(f"❌ Aucun fichier JSON trouvé dans {json_dir}")
            return
        
        print(f"📚 {len(json_files)} document(s) à traiter")
        
        # Traiter chaque document
        successful_extractions = 0
        
        for json_file in json_files:
            print(f"\n{'='*60}")
            if process_document(json_file, output_dir):
                successful_extractions += 1
        
        # Résumé final
        print(f"\n🎉 EXTRACTION TERMINÉE")
        print(f"{'='*60}")
        print(f"✅ Documents traités avec succès: {successful_extractions}/{len(json_files)}")
        print(f"📁 Fichiers CSV générés dans: {output_dir}")
        print(f"💡 Prêt pour analyse ou intégration dans Knowledge Graph")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()