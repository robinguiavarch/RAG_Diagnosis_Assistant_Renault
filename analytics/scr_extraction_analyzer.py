import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import yaml
from collections import Counter

# AJOUTER au début du script
import sys
from pathlib import Path

# Ajouter la racine du projet au Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_settings() -> Dict[str, Any]:
    """Charge la configuration depuis settings.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_scr_data(extract_dir: Path) -> pd.DataFrame:
    """Charge tous les fichiers CSV d'extraction SCR"""
    csv_files = list(extract_dir.glob("*_scr_triplets.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {extract_dir}")
    
    # Charger et combiner tous les CSV
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source_file'] = csv_file.name
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Nettoyer et préparer les données
    combined_df['error_code'] = combined_df['symptom'].str.extract(r'([A-Z]+-\d+)')
    combined_df['error_prefix'] = combined_df['error_code'].str.split('-').str[0]
    combined_df['cause_length'] = combined_df['cause'].str.len()
    combined_df['remedy_length'] = combined_df['remedy'].str.len()
    
    return combined_df

def generate_statistics_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Génère des statistiques détaillées sur l'extraction"""
    stats = {
        'total_triplets': len(df),
        'unique_documents': df['URL'].nunique(),
        'unique_equipments': df['equipment'].nunique(),
        'unique_error_codes': df['error_code'].nunique(),
        'unique_error_prefixes': df['error_prefix'].nunique(),
        'total_pages': df['page'].nunique(),
        'pages_range': (df['page'].min(), df['page'].max()),
        'avg_triplets_per_page': len(df) / df['page'].nunique(),
        'avg_cause_length': df['cause_length'].mean(),
        'avg_remedy_length': df['remedy_length'].mean(),
        'error_prefixes_distribution': dict(df['error_prefix'].value_counts().head(10))
    }
    
    return stats

def create_summary_report(df: pd.DataFrame, stats: Dict[str, Any], output_dir: Path):
    """Crée un rapport de synthèse en texte"""
    
    report_content = f"""
# RAPPORT D'ANALYSE - EXTRACTION SCR
=====================================================

## 📊 STATISTIQUES GÉNÉRALES
- **Total de triplets extraits**: {stats['total_triplets']:,}
- **Documents traités**: {stats['unique_documents']}
- **Équipements différents**: {stats['unique_equipments']}
- **Codes d'erreur uniques**: {stats['unique_error_codes']:,}
- **Préfixes d'erreur uniques**: {stats['unique_error_prefixes']}

## 📖 COUVERTURE DES PAGES
- **Total de pages concernées**: {stats['total_pages']:,}
- **Plage de pages**: {stats['pages_range'][0]} à {stats['pages_range'][1]}
- **Moyenne triplets/page**: {stats['avg_triplets_per_page']:.1f}

## 📝 ANALYSE DU CONTENU
- **Longueur moyenne des causes**: {stats['avg_cause_length']:.0f} caractères
- **Longueur moyenne des remèdes**: {stats['avg_remedy_length']:.0f} caractères

## 🔝 TOP PRÉFIXES D'ERREUR
"""
    
    for prefix, count in list(stats['error_prefixes_distribution'].items())[:10]:
        percentage = (count / stats['total_triplets']) * 100
        report_content += f"- **{prefix}**: {count:,} triplets ({percentage:.1f}%)\n"
    
    # Sauvegarder le rapport
    with open(output_dir / 'rapport_analyse_scr.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("📄 Rapport détaillé sauvegardé: rapport_analyse_scr.md")

def display_console_summary(stats: Dict[str, Any]):
    """Affiche un résumé dans la console"""
    print(f"\n📊 RÉSUMÉ DE L'ANALYSE SCR")
    print(f"{'='*60}")
    print(f"✅ Total triplets: {stats['total_triplets']:,}")
    print(f"📄 Documents: {stats['unique_documents']}")
    print(f"🏭 Équipements: {stats['unique_equipments']}")
    print(f"🚨 Codes d'erreur uniques: {stats['unique_error_codes']:,}")
    print(f"📖 Pages concernées: {stats['total_pages']:,}")
    print(f"📊 Triplets/page (moy): {stats['avg_triplets_per_page']:.1f}")
    
    print(f"\n🔝 TOP 10 PRÉFIXES D'ERREUR:")
    for i, (prefix, count) in enumerate(list(stats['error_prefixes_distribution'].items())[:10], 1):
        percentage = (count / stats['total_triplets']) * 100
        print(f"   {i:2d}. {prefix}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n📏 LONGUEURS MOYENNES:")
    print(f"   Causes: {stats['avg_cause_length']:.0f} caractères")
    print(f"   Remèdes: {stats['avg_remedy_length']:.0f} caractères")

def display_dataframe_sample(df: pd.DataFrame):
    """Affiche les 50 premières lignes du DataFrame de manière lisible"""
    print(f"\n📊 DATAFRAME - 50 PREMIÈRES LIGNES")
    print(f"{'='*120}")
    
    # Afficher les colonnes principales du CSV d'extraction
    main_columns = ['URL', 'equipment', 'page', 'symptom', 'cause', 'remedy']
    
    # Vérifier que les colonnes existent
    available_columns = [col for col in main_columns if col in df.columns]
    
    if len(available_columns) < 6:
        print(f"⚠️ Colonnes manquantes. Colonnes disponibles: {list(df.columns)}")
        display_df = df[available_columns]
    else:
        display_df = df[available_columns].copy()
        
        # Tronquer les textes longs pour l'affichage dans le tableau
        display_df['symptom_short'] = display_df['symptom'].str[:50] + '...'
        display_df['cause_short'] = display_df['cause'].str[:40] + '...'
        display_df['remedy_short'] = display_df['remedy'].str[:40] + '...'
        
        # Créer un DataFrame d'affichage avec textes tronqués
        clean_df = pd.DataFrame({
            'URL': display_df['URL'],
            'Equipment': display_df['equipment'].str[:20] + '...',
            'Page': display_df['page'],
            'Symptôme': display_df['symptom_short'],
            'Cause': display_df['cause_short'],
            'Remède': display_df['remedy_short']
        })
        
        # Configurer l'affichage pandas
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.max_colwidth', 50)
        pd.set_option('display.max_rows', 50)
        
        # Afficher le DataFrame nettoyé
        print(clean_df.head(50))
    
    print(f"\n📋 QUELQUES EXEMPLES DÉTAILLÉS:")
    print(f"{'─'*120}")
    
    # Afficher 3 exemples complets ligne par ligne
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\n🔹 TRIPLET {i+1}")
        print(f"   URL:       {row.get('URL', 'N/A')}")
        print(f"   Equipment: {row.get('equipment', 'N/A')}")
        print(f"   Page:      {row.get('page', 'N/A')}")
        print(f"   Symptôme:  {row.get('symptom', 'N/A')}")
        print(f"   Cause:     {row.get('cause', 'N/A')}")
        print(f"   Remède:    {row.get('remedy', 'N/A')}")
        print(f"   {'─'*100}")
    
    print(f"\n📋 INFORMATIONS SUR LE DATAFRAME:")
    print(f"{'─'*60}")
    print(f"📏 Forme: {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"📊 Colonnes réelles: {list(df.columns)}")
    print(f"📝 Colonnes attendues: ['URL', 'equipment', 'page', 'symptom', 'cause', 'remedy']")
    
    if 'page' in df.columns:
        print(f"\n📈 STATISTIQUES RAPIDES:")
        print(f"   Pages min/max: {df['page'].min()} - {df['page'].max()}")
        if 'cause' in df.columns:
            print(f"   Longueur cause moy/max: {df['cause'].str.len().mean():.0f} / {df['cause'].str.len().max()}")
        if 'remedy' in df.columns:
            print(f"   Longueur remède moy/max: {df['remedy'].str.len().mean():.0f} / {df['remedy'].str.len().max()}")

def main():
    try:
        print("🔍 Visualisation des résultats d'extraction SCR")
        print("="*60)
        
        # Charger la configuration
        settings = load_settings()
        base_extract_dir = Path(settings["paths"]["scr_triplets"])
        
        # Répertoire de sortie pour les visualisations
        output_dir = Path(settings["paths"]["outputs"]) / "analytic_reports"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Chargement des données depuis: {base_extract_dir}")
        
        # Charger les données
        df = load_scr_data(base_extract_dir)
        print(f"✅ Données chargées: {len(df):,} triplets")
        
        # Générer les statistiques
        print("📊 Génération des statistiques...")
        stats = generate_statistics_report(df)
        
        # Afficher le résumé dans la console
        display_console_summary(stats)
        
        # Afficher le DataFrame
        display_dataframe_sample(df)
        
        # Créer le rapport de synthèse
        print(f"\n📝 Génération du rapport...")
        create_summary_report(df, stats, output_dir)
        
        print(f"\n🎉 ANALYSE TERMINÉE")
        print(f"{'='*60}")
        print(f"📁 Rapport généré dans: {output_dir}")
        print(f"📄 Fichier: rapport_analyse_scr.md")
        
    except FileNotFoundError as e:
        print(f"❌ Fichiers non trouvés: {e}")
        print(f"💡 Exécutez d'abord: python scripts/00_extract_scr_triplets.py")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()