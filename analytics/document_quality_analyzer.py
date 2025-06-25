"""
Document Quality Analyzer: Content Assessment and SCR Extraction Validation

This module provides comprehensive quality analysis tools for processed JSON documents
in the RAG diagnosis system. It evaluates document structure, content quality, and
validates SCR (Symptom-Cause-Remedy) triplet extraction capabilities using sequential
pattern matching techniques.

Key components:
- Document loading and structural analysis from JSON files
- Statistical content assessment including word count and page coverage
- Sequential SCR triplet extraction with pattern matching optimization
- Random text sampling for content inspection and validation
- Quality metrics computation and reporting

Dependencies: json, re, pathlib, yaml, random
Usage: Run as standalone script for document quality assessment and SCR validation
"""

import os
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List
import yaml
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_settings() -> Dict[str, Any]:
    """
    Load configuration settings from YAML file.
    
    Reads the main configuration file to access project paths and settings
    required for document quality analysis.
    
    Returns:
        Dict[str, Any]: Configuration dictionary containing all project settings
        
    Raises:
        FileNotFoundError: If settings.yaml file is not found
        yaml.YAMLError: If YAML file is malformed
    """
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_all_documents(json_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all JSON documents from the intelligent extraction directory.
    
    Processes all JSON files in the specified directory and extracts document
    metadata including page count, text content, and basic statistics.
    
    Args:
        json_dir (Path): Directory containing processed JSON documents
        
    Returns:
        List[Dict[str, Any]]: List of document dictionaries with metadata and content
        
    Raises:
        Exception: If JSON files cannot be loaded or parsed
    """
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
            print(f"Warning: Error loading {json_file}: {e}")
    
    return documents

def extract_scr_triplets_sequential(text: str) -> List[Dict[str, Any]]:
    """
    Extract SCR triplets using optimized sequential pattern matching.
    
    Implements a window-based approach to identify error codes and their associated
    causes and remedies within contextual boundaries. Uses regex patterns to
    locate structured content and extract complete triplets.
    
    Args:
        text (str): Full document text to analyze for SCR patterns
        
    Returns:
        List[Dict[str, Any]]: List of extracted triplets with error codes, symptoms, causes, and remedies
    """
    triplets = []
    error_codes = re.findall(r'[A-Z]+-\d+', text)
    
    for i, code in enumerate(error_codes):
        # Find position of this code
        code_pos = text.find(code)
        if code_pos == -1:
            continue
            
        # Define search window (until next code or end of text)
        if i < len(error_codes) - 1:
            next_code = error_codes[i + 1]
            next_pos = text.find(next_code, code_pos + len(code))
            if next_pos != -1:
                window = text[code_pos:next_pos]
            else:
                window = text[code_pos:code_pos + 1500]  # Larger window
        else:
            window = text[code_pos:code_pos + 1500]
        
        # Search for cause and remedy in this window
        cause_match = re.search(r'Cause:\s*(.*?)(?=Remedy:|$)', window, re.DOTALL | re.IGNORECASE)
        remedy_match = re.search(r'Remedy:\s*(.*?)(?=$|\n\n|\d+\.\d+)', window, re.DOTALL | re.IGNORECASE)
        
        if cause_match and remedy_match:
            # Extract title/symptom (line with the code)
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
    """
    Extract random sample of tokens from document text for inspection.
    
    Selects a representative portion of the document text for quality assessment
    and content validation. Provides position information and coverage metrics.
    
    Args:
        text (str): Full document text to sample from
        num_tokens (int): Number of tokens to include in sample (default: 1000)
        
    Returns:
        Dict[str, Any]: Sample information including text, position, and coverage percentage
    """
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
    """
    Display essential document statistics and metrics.
    
    Computes and presents aggregate statistics across all loaded documents
    including page counts, word counts, and coverage metrics.
    
    Args:
        documents (List[Dict[str, Any]]): List of document metadata dictionaries
        
    Returns:
        None: Statistics are printed to console
    """
    if not documents:
        print("No documents found")
        return
    
    total_pages = sum(doc['num_pages'] for doc in documents)
    total_words = sum(doc['word_count'] for doc in documents)
    
    print(f"GENERAL STATISTICS")
    print(f"{'='*60}")
    print(f"Documents: {len(documents)}")
    print(f"Total pages: {total_pages:,}")
    print(f"Total words: {total_words:,}")
    print(f"Words per page (average): {total_words // total_pages:,}")
    print()

def display_scr_results(doc: Dict[str, Any], triplets: List[Dict[str, Any]]):
    """
    Display SCR extraction results and detection metrics.
    
    Presents comprehensive analysis of SCR triplet extraction including
    detection rates, example triplets, and quality assessment.
    
    Args:
        doc (Dict[str, Any]): Document metadata and content
        triplets (List[Dict[str, Any]]): Extracted SCR triplets
        
    Returns:
        None: Results are printed to console
    """
    total_error_codes = len(re.findall(r'[A-Z]+-\d+', doc['full_text']))
    detection_rate = (len(triplets) / total_error_codes * 100) if total_error_codes > 0 else 0
    
    print(f"SCR EXTRACTION (Sequential Method)")
    print(f"{'='*60}")
    print(f"Document analyzed: {doc['document_id']}")
    print(f"Total error codes: {total_error_codes:,}")
    print(f"SCR triplets detected: {len(triplets):,}")
    print(f"Detection rate: {detection_rate:.1f}%")
    print()
    
    # Display one example triplet
    if triplets:
        example = triplets[0]
        print(f"EXAMPLE SCR TRIPLET:")
        print(f"{'─'*60}")
        print(f"Error code: {example['error_code']}")
        print(f"Symptom: {example['symptom']}")
        print(f"Cause: {example['cause'][:150]}{'...' if len(example['cause']) > 150 else ''}")
        print(f"Remedy: {example['remedy'][:150]}{'...' if len(example['remedy']) > 150 else ''}")
        print()

def display_text_sample(doc: Dict[str, Any], sample_info: Dict[str, Any]):
    """
    Display document text sample for content inspection.
    
    Shows a formatted sample of document content with position information
    and basic formatting for readability assessment.
    
    Args:
        doc (Dict[str, Any]): Document metadata
        sample_info (Dict[str, Any]): Sample text and position information
        
    Returns:
        None: Sample text is printed to console
    """
    print(f"DOCUMENT JSON PREVIEW")
    print(f"{'='*60}")
    print(f"Document: {doc['document_id']}")
    print(f"Position: words {sample_info['start_position']:,} to {sample_info['end_position']:,}")
    print(f"Coverage: {sample_info['coverage_percent']:.1f}% of document")
    print(f"{'─'*60}")
    
    # Display text with simple formatting
    sample_text = sample_info['sample']
    lines = sample_text.replace('. ', '.\n').split('\n')
    
    # Take only the first 15 lines to avoid verbosity
    for line in lines[:15]:
        line = line.strip()
        if line and len(line) > 10:  # Ignore lines that are too short
            if len(line) > 100:
                print(f"   {line[:100]}...")
            else:
                print(f"   {line}")
    
    if len(lines) > 15:
        print(f"   ... ({len(lines) - 15} additional lines)")
    print()

def main():
    """
    Main execution function for document quality analysis.
    
    Orchestrates the complete analysis workflow including document loading,
    statistical analysis, SCR extraction validation, and content sampling.
    
    Returns:
        None: Results are displayed in console and analysis is completed
    """
    try:
        print("Analyzing cleaned JSON documents...")
        
        # Load configuration
        settings = load_settings()
        json_dir = Path(settings["paths"]["json_documents"])
        
        if not json_dir.exists():
            print(f"Directory {json_dir} does not exist")
            print(f"Please run first: python 01_extract_text_PyMuPDF_intelligent.py")
            return
        
        # Load documents
        documents = load_all_documents(json_dir)
        if not documents:
            print("No JSON documents found")
            return
        
        # General statistics
        display_document_stats(documents)
        
        # Analyze the largest document
        target_doc = max(documents, key=lambda x: x['word_count'])
        print(f"SCR analysis on main document: {target_doc['document_id']}")
        print()
        
        # SCR extraction with sequential method
        triplets = extract_scr_triplets_sequential(target_doc['full_text'])
        display_scr_results(target_doc, triplets)
        
        # Document preview
        sample_info = extract_random_sample(target_doc['full_text'], num_tokens=800)
        display_text_sample(target_doc, sample_info)
        
        # Final summary
        print(f"SUMMARY")
        print(f"{'='*40}")
        print(f"Sequential method validated")
        print(f"Approximately {len(triplets):,} extractable triplets")
        print(f"Ready to rewrite 00_extract_scr_triplets.py")
        print()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()