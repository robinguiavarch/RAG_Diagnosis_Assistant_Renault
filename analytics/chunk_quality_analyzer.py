"""
Chunk Quality Analyzer: Document Segmentation Assessment and Validation

This module provides comprehensive quality analysis tools for document chunks
in the RAG diagnosis system. It evaluates chunk structure, content quality,
text extraction integrity, and provides diagnostic capabilities for identifying
poorly segmented or corrupted text data.

Key components:
- Chunk loading and metadata analysis from JSON chunk files
- Text quality assessment including word concatenation detection
- Statistical analysis of chunk distribution and sizing patterns
- Diagnostic tools for identifying extraction method effectiveness
- Quality recommendations based on content analysis

Dependencies: json, pathlib, yaml, random
Usage: Run as standalone script for chunk quality assessment and text integrity validation
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List
import sys
import yaml

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_settings() -> Dict[str, Any]:
    """
    Load configuration settings from YAML file.
    
    Reads the main configuration file to access project paths and settings
    required for chunk quality analysis.
    
    Returns:
        Dict[str, Any]: Configuration dictionary containing all project settings
        
    Raises:
        FileNotFoundError: If settings.yaml file is not found
        yaml.YAMLError: If YAML file is malformed
    """
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_all_chunks(chunk_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all document chunks from the processing directory.
    
    Processes all chunk JSON files and performs quality diagnostics including
    metadata validation and text integrity assessment.
    
    Args:
        chunk_dir (Path): Directory containing processed chunk JSON files
        
    Returns:
        List[Dict[str, Any]]: List of chunk dictionaries with metadata and quality metrics
    """
    all_chunks = []
    chunk_files = list(chunk_dir.glob("*_chunks.json"))
    
    print(f"Chunk files found: {len(chunk_files)}")
    for chunk_file in chunk_files:
        print(f"   {chunk_file.name}")
    
    for chunk_file in chunk_files:
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Diagnostic: Check chunk source and quality
        print(f"\nAnalyzing {chunk_file.name}:")
        if "source_stats" in data:
            print(f"   Modern file with metadata")
            print(f"   Source document: {data.get('document_id', 'N/A')}")
            print(f"   Chunking method: {data.get('chunking_config', {}).get('method', 'N/A')}")
        else:
            print(f"   Legacy file without metadata")
        
        # Analyze sample chunks for concatenated words
        sample_chunks = data["chunks"][:2] if "chunks" in data else []
        for i, chunk in enumerate(sample_chunks):
            text = chunk.get("text", "")
            words = text.split()
            long_words = [w for w in words if len(w) > 15]
            if long_words:
                print(f"   Warning: Chunk {i}: suspicious words ({len(long_words)}): {long_words[:3]}")
            else:
                print(f"   Clean: Chunk {i}: text appears clean")
        
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
    """
    Display formatted chunk with quality diagnostics.
    
    Presents chunk content with metadata, quality assessment, and text integrity
    analysis including detection of concatenated words and extraction issues.
    
    Args:
        chunk (Dict[str, Any]): Chunk data with metadata
        index (int): Display index for identification
        
    Returns:
        None: Formatted output is printed to console
    """
    print(f"{'='*80}")
    print(f"CHUNK #{index + 1}")
    print(f"{'='*80}")
    print(f"Document: {chunk['document_id']}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Statistics: {chunk['word_count']} words | {chunk['char_count']} characters")
    print(f"Source file: {chunk['source_file']}")
    print(f"Has metadata: {'Yes' if chunk['has_metadata'] else 'No'}")
    print(f"Method: {chunk['extraction_method']}")
    
    # Text quality diagnostic
    text = chunk['text']
    words = text.split()
    long_words = [w for w in words if len(w) > 15]
    
    if long_words:
        print(f"ISSUE: {len(long_words)} suspicious words detected")
        print(f"   Examples: {long_words[:5]}")
    else:
        print(f"QUALITY: Text appears clean")
    
    print(f"{'─'*80}")
    print("CONTENT:")
    print(f"{'─'*80}")
    
    # Display text with line breaks for readability
    text_lines = chunk['text'].split('. ')
    for line in text_lines:
        if line.strip():
            print(f"   {line.strip()}{'.' if not line.endswith('.') else ''}")
    
    print(f"{'─'*80}")
    print()

def display_chunks_statistics(all_chunks: List[Dict[str, Any]]):
    """
    Display comprehensive chunk statistics with quality diagnostics.
    
    Computes and presents aggregate statistics across all chunks including
    quality metrics, metadata coverage, and distribution analysis.
    
    Args:
        all_chunks (List[Dict[str, Any]]): List of all chunk data
        
    Returns:
        None: Statistics are printed to console
    """
    if not all_chunks:
        print("No chunks found")
        return
    
    word_counts = [chunk['word_count'] for chunk in all_chunks]
    char_counts = [chunk['char_count'] for chunk in all_chunks]
    
    # Unique documents
    unique_docs = set(chunk['document_id'] for chunk in all_chunks)
    
    # Quality diagnostics
    chunks_with_metadata = [c for c in all_chunks if c['has_metadata']]
    chunks_with_long_words = []
    
    for chunk in all_chunks:
        words = chunk['text'].split()
        long_words = [w for w in words if len(w) > 15]
        if long_words:
            chunks_with_long_words.append(chunk)
    
    print(f"GENERAL STATISTICS")
    print(f"{'='*50}")
    print(f"Number of documents: {len(unique_docs)}")
    print(f"Total number of chunks: {len(all_chunks)}")
    print(f"Average size (words): {sum(word_counts) / len(word_counts):.1f}")
    print(f"Size range (words): {min(word_counts)} / {max(word_counts)}")
    print()
    
    # Quality diagnostics
    print(f"QUALITY DIAGNOSTICS")
    print(f"{'='*50}")
    print(f"Chunks with metadata: {len(chunks_with_metadata)}/{len(all_chunks)}")
    print(f"Chunks with suspicious words: {len(chunks_with_long_words)}/{len(all_chunks)}")
    
    if chunks_with_long_words:
        print(f"ISSUE DETECTED: Some chunks contain concatenated words")
        print(f"   This indicates that intelligent extraction was not used.")
        print(f"   Check the json_documents path in settings.yaml")
    else:
        print(f"QUALITY OK: All chunks appear clean")
    
    print()
    
    # Distribution by document
    doc_chunk_counts = {}
    for chunk in all_chunks:
        doc_id = chunk['document_id']
        doc_chunk_counts[doc_id] = doc_chunk_counts.get(doc_id, 0) + 1
    
    print("DISTRIBUTION BY DOCUMENT:")
    print(f"{'─'*50}")
    for doc_id, count in sorted(doc_chunk_counts.items()):
        print(f"   {doc_id}: {count} chunks")
    print()

def search_chunks_by_keyword(all_chunks: List[Dict[str, Any]], keyword: str) -> List[Dict[str, Any]]:
    """
    Search chunks containing a specific keyword for debugging purposes.
    
    Provides keyword-based filtering of chunks to help identify specific
    content patterns or quality issues in the chunk collection.
    
    Args:
        all_chunks (List[Dict[str, Any]]): List of all chunk data
        keyword (str): Keyword to search for in chunk text
        
    Returns:
        List[Dict[str, Any]]: List of chunks containing the keyword
    """
    matching_chunks = []
    keyword_lower = keyword.lower()
    
    for chunk in all_chunks:
        if keyword_lower in chunk['text'].lower():
            matching_chunks.append(chunk)
    
    return matching_chunks

def main():
    """
    Main execution function for chunk quality analysis.
    
    Orchestrates the complete analysis workflow including path validation,
    chunk loading, quality assessment, and recommendation generation.
    
    Returns:
        None: Results are displayed in console with quality recommendations
    """
    try:
        # Load configuration
        settings = load_settings()
        chunk_dir = Path(settings["paths"]["chunk_documents"])
        json_dir = Path(settings["paths"]["json_documents"])
        
        # Diagnostic: Display paths used
        print(f"PATH DIAGNOSTICS:")
        print(f"{'='*50}")
        print(f"JSON source: {json_dir}")
        print(f"Chunks: {chunk_dir}")
        print(f"JSON exists: {json_dir.exists()}")
        print(f"Chunks exists: {chunk_dir.exists()}")
        
        if json_dir.exists():
            json_files = list(json_dir.glob("*.json"))
            print(f"JSON files found: {len(json_files)}")
            if json_files:
                print(f"First file: {json_files[0].name}")
                
                # Check quality of first JSON file
                with open(json_files[0], "r", encoding="utf-8") as f:
                    sample_data = json.load(f)
                
                if "pages" in sample_data and sample_data["pages"]:
                    sample_text = sample_data["pages"][0]["text"][:200]
                    sample_words = sample_text.split()
                    long_words = [w for w in sample_words if len(w) > 15]
                    
                    print(f"JSON source quality:")
                    if long_words:
                        print(f"   Warning: JSON contains concatenated words: {long_words[:3]}")
                        print(f"   Tip: Use intelligent extraction: scripts/01_extract_text_PyMuPDF_intelligent.py")
                    else:
                        print(f"   Clean: JSON appears clean")
        
        print()
        
        if not chunk_dir.exists():
            print(f"Directory {chunk_dir} does not exist")
            return
        
        print("Loading chunks...")
        all_chunks = load_all_chunks(chunk_dir)
        
        if not all_chunks:
            print("No chunks found in directory")
            return
        
        # Display statistics with diagnostics
        display_chunks_statistics(all_chunks)
        
        # Select 3 random chunks for display
        num_samples = min(3, len(all_chunks))
        random_chunks = random.sample(all_chunks, num_samples)
        
        print(f"DISPLAYING {num_samples} RANDOM CHUNKS")
        print("="*80)
        print()
        
        for i, chunk in enumerate(random_chunks):
            display_chunk(chunk, i)
        
        # Recommendations based on diagnostics
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        chunks_with_issues = [c for c in all_chunks if len([w for w in c['text'].split() if len(w) > 15]) > 0]
        
        if chunks_with_issues:
            print("ISSUE DETECTED:")
            print("   Your chunks contain concatenated words (poorly extracted text)")
            print()
            print("SOLUTION:")
            print("   1. Use intelligent extraction:")
            print("      poetry run python scripts/01_extract_text_PyMuPDF.py")
            print()
            print("   2. Modify settings.yaml to point to JSONs:")
            print("      json_documents: data/json_documents/")
            print()
            print("   3. Re-chunk with clean text:")
            print("      poetry run python scripts/02_chunk_documents.py")
            print()
            print("   4. Re-visualize:")
            print("      poetry run python visualization/visualize_chunks.py")
        else:
            print("QUALITY OK:")
            print("   Your chunks appear clean and ready for RAG")
        
        print("\nDiagnostic completed")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()