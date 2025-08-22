#!/usr/bin/env python3
"""
Investigation script to analyze content quality issues in search results.
Specifically looking for citation-heavy chunks that rank inappropriately high.
"""
import pandas as pd
import re
from typing import List, Dict, Any

def analyze_chunk_content(content: str) -> Dict[str, Any]:
    """Analyze a chunk to detect if it's citation-heavy or low-quality"""
    
    # Citation patterns
    citation_patterns = [
        r'\d{4}[.,;]',  # Years (2020, 2021, etc.)
        r'[A-Z][a-z]+\s+et\s+al[.,]',  # "Author et al."
        r'[A-Z][a-z]+,\s*[A-Z]\.',  # "Smith, J."
        r'[A-Z][a-z]+\s+&\s+[A-Z][a-z]+',  # "Smith & Jones"
        r'pp?\.\s*\d+',  # Page numbers "p. 123" or "pp. 123-456"
        r'Vol\.\s*\d+',  # Volume numbers
        r'doi:', # DOI identifiers
        r'Published in final edited form as:', # Common citation text
    ]
    
    # Content quality indicators
    metadata_patterns = [
        r'Abstract',
        r'Keywords:',
        r'References',
        r'Bibliography',
        r'Table of Contents',
        r'Index',
    ]
    
    # Count matches
    citation_matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in citation_patterns)
    metadata_matches = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in metadata_patterns)
    
    # Calculate ratios
    word_count = len(content.split())
    citation_ratio = citation_matches / max(word_count, 1)
    metadata_ratio = metadata_matches / max(word_count, 1)
    
    # Determine content type
    is_citation_heavy = citation_ratio > 0.1 or citation_matches > 5
    is_metadata = metadata_ratio > 0.05 or any(pattern.lower() in content.lower() for pattern in ['abstract', 'keywords:', 'references', 'bibliography'])
    
    # Check for actual substantive content
    sentences = re.split(r'[.!?]+', content)
    substantive_sentences = [s for s in sentences if len(s.split()) > 5 and not any(re.search(p, s, re.IGNORECASE) for p in citation_patterns[:3])]
    
    return {
        "word_count": word_count,
        "citation_matches": citation_matches,
        "citation_ratio": citation_ratio,
        "metadata_matches": metadata_matches,
        "is_citation_heavy": is_citation_heavy,
        "is_metadata": is_metadata,
        "substantive_sentences": len(substantive_sentences),
        "content_quality_score": len(substantive_sentences) / max(len(sentences), 1),
        "first_100_chars": content[:100] + "..." if len(content) > 100 else content
    }

def investigate_processed_documents():
    """Analyze the processed documents CSV for content quality patterns"""
    
    print("🔍 CONTENT QUALITY INVESTIGATION")
    print("=" * 60)
    
    try:
        # Load processed documents
        df = pd.read_csv("processed_rag_documents.csv")
        print(f"✅ Loaded {len(df)} processed chunks")
        
        # Analyze each chunk
        analyses = []
        for idx, row in df.iterrows():
            chunk_id = row['uuid_chunk']
            content = str(row['chunk_text'])
            
            analysis = analyze_chunk_content(content)
            analysis['chunk_id'] = chunk_id
            analysis['file_name'] = row['file_name']
            analyses.append(analysis)
        
        results_df = pd.DataFrame(analyses)
        
        # Summary statistics
        print(f"\n📊 CONTENT QUALITY SUMMARY:")
        print(f"Total chunks: {len(results_df)}")
        print(f"Citation-heavy chunks: {results_df['is_citation_heavy'].sum()} ({results_df['is_citation_heavy'].mean():.1%})")
        print(f"Metadata chunks: {results_df['is_metadata'].sum()} ({results_df['is_metadata'].mean():.1%})")
        print(f"Average content quality score: {results_df['content_quality_score'].mean():.3f}")
        
        # Show worst offenders
        print(f"\n⚠️  TOP 10 CITATION-HEAVY CHUNKS:")
        citation_heavy = results_df[results_df['is_citation_heavy']].nlargest(10, 'citation_ratio')
        for idx, row in citation_heavy.iterrows():
            print(f"  {row['chunk_id']}: {row['citation_ratio']:.3f} ratio, {row['citation_matches']} matches")
            print(f"    Content: {row['first_100_chars']}")
            print(f"    File: {row['file_name']}\n")
        
        # Show highest quality chunks
        print(f"\n✅ TOP 10 HIGHEST QUALITY CHUNKS:")
        high_quality = results_df.nlargest(10, 'content_quality_score')
        for idx, row in high_quality.iterrows():
            print(f"  {row['chunk_id']}: Quality score {row['content_quality_score']:.3f}")
            print(f"    Content: {row['first_100_chars']}")
            print(f"    File: {row['file_name']}\n")
        
        # Check specific problematic chunk if provided
        problematic_chunk = "5c76efb6-7f3d-11f0-aed4-191370c92eef"
        problem_analysis = results_df[results_df['chunk_id'].str.contains(problematic_chunk[:8], na=False)]
        
        if not problem_analysis.empty:
            print(f"\n🎯 ANALYSIS OF PROBLEMATIC CHUNK {problematic_chunk}:")
            row = problem_analysis.iloc[0]
            print(f"  Citation ratio: {row['citation_ratio']:.3f}")
            print(f"  Citation matches: {row['citation_matches']}")
            print(f"  Is citation-heavy: {row['is_citation_heavy']}")
            print(f"  Content quality score: {row['content_quality_score']:.3f}")
            print(f"  Content preview: {row['first_100_chars']}")
        else:
            print(f"\n❌ Problematic chunk {problematic_chunk} not found in dataset")
        
        return results_df
        
    except FileNotFoundError:
        print("❌ processed_rag_documents.csv not found")
        return None
    except Exception as e:
        print(f"❌ Error analyzing documents: {e}")
        return None

def suggest_fixes(results_df: pd.DataFrame) -> List[str]:
    """Suggest fixes based on analysis results"""
    
    fixes = []
    
    if results_df is None:
        return ["Cannot suggest fixes - analysis failed"]
    
    citation_heavy_pct = results_df['is_citation_heavy'].mean()
    avg_quality = results_df['content_quality_score'].mean()
    
    if citation_heavy_pct > 0.1:
        fixes.append("🔧 HIGH PRIORITY: Filter out citation-heavy chunks (>10% are citation-heavy)")
        fixes.append("   Implementation: Add content quality filter in document processing")
    
    if avg_quality < 0.5:
        fixes.append("🔧 MEDIUM PRIORITY: Improve chunk quality scoring")
        fixes.append("   Implementation: Weight results by content quality score")
    
    fixes.append("🔧 IMMEDIATE FIX: Adjust BM25 scoring to penalize citation-heavy content")
    fixes.append("   Implementation: Multiply BM25 score by content_quality_score")
    
    fixes.append("🔧 LONG-TERM: Improve document chunking strategy")
    fixes.append("   Implementation: Separate citations from main content during chunking")
    
    return fixes

if __name__ == "__main__":
    results = investigate_processed_documents()
    
    print(f"\n🛠️  RECOMMENDED FIXES:")
    fixes = suggest_fixes(results)
    for fix in fixes:
        print(f"  {fix}")
    
    print(f"\n📝 CONCLUSION:")
    print("  Run this investigation after each test query to see if citation-heavy")
    print("  chunks are consistently ranking high inappropriately.")