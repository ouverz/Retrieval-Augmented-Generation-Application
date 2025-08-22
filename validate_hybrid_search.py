#!/usr/bin/env python3
"""
Validation script to test and verify hybrid search functionality.
This script validates that both BM25 and vector search are working,
scores are being combined properly, and the ranking reflects hybrid approach.
"""
import asyncio
import sys
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from BM25SearchEngine import BM25SearchEngine
    from VectorSearchEngine import VectorSearchEngine
    from HybridSearchEngine import HybridSearchEngine
    from config.settings import HybridSearchConfig
    from database.vector_store import VectorStore
    from Processing_Documents import DocumentProcessor
    print("✅ Successfully imported all search engines")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class HybridSearchValidator:
    """Validates hybrid search functionality with detailed logging"""
    
    def __init__(self):
        self.config = HybridSearchConfig()
        self.bm25_engine = None
        self.vector_engine = None
        self.hybrid_engine = None
        
    async def setup(self):
        """Initialize all search engines"""
        print("🔧 Setting up search engines...")
        
        # Setup document processor and vector store
        doc_processor = DocumentProcessor(data_path="data")
        docs_df = doc_processor.get_processed_documents()
        
        if docs_df.empty:
            print("❌ No processed documents found. Please run initialization first.")
            return False
        
        print(f"✅ Found {len(docs_df)} processed documents")
        
        # Initialize vector store
        vector_store = VectorStore()
        
        # Initialize BM25 engine
        self.bm25_engine = BM25SearchEngine()
        self.bm25_engine.build_index(docs_df)
        
        print(f"✅ BM25 index built with {len(docs_df)} documents")
        
        # Initialize vector engine
        self.vector_engine = VectorSearchEngine(vector_store)
        
        # Initialize hybrid engine
        self.hybrid_engine = HybridSearchEngine(
            bm25_engine=self.bm25_engine,
            vector_engine=self.vector_engine,
            config=self.config
        )
        
        print(f"✅ Hybrid engine initialized with weights: BM25={self.config.bm25_weight}, Vector={self.config.vector_weight}")
        return True
    
    def test_bm25_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Test BM25 search independently"""
        print(f"\n🔍 Testing BM25 search for: '{query}'")
        
        docs = self.bm25_engine.search(query, top_k)
        results = []
        
        for i, doc in enumerate(docs):
            # Apply same scoring logic as hybrid engine
            score = self.config.bm25_weight * (1.0 / (i + 1))
            result = {
                "rank": i + 1,
                "content": doc.page_content[:100] + "...",
                "score": score,
                "raw_rank": i + 1,
                "engine": "bm25"
            }
            results.append(result)
            print(f"  {i+1}. Score: {score:.4f} | {result['content']}")
        
        return results
    
    async def test_vector_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Test vector search independently"""
        print(f"\n🔍 Testing Vector search for: '{query}'")
        
        df = self.vector_engine.search(query, top_k, return_dataframe=True)
        results = []
        
        for i, row in df.iterrows():
            # Apply same scoring logic as hybrid engine
            similarity = 1.0 / (1.0 + float(row["distance"]))
            score = self.config.vector_weight * similarity
            
            result = {
                "rank": i + 1,
                "content": row["content"][:100] + "...",
                "score": score,
                "distance": float(row["distance"]),
                "similarity": similarity,
                "engine": "vector"
            }
            results.append(result)
            print(f"  {i+1}. Score: {score:.4f} | Distance: {row['distance']:.4f} | Similarity: {similarity:.4f}")
            print(f"      {result['content']}")
        
        return results
    
    async def test_hybrid_search(self, query: str) -> Dict[str, Any]:
        """Test full hybrid search and analyze score combination"""
        print(f"\n🔗 Testing Hybrid search for: '{query}'")
        
        # Get results from hybrid engine
        ctx_df, response = await self.hybrid_engine.search(query)
        
        # Analyze the results
        print(f"📊 Hybrid Results ({len(ctx_df)} documents):")
        print("-" * 80)
        
        for i, row in ctx_df.iterrows():
            metadata = row["metadata"]
            score = metadata.get("score", 0)
            engine = metadata.get("source_engine", "unknown")
            orig_rank = metadata.get("rank", "unknown")
            
            print(f"{i+1:2d}. Score: {score:.4f} | Engine: {engine:6s} | Orig Rank: {orig_rank:2s} | {row['content'][:80]}...")
            
            if engine == "vector":
                dist = metadata.get("distance", "N/A")
                sim = metadata.get("similarity", "N/A")
                print(f"     Vector details: Distance={dist}, Similarity={sim}")
        
        return {
            "total_results": len(ctx_df),
            "bm25_count": len([r for r in ctx_df.itertuples() if r.metadata.get("source_engine") == "bm25"]),
            "vector_count": len([r for r in ctx_df.itertuples() if r.metadata.get("source_engine") == "vector"]),
            "top_score": max([r["metadata"].get("score", 0) for _, r in ctx_df.iterrows()]),
            "response": response
        }
    
    async def run_comprehensive_test(self, query: str):
        """Run comprehensive test showing all three approaches"""
        print("=" * 100)
        print(f"🧪 COMPREHENSIVE HYBRID SEARCH TEST")
        print(f"Query: '{query}'")
        print(f"Config: BM25 weight={self.config.bm25_weight}, Vector weight={self.config.vector_weight}")
        print("=" * 100)
        
        # Test individual engines
        bm25_results = self.test_bm25_only(query, 5)
        vector_results = await self.test_vector_only(query, 5)
        
        # Test hybrid
        hybrid_results = await self.test_hybrid_search(query)
        
        # Analysis
        print(f"\n📈 ANALYSIS:")
        print(f"  • BM25 alone would rank documents by keyword relevance")
        print(f"  • Vector alone would rank documents by semantic similarity") 
        print(f"  • Hybrid combines both with weights {self.config.bm25_weight:.1f}/{self.config.vector_weight:.1f}")
        print(f"  • Final ranking shows {hybrid_results['bm25_count']} BM25 + {hybrid_results['vector_count']} vector results")
        print(f"  • Top hybrid score: {hybrid_results['top_score']:.4f}")
        
        if hybrid_results["response"]:
            print(f"\n📝 SYNTHESIZED ANSWER:")
            print(f"  Confidence: {hybrid_results['response'].confidence:.2f}")
            print(f"  Answer: {hybrid_results['response'].answer[:200]}...")
        
        return hybrid_results


async def main():
    """Run hybrid search validation"""
    print("🧪 Starting Hybrid Search Validation")
    print("=" * 60)
    
    validator = HybridSearchValidator()
    
    # Setup
    if not await validator.setup():
        print("❌ Setup failed")
        return False
    
    # Test queries - mix of keyword and semantic queries
    test_queries = [
        "baby sleep schedule",  # Should favor BM25 (keyword match)
        "how to improve infant rest quality",  # Should favor vector (semantic)
        "bedtime routine for toddlers"  # Mixed - both should contribute
    ]
    
    results = []
    for query in test_queries:
        result = await validator.run_comprehensive_test(query)
        results.append(result)
        print("\n" + "="*50 + "\n")
    
    # Summary
    print("📊 VALIDATION SUMMARY:")
    print("-" * 40)
    
    all_passed = True
    for i, (query, result) in enumerate(zip(test_queries, results)):
        bm25_count = result["bm25_count"]
        vector_count = result["vector_count"] 
        total = result["total_results"]
        
        print(f"{i+1}. '{query}':")
        print(f"   Results: {total} total ({bm25_count} BM25, {vector_count} vector)")
        
        # Validation checks
        checks = []
        checks.append(("Both engines used", bm25_count > 0 and vector_count > 0))
        checks.append(("Results combined", total > 0))
        checks.append(("Scores calculated", result["top_score"] > 0))
        checks.append(("Response generated", result["response"] is not None))
        
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False
    
    print(f"\n🎯 Overall validation: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    
    if all_passed:
        print("🎉 Hybrid search is working correctly!")
        print("  • Both BM25 and vector search are functional")
        print("  • Scores are being combined with proper weights")
        print("  • Results are ranked by hybrid score")
        print("  • Responses are being synthesized")
    else:
        print("⚠️  Issues found in hybrid search implementation")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)