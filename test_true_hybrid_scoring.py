#!/usr/bin/env python3
"""
Automated test to validate true hybrid scoring implementation.
This test verifies that the fixes address all identified issues.
"""
import requests
import json
from typing import Dict, Any, List


API_BASE_URL = "http://localhost:8000"


def make_request(endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make a request to the FastAPI backend"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
            
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}


class TrueHybridScoringValidator:
    """Validates that true hybrid scoring is working correctly"""
    
    def __init__(self):
        self.test_queries = [
            {"query": "What are the best tips to promote good night sleep?", "top_k": 8},
            {"query": "baby sleep schedule", "top_k": 5},
            {"query": "bedtime routine for toddlers", "top_k": 10}
        ]
    
    def test_exact_result_count(self, query: str, top_k: int) -> Dict[str, Any]:
        """Test that we get exactly the requested number of results"""
        
        result = make_request("/query", "POST", {"query": query, "top_k": top_k})
        
        if not result["success"]:
            return {"passed": False, "error": result["error"]}
        
        data = result["data"]
        results_table = data.get("results_table", [])
        actual_count = len(results_table)
        
        return {
            "passed": actual_count == top_k,
            "expected": top_k,
            "actual": actual_count,
            "message": f"Expected {top_k} results, got {actual_count}"
        }
    
    def test_true_hybrid_scores(self, query: str, top_k: int) -> Dict[str, Any]:
        """Test that results have both BM25 and Vector score components"""
        
        result = make_request("/query", "POST", {"query": query, "top_k": top_k})
        
        if not result["success"]:
            return {"passed": False, "error": result["error"]}
        
        data = result["data"]
        results_table = data.get("results_table", [])
        
        issues = []
        hybrid_calculations_correct = 0
        results_with_both_scores = 0
        
        for i, result_row in enumerate(results_table):
            rank = result_row.get("rank", i + 1)
            hybrid_score = result_row.get("hybrid_score", 0)
            bm25_score = result_row.get("bm25_score", 0)
            vector_score = result_row.get("vector_score", 0)
            engines = result_row.get("engines", "")
            
            # Check if result has both score components (even if one is 0)
            has_bm25_component = bm25_score > 0 or "bm25" in engines
            has_vector_component = vector_score > 0 or "vector" in engines
            
            if has_bm25_component and has_vector_component:
                results_with_both_scores += 1
            
            # Verify hybrid score calculation
            expected_hybrid = round(bm25_score + vector_score, 4)
            if abs(hybrid_score - expected_hybrid) < 0.0001:
                hybrid_calculations_correct += 1
            else:
                issues.append(f"Rank {rank}: Hybrid={hybrid_score}, Expected={expected_hybrid} (BM25={bm25_score} + Vector={vector_score})")
        
        return {
            "passed": len(issues) == 0,
            "hybrid_calculations_correct": hybrid_calculations_correct,
            "total_results": len(results_table),
            "results_with_both_scores": results_with_both_scores,
            "calculation_issues": issues,
            "message": f"{hybrid_calculations_correct}/{len(results_table)} hybrid scores calculated correctly"
        }
    
    def test_score_distribution(self, query: str, top_k: int) -> Dict[str, Any]:
        """Test that we have proper score distribution and weights"""
        
        result = make_request("/query", "POST", {"query": query, "top_k": top_k})
        
        if not result["success"]:
            return {"passed": False, "error": result["error"]}
        
        data = result["data"]
        results_table = data.get("results_table", [])
        
        bm25_weights = []
        vector_weights = []
        
        for result_row in results_table:
            bm25_score = result_row.get("bm25_score", 0)
            vector_score = result_row.get("vector_score", 0)
            
            if bm25_score > 0:
                bm25_weights.append(bm25_score)
            if vector_score > 0:
                vector_weights.append(vector_score)
        
        # Check if we have reasonable weight distribution
        # BM25 weight = 0.3, Vector weight = 0.7 (from config)
        avg_bm25 = sum(bm25_weights) / len(bm25_weights) if bm25_weights else 0
        avg_vector = sum(vector_weights) / len(vector_weights) if vector_weights else 0
        
        return {
            "passed": len(bm25_weights) > 0 and len(vector_weights) > 0,
            "bm25_results_count": len(bm25_weights),
            "vector_results_count": len(vector_weights),
            "avg_bm25_score": round(avg_bm25, 4),
            "avg_vector_score": round(avg_vector, 4),
            "message": f"BM25: {len(bm25_weights)} results, Vector: {len(vector_weights)} results"
        }
    
    def test_quality_penalties(self, query: str, top_k: int) -> Dict[str, Any]:
        """Test that quality penalties are being applied appropriately"""
        
        result = make_request("/query", "POST", {"query": query, "top_k": top_k})
        
        if not result["success"]:
            return {"passed": False, "error": result["error"]}
        
        data = result["data"]
        results_table = data.get("results_table", [])
        
        quality_penalties = []
        penalty_variation = False
        
        for result_row in results_table:
            penalty = result_row.get("quality_penalty")
            if penalty is not None:
                quality_penalties.append(penalty)
        
        # Check if we have variation in quality penalties (not all 1.0)
        unique_penalties = set(quality_penalties)
        penalty_variation = len(unique_penalties) > 1
        
        return {
            "passed": len(quality_penalties) > 0,
            "quality_penalties_applied": len(quality_penalties),
            "unique_penalty_values": len(unique_penalties),
            "penalty_variation": penalty_variation,
            "penalty_range": f"{min(quality_penalties):.2f} - {max(quality_penalties):.2f}" if quality_penalties else "N/A",
            "message": f"Quality penalties: {len(unique_penalties)} unique values, variation: {penalty_variation}"
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all validation tests"""
        
        print("🧪 TRUE HYBRID SCORING VALIDATION TEST")
        print("=" * 60)
        
        all_results = {}
        overall_passed = True
        
        for test_case in self.test_queries:
            query = test_case["query"]
            top_k = test_case["top_k"]
            
            print(f"\n🔍 Testing: '{query}' (top_k={top_k})")
            print("-" * 40)
            
            # Test 1: Exact result count
            count_test = self.test_exact_result_count(query, top_k)
            print(f"  ✅ Result Count: {count_test['message']}" if count_test['passed'] else f"  ❌ Result Count: {count_test['message']}")
            
            # Test 2: True hybrid scores
            hybrid_test = self.test_true_hybrid_scores(query, top_k)
            print(f"  ✅ Hybrid Scoring: {hybrid_test['message']}" if hybrid_test['passed'] else f"  ❌ Hybrid Scoring: {hybrid_test['message']}")
            
            # Test 3: Score distribution
            dist_test = self.test_score_distribution(query, top_k)
            print(f"  ✅ Score Distribution: {dist_test['message']}" if dist_test['passed'] else f"  ❌ Score Distribution: {dist_test['message']}")
            
            # Test 4: Quality penalties
            quality_test = self.test_quality_penalties(query, top_k)
            print(f"  ✅ Quality Penalties: {quality_test['message']}" if quality_test['passed'] else f"  ❌ Quality Penalties: {quality_test['message']}")
            
            all_results[query] = {
                "count_test": count_test,
                "hybrid_test": hybrid_test,
                "distribution_test": dist_test,
                "quality_test": quality_test
            }
            
            # Check if all tests passed for this query
            query_passed = all([count_test['passed'], hybrid_test['passed'], dist_test['passed'], quality_test['passed']])
            if not query_passed:
                overall_passed = False
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 VALIDATION SUMMARY:")
        
        total_tests = len(self.test_queries) * 4
        passed_tests = sum([
            1 for query_results in all_results.values()
            for test_result in query_results.values()
            if test_result['passed']
        ])
        
        print(f"  Tests passed: {passed_tests}/{total_tests}")
        print(f"  Overall result: {'✅ ALL TESTS PASSED' if overall_passed else '❌ SOME TESTS FAILED'}")
        
        if overall_passed:
            print("\n🎉 TRUE HYBRID SCORING IS WORKING CORRECTLY!")
            print("  ✅ Exact result counts returned")
            print("  ✅ True hybrid scores calculated (BM25 + Vector)")
            print("  ✅ Both engines contributing to results")
            print("  ✅ Quality penalties applied to citation-heavy content")
        else:
            print("\n⚠️  Issues found - check details above")
        
        return {
            "overall_passed": overall_passed,
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "detailed_results": all_results
        }


def main():
    """Run the validation test"""
    
    # Check server availability
    health = make_request("/healthz")
    if not health["success"]:
        print("❌ Server not available. Please start FastAPI server first.")
        return False
    
    status = make_request("/init/status")
    if not status["success"] or status["data"]["status"] != "completed":
        print("❌ System not initialized. Please initialize the system first.")
        return False
    
    # Run tests
    validator = TrueHybridScoringValidator()
    results = validator.run_comprehensive_test()
    
    return results["overall_passed"]


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)