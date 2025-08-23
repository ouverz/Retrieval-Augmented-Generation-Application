#!/usr/bin/env python3
"""
Validation script to test the LLM factory and synthesizer fixes.
This script tests that the pydantic-ai integration works correctly.
"""
import asyncio
import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from services.llm_factory import LLMFactory
    from services.synthesizer import synthesize_answer, SynthesizedResponse
    print("✅ Successfully imported LLMFactory and synthesizer")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


async def test_llm_factory():
    """Test LLM factory basic functionality"""
    print("\n🔧 Testing LLM Factory...")
    
    try:
        factory = LLMFactory()
        print(f"✅ LLMFactory created with provider: {factory.provider}")
        
        # Test building agent
        agent = factory.build_agent(
            system_prompt="You are a helpful assistant."
        )
        print("✅ Agent built successfully with result_type")
        
        # Test run_structured method
        result = await factory.run_structured(
            result_type=SynthesizedResponse,
            system_prompt="You are a helpful assistant that answers questions about sleep.",
            user_message="What is good sleep hygiene? Provide a brief answer.",
        )
        print(f"✅ run_structured completed, result type: {type(result)}")
        print(f"✅ Result is SynthesizedResponse: {isinstance(result, SynthesizedResponse)}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLMFactory test failed: {e}")
        return False


async def test_synthesizer():
    """Test synthesizer with mock context data"""
    print("\n📝 Testing Synthesizer...")
    
    try:
        # Create mock context DataFrame matching HybridSearchEngine output
        mock_context = pd.DataFrame([
            {
                "id": "doc1_chunk1", 
                "content": "Good sleep hygiene includes maintaining a consistent sleep schedule.",
                "metadata": {"source": "sleep_guide.pdf", "page": 1}
            },
            {
                "id": "doc1_chunk2",
                "content": "Avoid caffeine and screens before bedtime for better sleep quality.", 
                "metadata": {"source": "sleep_guide.pdf", "page": 2}
            }
        ])
        
        factory = LLMFactory()
        
        result = await synthesize_answer(
            query="What are some good sleep hygiene practices?",
            context=mock_context,
            factory=factory,
            max_attempts=2
        )
        
        print(f"✅ Synthesizer completed, result type: {type(result)}")
        print(f"✅ Result is SynthesizedResponse: {isinstance(result, SynthesizedResponse)}")
        print(f"✅ Answer: {result.answer[:100]}...")
        print(f"✅ Confidence: {result.confidence}")
        print(f"✅ Enough context: {result.enough_context}")
        
        return True
        
    except Exception as e:
        print(f"❌ Synthesizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling():
    """Test error handling and retries"""
    print("\n🛡️ Testing Error Handling...")
    
    try:
        # Test with empty context (should not fail, just return low confidence)
        empty_context = pd.DataFrame(columns=["id", "content", "metadata"])
        
        factory = LLMFactory()
        
        result = await synthesize_answer(
            query="What is baby sleep?",
            context=empty_context,
            factory=factory,
            max_attempts=1
        )
        
        print(f"✅ Empty context handled, result type: {type(result)}")
        print(f"✅ Answer for empty context: {result.answer}")
        print(f"✅ Confidence for empty context: {result.confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def main():
    """Run all validation tests"""
    print("🧪 Starting LLM Factory and Synthesizer Validation Tests")
    print("=" * 60)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Tests may fail without valid API key")
    
    tests = [
        ("LLM Factory", test_llm_factory),
        ("Synthesizer", test_synthesizer), 
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🎯 Running {test_name} test...")
            result = await test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test CRASHED: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The LLM integration fix is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)