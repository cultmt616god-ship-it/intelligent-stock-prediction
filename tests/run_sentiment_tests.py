#!/usr/bin/env python3
"""
Test Runner Script for News Sentiment Analysis

This script provides multiple ways to run tests for the sentiment analysis system:
1. Quick smoke tests
2. Comprehensive test suite
3. Specific test categories
4. Performance benchmarks
"""

import sys
import os
import argparse
import unittest
import time
from unittest.mock import patch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_smoke_tests():
    """Run quick smoke tests to verify basic functionality"""
    print("Running Smoke Tests...")
    print("=" * 50)
    
    try:
        # Test basic imports
        from news_sentiment import (
            ComprehensiveSentimentAnalyzer,
            retrieving_news_polarity,
            finviz_finvader_sentiment,
            hft_sentiment
        )
        print("✓ All imports successful")
        
        # Test analyzer initialization
        analyzer = ComprehensiveSentimentAnalyzer()
        print("✓ Analyzer initialization successful")
        
        # Test enum availability
        from news_sentiment import SentimentSource, UseCase
        print("✓ Enums available")
        
        # Test basic function calls (mocked to avoid external dependencies)
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment') as mock_get_sentiment:
            mock_get_sentiment.return_value = (0.1, ['Test'], 'Neutral', 0, 0, 1)
            
            # Test main function
            result = retrieving_news_polarity('AAPL', 1)
            print("✓ Main sentiment function works")
            
            # Test source-specific function
            result = finviz_finvader_sentiment('AAPL', 1)
            print("✓ Source-specific function works")
            
            # Test use case function
            result = hft_sentiment('AAPL', 1)
            print("✓ Use case function works")
        
        print("\n🎉 All smoke tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmark tests"""
    print("Running Performance Benchmarks...")
    print("=" * 50)
    
    try:
        from news_sentiment import ComprehensiveSentimentAnalyzer
        import time
        
        analyzer = ComprehensiveSentimentAnalyzer()
        
        # Benchmark analyzer initialization
        start_time = time.time()
        for i in range(100):
            temp_analyzer = ComprehensiveSentimentAnalyzer()
        init_time = time.time() - start_time
        print(f"✓ Analyzer initialization: {init_time:.4f}s for 100 instances")
        
        # Benchmark source selection
        start_time = time.time()
        for i in range(1000):
            analyzer._should_use_source("test_source")
        selection_time = time.time() - start_time
        print(f"✓ Source selection: {selection_time:.4f}s for 1000 calls")
        
        # Benchmark cache key generation
        start_time = time.time()
        for i in range(1000):
            analyzer._get_cache_key(f"SYMBOL{i}", f"Text content {i}")
        cache_time = time.time() - start_time
        print(f"✓ Cache key generation: {cache_time:.4f}s for 1000 calls")
        
        print("\n📊 Performance benchmarks completed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def run_specific_test_category(category):
    """Run tests for a specific category"""
    print(f"Running tests for category: {category}")
    print("=" * 50)
    
    try:
        # Import the comprehensive test framework
        import test_comprehensive_framework as test_framework
        
        # Map categories to test classes
        category_map = {
            'core': test_framework.TestCoreFunctionality,
            'sources': test_framework.TestSourceSpecificFunctions,
            'usecases': test_framework.TestUseCaseConfigurations,
            'advanced': test_framework.TestAdvancedFeatures,
            'errors': test_framework.TestErrorHandlingAndMonitoring,
            'fallback': test_framework.TestFallbackMechanisms,
            'performance': test_framework.TestPerformanceOptimizations,
            'integration': test_framework.TestIntegrationScenarios,
            'edge': test_framework.TestEdgeCases
        }
        
        if category not in category_map:
            print(f"Unknown category: {category}")
            print("Available categories:", ", ".join(category_map.keys()))
            return False
        
        # Run tests for the specific category
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(category_map[category]))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Category test failed: {e}")
        return False

def main():
    """Main function to run selected tests"""
    parser = argparse.ArgumentParser(description='Run sentiment analysis tests')
    parser.add_argument('--smoke', action='store_true', help='Run smoke tests')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive test suite')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--category', type=str, help='Run tests for specific category')
    parser.add_argument('--list-categories', action='store_true', help='List available test categories')
    
    args = parser.parse_args()
    
    # Handle list categories option
    if args.list_categories:
        print("Available test categories:")
        print("  core        - Core functionality tests")
        print("  sources     - Source-specific tests")
        print("  usecases    - Use case configuration tests")
        print("  advanced    - Advanced feature tests")
        print("  errors      - Error handling tests")
        print("  fallback    - Fallback mechanism tests")
        print("  performance - Performance optimization tests")
        print("  integration - Integration scenario tests")
        print("  edge        - Edge case tests")
        return 0
    
    # Run selected tests
    results = []
    
    if args.smoke or not any([args.smoke, args.comprehensive, args.benchmark, args.category]):
        # Run smoke tests by default if no other option specified
        results.append(("Smoke Tests", run_smoke_tests()))
    
    if args.comprehensive:
        print("Running Comprehensive Test Suite...")
        print("=" * 50)
        try:
            import test_comprehensive_framework as test_framework
            success = test_framework.run_comprehensive_tests()
            results.append(("Comprehensive Tests", success))
        except Exception as e:
            print(f"❌ Failed to run comprehensive tests: {e}")
            results.append(("Comprehensive Tests", False))
    
    if args.benchmark:
        results.append(("Performance Benchmarks", run_performance_benchmark()))
    
    if args.category:
        results.append((f"Category: {args.category}", run_specific_test_category(args.category)))
    
    # Print final summary
    print("\n" + "=" * 50)
    print("TEST EXECUTION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:.<30} {status}")
        if not passed:
            all_passed = False
    
    print("-" * 50)
    if all_passed:
        print("🎉 All selected tests passed!")
    else:
        print("❌ Some tests failed. Please review output above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())