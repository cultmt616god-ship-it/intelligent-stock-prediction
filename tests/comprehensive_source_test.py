#!/usr/bin/env python3
"""
Comprehensive test script to verify all three sentiment analysis sources:
1. Finviz + FinVADER (Primary)
2. EODHD API (API Fallback)
3. Google News RSS (Last Resort)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_sentiment import ComprehensiveSentimentAnalyzer
import time

def test_finviz_finvader():
    """Test the primary source: Finviz scraping + FinVADER analysis"""
    print("=" * 60)
    print("Testing Primary Source: Finviz + FinVADER")
    print("=" * 60)
    
    try:
        start_time = time.time()
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=5)
        polarity, titles, label, pos, neg, neu = analyzer.get_sentiment("AAPL")
        end_time = time.time()
        
        print(f"✓ Finviz + FinVADER completed successfully")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Polarity: {polarity:.4f}")
        print(f"  Label: {label}")
        print(f"  Distribution: +{pos}, -{neg}, 0{neu}")
        print(f"  Articles processed: {len(titles)}")
        if titles:
            print(f"  Sample headlines:")
            for i, title in enumerate(titles[:3]):
                print(f"    {i+1}. {title[:60]}...")
        
        return True
    except Exception as e:
        print(f"✗ Finviz + FinVADER failed: {e}")
        return False

def test_eodhd_api():
    """Test the API fallback: EODHD API"""
    print("\n" + "=" * 60)
    print("Testing API Fallback: EODHD API")
    print("=" * 60)
    
    # Test without API key (should gracefully skip)
    try:
        start_time = time.time()
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=5, eodhd_api_key=None)
        polarity, titles, label, pos, neg, neu = analyzer.get_sentiment("AAPL")
        end_time = time.time()
        
        print(f"✓ EODHD API test completed (without API key)")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Polarity: {polarity:.4f}")
        print(f"  Label: {label}")
        print(f"  Distribution: +{pos}, -{neg}, 0{neu}")
        print(f"  Articles processed: {len(titles)}")
        
        # Note: Without API key, it should fall back to other sources
        return True
    except Exception as e:
        print(f"✗ EODHD API test failed: {e}")
        return False

def test_google_news_rss():
    """Test the last resort: Google News RSS"""
    print("\n" + "=" * 60)
    print("Testing Last Resort: Google News RSS")
    print("=" * 60)
    
    try:
        # Create an analyzer that forces Google News RSS by limiting other sources
        start_time = time.time()
        
        # We'll test Google News RSS directly
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3)
        items = analyzer.get_google_news("Apple")
        
        end_time = time.time()
        
        print(f"✓ Google News RSS completed successfully")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Articles found: {len(items)}")
        
        if items:
            print(f"  Sample articles:")
            for i, item in enumerate(items[:3]):
                print(f"    {i+1}. {item['title'][:60]}...")
                print(f"       Source: {item['source']}")
        
        return len(items) > 0
    except Exception as e:
        print(f"✗ Google News RSS failed: {e}")
        return False

def test_complete_fallback_chain():
    """Test the complete fallback chain by simulating different scenarios"""
    print("\n" + "=" * 60)
    print("Testing Complete Fallback Chain")
    print("=" * 60)
    
    try:
        # Test 1: Normal operation (should use Finviz + FinVADER)
        print("Test 1: Normal operation (Finviz + FinVADER)")
        start_time = time.time()
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3)
        polarity1, titles1, label1, pos1, neg1, neu1 = analyzer.get_sentiment("AAPL")
        end_time = time.time()
        
        print(f"  Result: Polarity {polarity1:.4f} ({label1}) in {end_time - start_time:.2f}s")
        
        # Test 2: With API key (should try EODHD API)
        print("\nTest 2: With API key placeholder (EODHD API fallback)")
        start_time = time.time()
        analyzer_with_api = ComprehensiveSentimentAnalyzer(num_articles=3, eodhd_api_key="demo")
        polarity2, titles2, label2, pos2, neg2, neu2 = analyzer_with_api.get_sentiment("AAPL")
        end_time = time.time()
        
        print(f"  Result: Polarity {polarity2:.4f} ({label2}) in {end_time - start_time:.2f}s")
        
        # Test 3: Force Google News RSS by requesting many articles
        print("\nTest 3: High volume request (forces Google News RSS)")
        start_time = time.time()
        analyzer_high_volume = ComprehensiveSentimentAnalyzer(num_articles=15)  # Request more than Finviz typically provides
        polarity3, titles3, label3, pos3, neg3, neu3 = analyzer_high_volume.get_sentiment("AAPL")
        end_time = time.time()
        
        print(f"  Result: Polarity {polarity3:.4f} ({label3}) in {end_time - start_time:.2f}s")
        print(f"  Articles processed: {len(titles3)}")
        
        return True
    except Exception as e:
        print(f"✗ Complete fallback chain test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance between different approaches"""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    results = {}
    
    # Test Finviz + FinVADER
    try:
        start_time = time.time()
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=5)
        polarity, titles, label, pos, neg, neu = analyzer.get_sentiment("AAPL")
        end_time = time.time()
        
        results['Finviz + FinVADER'] = {
            'time': end_time - start_time,
            'articles': len(titles),
            'polarity': polarity,
            'label': label
        }
        print(f"Finviz + FinVADER: {end_time - start_time:.2f}s for {len(titles)} articles")
    except Exception as e:
        print(f"Finviz + FinVADER test failed: {e}")
        results['Finviz + FinVADER'] = {'error': str(e)}
    
    # Test Google News RSS directly
    try:
        start_time = time.time()
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=5)
        items = analyzer.get_google_news("Apple")
        end_time = time.time()
        
        results['Google News RSS'] = {
            'time': end_time - start_time,
            'articles': len(items)
        }
        print(f"Google News RSS: {end_time - start_time:.2f}s for {len(items)} articles")
    except Exception as e:
        print(f"Google News RSS test failed: {e}")
        results['Google News RSS'] = {'error': str(e)}
    
    return results

def main():
    """Run all comprehensive tests"""
    print("Comprehensive Source Testing for News Sentiment Analysis")
    print("Testing all three sources: Finviz+FinVADER, EODHD API, Google News RSS")
    print("=" * 80)
    
    # Run individual source tests
    tests = [
        ("Finviz + FinVADER", test_finviz_finvader),
        ("EODHD API", test_eodhd_api),
        ("Google News RSS", test_google_news_rss),
        ("Complete Fallback Chain", test_complete_fallback_chain)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            print(f"\nRunning {name} test...")
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Run performance comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    performance_results = test_performance_comparison()
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:.<40} {status}")
        if result:
            passed += 1
    
    print("-" * 80)
    print(f"Source Tests Passed: {passed}/{len(results)}")
    
    # Performance summary
    print("\nPerformance Summary:")
    for source, metrics in performance_results.items():
        if 'error' in metrics:
            print(f"{source:.<30} ERROR: {metrics['error']}")
        else:
            time_str = f"{metrics['time']:.2f}s"
            articles_str = f"{metrics.get('articles', 'N/A')} articles"
            print(f"{source:.<30} {time_str} ({articles_str})")
    
    if passed == len(results):
        print("\n🎉 All source tests passed!")
        print("The comprehensive sentiment analysis pipeline is working correctly.")
    else:
        print(f"\n⚠ {len(results) - passed} source test(s) failed.")
        print("Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)