#!/usr/bin/env python3
"""
Component verification script to test each source independently
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_sentiment import ComprehensiveSentimentAnalyzer, InvestingComScraper

def test_finviz():
    """Test Finviz component"""
    print("1. Testing Finviz component...")
    try:
        analyzer = ComprehensiveSentimentAnalyzer()
        items = analyzer.get_finviz_news("AAPL")
        print(f"   ✓ Finviz returned {len(items)} items")
        if items:
            print(f"   First item: {items[0].get('title', 'No title')[:50]}...")
        return True
    except Exception as e:
        print(f"   ✗ Finviz failed: {e}")
        return False

def test_google_news():
    """Test Google News RSS component"""
    print("2. Testing Google News RSS component...")
    try:
        analyzer = ComprehensiveSentimentAnalyzer()
        items = analyzer.get_google_news("Apple")
        print(f"   ✓ Google News RSS returned {len(items)} items")
        if items:
            print(f"   First item: {items[0].get('title', 'No title')[:50]}...")
        return True
    except Exception as e:
        print(f"   ✗ Google News RSS failed: {e}")
        return False

def test_investing_com_import():
    """Test that Investing.com scraper can be imported"""
    print("3. Testing Investing.com scraper import...")
    try:
        scraper = InvestingComScraper()
        print("   ✓ Investing.com scraper imported successfully")
        return True
    except Exception as e:
        print(f"   ✗ Investing.com scraper import failed: {e}")
        return False

def test_vader():
    """Test VADER sentiment analyzer"""
    print("4. Testing VADER sentiment analyzer...")
    try:
        analyzer = ComprehensiveSentimentAnalyzer()
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores("This is a great test sentence!")
        print(f"   ✓ VADER working correctly")
        print(f"   Sample scores: {scores}")
        return True
    except Exception as e:
        print(f"   ✗ VADER failed: {e}")
        return False

def test_full_pipeline():
    """Test the full pipeline with a small request"""
    print("5. Testing full pipeline...")
    try:
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3)
        polarity, titles, label, pos, neg, neu = analyzer.get_sentiment("AAPL", "Apple Inc")
        print(f"   ✓ Full pipeline executed successfully")
        print(f"   Results: {len(titles)} articles, polarity {polarity:.4f} ({label})")
        return True
    except Exception as e:
        print(f"   ✗ Full pipeline failed: {e}")
        return False

def main():
    """Run all component tests"""
    print("Component Verification for News Sentiment Analysis Pipeline")
    print("=" * 60)
    
    tests = [
        test_finviz,
        test_google_news,
        test_investing_com_import,
        test_vader,
        test_full_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ✗ Test failed with exception: {e}")
            results.append(False)
        print()  # Add spacing
    
    # Summary
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    component_names = [
        "Finviz Component",
        "Google News RSS",
        "Investing.com Import",
        "VADER Sentiment",
        "Full Pipeline"
    ]
    
    for i, (name, result) in enumerate(zip(component_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<25} {status}")
    
    print("-" * 60)
    print(f"Passed: {passed}/{total} components")
    
    if passed == total:
        print("\n🎉 All components are working correctly!")
        print("The news sentiment analysis pipeline is fully functional.")
    else:
        print(f"\n⚠ {total-passed} component(s) failed.")
        print("Some parts of the pipeline may need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)