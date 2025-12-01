#!/usr/bin/env python3
"""
Test script to verify the new FinVADER implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_sentiment import ComprehensiveSentimentAnalyzer

def test_finvader_import():
    """Test that FinVADER can be imported"""
    print("Testing FinVADER import...")
    try:
        from finvader import finvader
        print("✓ FinVADER imported successfully")
        # Test with a sample text
        sample_text = "The company reported strong earnings and exceeded expectations."
        score = finvader(sample_text, use_sentibignomics=True, use_henry=True, indicator='compound')
        print(f"✓ Sample sentiment score: {score}")
        return True
    except ImportError:
        print("⚠ FinVADER not available - will use standard VADER")
        return False
    except Exception as e:
        print(f"✗ FinVADER test failed: {e}")
        return False

def test_finviz_scraping():
    """Test Finviz scraping"""
    print("\nTesting Finviz scraping...")
    try:
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3)
        items = analyzer.get_finviz_news("AAPL")
        print(f"✓ Finviz returned {len(items)} items")
        if items:
            print(f"  First item: {items[0]['title'][:50]}...")
        return len(items) > 0
    except Exception as e:
        print(f"✗ Finviz scraping failed: {e}")
        return False

def test_sentiment_analysis():
    """Test the complete sentiment analysis pipeline"""
    print("\nTesting complete sentiment analysis pipeline...")
    try:
        # Test with a small number of articles for speed
        polarity, titles, label, pos, neg, neu = ComprehensiveSentimentAnalyzer(3).get_sentiment("AAPL")
        print(f"✓ Sentiment analysis completed")
        print(f"  Polarity: {polarity:.4f}")
        print(f"  Label: {label}")
        print(f"  Distribution: +{pos}, -{neg}, 0{neu}")
        print(f"  Articles processed: {len(titles)}")
        return True
    except Exception as e:
        print(f"✗ Sentiment analysis failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing New FinVADER Implementation")
    print("=" * 40)
    
    tests = [
        ("FinVADER Import", test_finvader_import),
        ("Finviz Scraping", test_finviz_scraping),
        ("Sentiment Analysis", test_sentiment_analysis)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:.<25} {status}")
        if result:
            passed += 1
    
    print("-" * 40)
    print(f"Passed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed!")
        print("The new implementation with Finviz + FinVADER is working correctly.")
    else:
        print("\n⚠ Some tests failed.")
        print("Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)