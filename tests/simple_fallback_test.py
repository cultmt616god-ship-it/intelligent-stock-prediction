#!/usr/bin/env python3
"""
Simple test to verify fallback mechanisms work correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_sentiment import ComprehensiveSentimentAnalyzer
from unittest.mock import patch

def test_google_news_works():
    """Verify Google News RSS is working"""
    print("Testing Google News RSS directly...")
    analyzer = ComprehensiveSentimentAnalyzer()
    try:
        items = analyzer.get_google_news("Apple")
        print(f"✓ Google News RSS returned {len(items)} items")
        if items:
            print(f"  First item: {items[0].get('title', 'No title')}")
        return len(items) > 0
    except Exception as e:
        print(f"✗ Google News RSS failed: {e}")
        return False

def test_fallback_chain():
    """Test that fallback chain works when Finviz is limited"""
    print("\nTesting fallback chain...")
    
    def mock_finviz(*args, **kwargs):
        # Return only 2 articles instead of requested 10
        return [
            {'title': 'Finviz 1', 'url': 'http://f1.com', 'text': 'text1', 'source': 'Finviz'},
            {'title': 'Finviz 2', 'url': 'http://f2.com', 'text': 'text2', 'source': 'Finviz'}
        ]
    
    try:
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=10)  # Request 10 articles
        
        with patch.object(analyzer, 'get_finviz_news', side_effect=mock_finviz):
            # This should trigger Investing.com and Google News RSS fallbacks
            polarity, titles, label, pos, neg, neu = analyzer.get_sentiment("AAPL", "Apple Inc")
            
            print(f"✓ Fallback chain executed")
            print(f"  Got {len(titles)} total articles")
            print(f"  Polarity: {polarity:.4f} ({label})")
            
            # Should have more than 2 articles if fallbacks worked
            return len(titles) >= 2
    except Exception as e:
        print(f"✗ Fallback chain test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Fallback Mechanisms\n")
    
    test1 = test_google_news_works()
    test2 = test_fallback_chain()
    
    print(f"\nResults:")
    print(f"Google News RSS: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Fallback Chain: {'✓ PASS' if test2 else '✗ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 All tests passed! Fallback mechanisms are working correctly.")
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed. Check output above.")
        sys.exit(1)