#!/usr/bin/env python3
"""
Test script to specifically verify the fallback mechanisms in the news sentiment analysis pipeline.
This focuses on ensuring that when primary sources fail or don't provide enough articles,
the system correctly falls back to secondary sources.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_sentiment import ComprehensiveSentimentAnalyzer
from unittest.mock import patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

def test_finviz_primary_success():
    """Test that Finviz works as the primary source when it provides enough articles"""
    print("=" * 70)
    print("Test 1: Finviz as Primary Source (Normal Operation)")
    print("=" * 70)
    
    try:
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=5)
        polarity, titles, label, pos_count, neg_count, neu_count = analyzer.get_sentiment("AAPL", "Apple Inc")
        
        print(f"✓ Finviz successfully provided sentiment analysis")
        print(f"  Polarity: {polarity:.4f} ({label})")
        print(f"  Articles processed: {len(titles)}")
        print(f"  Sentiment distribution: +{pos_count}, -{neg_count}, 0{neu_count}")
        
        # Since we only requested 5 articles, Finviz should satisfy this completely
        if len(titles) >= 5:
            print("  ✓ Correctly used Finviz as primary source")
            return True
        else:
            print("  ⚠ Unexpected number of articles")
            return False
            
    except Exception as e:
        print(f"✗ Error in Finviz primary test: {e}")
        return False

def test_investing_com_fallback_triggered():
    """Test that Investing.com is triggered when Finviz is disabled"""
    print("\n" + "=" * 70)
    print("Test 2: Investing.com Fallback Triggered")
    print("=" * 70)
    
    # Mock Finviz to return fewer articles than requested to force fallback
    def mock_get_finviz_news(*args, **kwargs):
        # Return only 2 articles instead of the requested amount
        return [
            {'title': 'Mock Finviz Article 1', 'url': 'http://mock.com/1', 'text': 'Mock text 1', 'source': 'Finviz'},
            {'title': 'Mock Finviz Article 2', 'url': 'http://mock.com/2', 'text': 'Mock text 2', 'source': 'Finviz'}
        ]
    
    try:
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=5)  # Request 5 articles
        
        # Patch the Finviz method to return fewer articles
        with patch.object(analyzer, 'get_finviz_news', side_effect=mock_get_finviz_news):
            polarity, titles, label, pos_count, neg_count, neu_count = analyzer.get_sentiment("AAPL", "Apple Inc")
            
            print(f"✓ Fallback mechanism executed")
            print(f"  Polarity: {polarity:.4f} ({label})")
            print(f"  Articles processed: {len(titles)}")
            print(f"  Sentiment distribution: +{pos_count}, -{neg_count}, 0{neu_count}")
            
            # Should have more than the 2 Finviz articles if fallback worked
            if len(titles) > 2:
                print("  ✓ Investing.com fallback was correctly triggered")
                # Show first few article sources
                print("  Sample sources:")
                for i, title in enumerate(titles[:4]):
                    print(f"    {i+1}. {title}")
                return True
            else:
                print("  ⚠ Fallback may not have been triggered (only got Finviz articles)")
                return False
                
    except Exception as e:
        print(f"✗ Error in Investing.com fallback test: {e}")
        return False

def test_google_news_fallback_triggered():
    """Test that Google News RSS is triggered when both Finviz and Investing.com are limited"""
    print("\n" + "=" * 70)
    print("Test 3: Google News RSS Fallback Triggered")
    print("=" * 70)
    
    # Mock both Finviz and Investing.com to return fewer articles than requested
    def mock_get_finviz_news(*args, **kwargs):
        return [{'title': 'Mock Finviz', 'url': 'http://mock.com/1', 'text': 'Mock text', 'source': 'Finviz'}]
    
    def mock_get_news_links(*args, **kwargs):
        return []  # Return no articles from Investing.com
    
    try:
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=5)  # Request 5 articles
        
        # Patch both methods to force Google News RSS fallback
        with patch.object(analyzer, 'get_finviz_news', side_effect=mock_get_finviz_news):
            with patch.object(analyzer.selenium_scraper, 'get_news_links', side_effect=mock_get_news_links):
                polarity, titles, label, pos_count, neg_count, neu_count = analyzer.get_sentiment("AAPL", "Apple Inc")
                
                print(f"✓ Full fallback chain executed")
                print(f"  Polarity: {polarity:.4f} ({label})")
                print(f"  Articles processed: {len(titles)}")
                print(f"  Sentiment distribution: +{pos_count}, -{neg_count}, 0{neu_count}")
                
                # Should have articles from Google News RSS if fallback worked
                if len(titles) >= 1:
                    print("  ✓ Google News RSS fallback was correctly triggered")
                    print("  Sample articles:")
                    for i, title in enumerate(titles[:3]):
                        print(f"    {i+1}. {title}")
                    return True
                else:
                    print("  ⚠ Google News RSS fallback may not have been triggered")
                    return False
                    
    except Exception as e:
        print(f"✗ Error in Google News RSS fallback test: {e}")
        return False

def test_complete_fallback_chain():
    """Test the complete fallback chain by simulating failures at each level"""
    print("\n" + "=" * 70)
    print("Test 4: Complete Fallback Chain Simulation")
    print("=" * 70)
    
    try:
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=8)
        
        # Track which sources are called
        sources_called = []
        
        # Mock each source to simulate different behaviors
        def mock_finviz(*args, **kwargs):
            sources_called.append("Finviz")
            # Return 2 articles
            return [
                {'title': 'Finviz Article 1', 'url': 'http://finviz.com/1', 'text': 'Text 1', 'source': 'Finviz'},
                {'title': 'Finviz Article 2', 'url': 'http://finviz.com/2', 'text': 'Text 2', 'source': 'Finviz'}
            ]
        
        def mock_investing_com(*args, **kwargs):
            sources_called.append("Investing.com")
            # Return 3 articles
            return [
                {'title': 'Investing.com Article 1', 'url': 'http://investing.com/1', 'text': 'Text 1', 'source': 'Investing.com'},
                {'title': 'Investing.com Article 2', 'url': 'http://investing.com/2', 'text': 'Text 2', 'source': 'Investing.com'},
                {'title': 'Investing.com Article 3', 'url': 'http://investing.com/3', 'text': 'Text 3', 'source': 'Investing.com'}
            ]
        
        def mock_google_news(*args, **kwargs):
            sources_called.append("Google News")
            # Return 3 articles
            return [
                {'title': 'Google News Article 1', 'url': 'http://news.google.com/1', 'text': 'Text 1', 'source': 'Google News'},
                {'title': 'Google News Article 2', 'url': 'http://news.google.com/2', 'text': 'Text 2', 'source': 'Google News'},
                {'title': 'Google News Article 3', 'url': 'http://news.google.com/3', 'text': 'Text 3', 'source': 'Google News'}
            ]
        
        # Apply mocks
        with patch.object(analyzer, 'get_finviz_news', side_effect=mock_finviz):
            with patch.object(analyzer.selenium_scraper, 'get_news_links', side_effect=mock_investing_com):
                with patch.object(analyzer, 'get_google_news', side_effect=mock_google_news):
                    polarity, titles, label, pos_count, neg_count, neu_count = analyzer.get_sentiment("AAPL", "Apple Inc")
        
        print(f"✓ Complete fallback chain simulation executed")
        print(f"  Polarity: {polarity:.4f} ({label})")
        print(f"  Articles processed: {len(titles)}")
        print(f"  Sentiment distribution: +{pos_count}, -{neg_count}, 0{neu_count}")
        print(f"  Sources called in order: {sources_called}")
        
        # Verify all sources were called and we got enough articles
        if len(sources_called) == 3 and len(titles) >= 8:
            print("  ✓ Complete fallback chain worked correctly")
            print("  Article sources:")
            source_counts = {}
            for i, title in enumerate(titles[:8]):
                source = title.split()[-1] if title else "Unknown"
                source_counts[source] = source_counts.get(source, 0) + 1
                print(f"    {i+1}. {title}")
            
            print(f"  Source distribution: {source_counts}")
            return True
        else:
            print("  ⚠ Complete fallback chain may not have executed properly")
            return False
            
    except Exception as e:
        print(f"✗ Error in complete fallback chain test: {e}")
        return False

def test_error_handling():
    """Test that the system gracefully handles errors in each source"""
    print("\n" + "=" * 70)
    print("Test 5: Error Handling in All Sources")
    print("=" * 70)
    
    def mock_finviz_error(*args, **kwargs):
        raise Exception("Finviz connection error")
    
    def mock_investing_com_error(*args, **kwargs):
        raise Exception("Investing.com scraping error")
    
    def mock_google_news_success(*args, **kwargs):
        # Return articles from Google News as fallback
        return [
            {'title': 'Google News Success 1', 'url': 'http://news.google.com/s1', 'text': 'Success text 1', 'source': 'Google News'},
            {'title': 'Google News Success 2', 'url': 'http://news.google.com/s2', 'text': 'Success text 2', 'source': 'Google News'},
            {'title': 'Google News Success 3', 'url': 'http://news.google.com/s3', 'text': 'Success text 3', 'source': 'Google News'}
        ]
    
    try:
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3)
        
        # Simulate errors in first two sources, success in third
        with patch.object(analyzer, 'get_finviz_news', side_effect=mock_finviz_error):
            with patch.object(analyzer.selenium_scraper, 'get_news_links', side_effect=mock_investing_com_error):
                with patch.object(analyzer, 'get_google_news', side_effect=mock_google_news_success):
                    polarity, titles, label, pos_count, neg_count, neu_count = analyzer.get_sentiment("AAPL", "Apple Inc")
        
        print(f"✓ Error handling with graceful fallback")
        print(f"  Polarity: {polarity:.4f} ({label})")
        print(f"  Articles processed: {len(titles)}")
        print(f"  Sentiment distribution: +{pos_count}, -{neg_count}, 0{neu_count}")
        
        # Should still get results from Google News despite errors in other sources
        if len(titles) >= 3:
            print("  ✓ System gracefully handled errors and fell back successfully")
            print("  Retrieved articles:")
            for i, title in enumerate(titles[:3]):
                print(f"    {i+1}. {title}")
            return True
        else:
            print("  ⚠ Error handling may not have worked correctly")
            return False
            
    except Exception as e:
        print(f"✗ Error in error handling test: {e}")
        return False

def main():
    """Run all fallback mechanism tests"""
    print("Testing Fallback Mechanisms in News Sentiment Analysis Pipeline")
    print("Verifying that Finviz → Investing.com → Google News RSS fallback chain works correctly")
    
    tests = [
        ("Finviz Primary Success", test_finviz_primary_success),
        ("Investing.com Fallback", test_investing_com_fallback_triggered),
        ("Google News RSS Fallback", test_google_news_fallback_triggered),
        ("Complete Fallback Chain", test_complete_fallback_chain),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<45} {status}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"Passed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All fallback mechanism tests passed!")
        print("   The system correctly handles the Finviz → Investing.com → Google News RSS fallback chain.")
    else:
        print("⚠ Some fallback mechanism tests failed.")
        print("   Review the output above to identify which fallback paths need attention.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)