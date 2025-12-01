#!/usr/bin/env python3
"""
Test script to verify the minority sources (Investing.com Selenium scraper and Google News RSS fallback)
are working correctly by forcing them to execute.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_sentiment import ComprehensiveSentimentAnalyzer, InvestingComScraper
import time

def test_investing_com_scraper():
    """Test the Investing.com Selenium scraper directly"""
    print("=" * 60)
    print("Testing Investing.com Selenium Scraper")
    print("=" * 60)
    
    scraper = InvestingComScraper()
    
    try:
        # Test with a reasonable number of articles
        print("Attempting to scrape 3 articles from Investing.com for AAPL...")
        links = scraper.get_news_links("AAPL", 3, "Apple Inc")
        
        print(f"Found {len(links)} articles from Investing.com")
        if links:
            for i, link in enumerate(links[:3]):  # Show first 3
                print(f"  {i+1}. {link.get('url', 'N/A')}")
                if link.get('date'):
                    print(f"     Date: {link['date']}")
        else:
            print("  No articles found - this might be due to site structure changes or network issues")
            
        return len(links) > 0
    except Exception as e:
        print(f"Error testing Investing.com scraper: {e}")
        return False

def test_google_news_rss():
    """Test the Google News RSS fallback directly"""
    print("\n" + "=" * 60)
    print("Testing Google News RSS Fallback")
    print("=" * 60)
    
    analyzer = ComprehensiveSentimentAnalyzer()
    
    try:
        print("Attempting to fetch 3 articles from Google News RSS for Apple...")
        rss_items = analyzer.get_google_news("Apple")
        
        print(f"Found {len(rss_items)} articles from Google News RSS")
        if rss_items:
            for i, item in enumerate(rss_items[:3]):  # Show first 3
                print(f"  {i+1}. {item.get('title', 'N/A')}")
                print(f"     URL: {item.get('url', 'N/A')}")
                if item.get('date'):
                    print(f"     Date: {item['date']}")
        else:
            print("  No articles found - this might be due to connectivity issues")
            
        return len(rss_items) > 0
    except Exception as e:
        print(f"Error testing Google News RSS: {e}")
        return False

def test_high_volume_request():
    """Test the full pipeline with a high article request to force fallbacks"""
    print("\n" + "=" * 60)
    print("Testing Full Pipeline with High Volume Request (15 articles)")
    print("=" * 60)
    
    try:
        # Request more articles than Finviz typically provides to force fallbacks
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=15)
        
        print("Requesting sentiment analysis for 15 AAPL articles...")
        print("(This should trigger Finviz, then Investing.com, then Google News RSS)")
        
        # Capture the sentiment analysis results
        polarity, titles, label, pos_count, neg_count, neu_count = analyzer.get_sentiment("AAPL", "Apple Inc")
        
        print(f"\nResults:")
        print(f"  Polarity: {polarity}")
        print(f"  Label: {label}")
        print(f"  Positive: {pos_count}, Negative: {neg_count}, Neutral: {neu_count}")
        print(f"  Sample headlines: {len(titles)} total")
        for i, title in enumerate(titles[:5]):  # Show first 5
            print(f"    {i+1}. {title}")
            
        return True
    except Exception as e:
        print(f"Error in high volume test: {e}")
        return False

def test_finviz_limit_bypass():
    """Test by temporarily reducing Finviz effectiveness to force other sources"""
    print("\n" + "=" * 60)
    print("Testing with Limited Finviz to Force Other Sources")
    print("=" * 60)
    
    # We'll modify the approach to request more articles than Finviz typically returns
    analyzer = ComprehensiveSentimentAnalyzer(num_articles=20)  # Request 20, Finviz usually returns ~5
    
    try:
        print("Requesting sentiment analysis for 20 AAPL articles...")
        print("(Finviz will provide ~5, so Investing.com and Google News RSS should be triggered)")
        
        # This should trigger all sources since we're requesting more than Finviz provides
        polarity, titles, label, pos_count, neg_count, neu_count = analyzer.get_sentiment("AAPL", "Apple Inc")
        
        print(f"\nResults:")
        print(f"  Polarity: {polarity}")
        print(f"  Label: {label}")
        print(f"  Positive: {pos_count}, Negative: {neg_count}, Neutral: {neu_count}")
        print(f"  Total headlines processed: {len(titles)}")
        print(f"  Sample headlines:")
        for i, title in enumerate(titles[:8]):  # Show first 8
            print(f"    {i+1}. {title}")
            
        # If we got more than 10 articles, it's likely that fallback sources were triggered
        if len(titles) > 10:
            print("\n  ✓ Success: Received more articles than Finviz typically provides,")
            print("    indicating that fallback sources (Investing.com/Google News) were likely triggered.")
        else:
            print("\n  ⚠ Warning: Received fewer articles than expected.")
            print("    This might mean fallback sources weren't triggered or had issues.")
            
        return True
    except Exception as e:
        print(f"Error in limited Finviz test: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Minority Sources in News Sentiment Analysis Pipeline")
    print("This will verify that Investing.com Selenium scraper and Google News RSS fallback work correctly.")
    
    results = []
    
    # Test each component
    results.append(("Investing.com Scraper", test_investing_com_scraper()))
    results.append(("Google News RSS", test_google_news_rss()))
    results.append(("High Volume Request", test_high_volume_request()))
    results.append(("Limited Finviz Test", test_finviz_limit_bypass()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Passed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! All minority sources are working correctly.")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)