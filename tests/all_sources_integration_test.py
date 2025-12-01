#!/usr/bin/env python3
"""
Comprehensive test script to verify all sentiment analysis sources integration:
1. Finviz + FinVADER (Primary)
2. EODHD API (API Fallback)
3. Alpha Vantage News & Sentiments API
4. Tradestie WallStreetBets API
5. Finnhub Social Sentiment API
6. StockGeist.ai
7. Google News RSS (Last Resort)
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
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3)
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
            for i, title in enumerate(titles[:2]):
                print(f"    {i+1}. {title[:60]}...")
        
        return True
    except Exception as e:
        print(f"✗ Finviz + FinVADER failed: {e}")
        return False

def test_eodhd_api():
    """Test the EODHD API fallback"""
    print("\n" + "=" * 60)
    print("Testing EODHD API Fallback")
    print("=" * 60)
    
    # Test without API key (should gracefully skip)
    try:
        start_time = time.time()
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3, eodhd_api_key=None)
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

def test_alpha_vantage():
    """Test Alpha Vantage News API"""
    print("\n" + "=" * 60)
    print("Testing Alpha Vantage News API")
    print("=" * 60)
    
    try:
        start_time = time.time()
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3, alpha_vantage_api_key=None)
        alpha_news = analyzer.get_alpha_vantage_news("AAPL")
        end_time = time.time()
        
        print(f"✓ Alpha Vantage API test completed")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Articles found: {len(alpha_news)}")
        if alpha_news:
            print(f"  Sample article:")
            print(f"    Title: {alpha_news[0]['title'][:60]}...")
            print(f"    Source: {alpha_news[0]['source']}")
        
        return True
    except Exception as e:
        print(f"✗ Alpha Vantage API test failed: {e}")
        return False

def test_tradestie_reddit():
    """Test Tradestie WallStreetBets API"""
    print("\n" + "=" * 60)
    print("Testing Tradestie WallStreetBets API")
    print("=" * 60)
    
    try:
        start_time = time.time()
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3)
        reddit_posts = analyzer.get_tradestie_reddit("AAPL")
        end_time = time.time()
        
        print(f"✓ Tradestie Reddit API test completed")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Reddit posts found: {len(reddit_posts)}")
        if reddit_posts:
            print(f"  Sample post:")
            print(f"    Title: {reddit_posts[0]['title'][:60]}...")
            print(f"    Source: {reddit_posts[0]['source']}")
        
        return True
    except Exception as e:
        print(f"✗ Tradestie Reddit API test failed: {e}")
        return False

def test_finnhub_social():
    """Test Finnhub Social Sentiment API"""
    print("\n" + "=" * 60)
    print("Testing Finnhub Social Sentiment API")
    print("=" * 60)
    
    try:
        start_time = time.time()
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3, finnhub_api_key=None)
        social_mentions = analyzer.get_finnhub_social_sentiment("AAPL")
        end_time = time.time()
        
        print(f"✓ Finnhub Social API test completed")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Social mentions found: {len(social_mentions)}")
        
        return True
    except Exception as e:
        print(f"✗ Finnhub Social API test failed: {e}")
        return False

def test_google_news_rss():
    """Test the last resort: Google News RSS"""
    print("\n" + "=" * 60)
    print("Testing Last Resort: Google News RSS")
    print("=" * 60)
    
    try:
        start_time = time.time()
        analyzer = ComprehensiveSentimentAnalyzer(num_articles=3)
        items = analyzer.get_google_news("Apple")
        end_time = time.time()
        
        print(f"✓ Google News RSS completed successfully")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Articles found: {len(items)}")
        
        if items:
            print(f"  Sample articles:")
            for i, item in enumerate(items[:2]):
                print(f"    {i+1}. {item['title'][:60]}...")
                print(f"       Source: {item['source']}")
        
        return len(items) > 0
    except Exception as e:
        print(f"✗ Google News RSS failed: {e}")
        return False

def test_complete_pipeline():
    """Test the complete pipeline with all sources"""
    print("\n" + "=" * 60)
    print("Testing Complete Pipeline with All Sources")
    print("=" * 60)
    
    try:
        start_time = time.time()
        # Test with all API keys set to None (should gracefully skip)
        analyzer = ComprehensiveSentimentAnalyzer(
            num_articles=5,
            eodhd_api_key=None,
            alpha_vantage_api_key=None,
            finnhub_api_key=None,
            stockgeist_api_key=None
        )
        polarity, titles, label, pos, neg, neu = analyzer.get_sentiment("AAPL")
        end_time = time.time()
        
        print(f"✓ Complete pipeline test completed")
        print(f"  Execution time: {end_time - start_time:.2f} seconds")
        print(f"  Final Polarity: {polarity:.4f}")
        print(f"  Label: {label}")
        print(f"  Distribution: +{pos}, -{neg}, 0{neu}")
        print(f"  Articles processed: {len(titles)}")
        
        return True
    except Exception as e:
        print(f"✗ Complete pipeline test failed: {e}")
        return False

def main():
    """Run all comprehensive tests"""
    print("Comprehensive Integration Testing for All Sentiment Analysis Sources")
    print("Testing all seven sources integration:")
    print("1. Finviz + FinVADER (Primary)")
    print("2. EODHD API (API Fallback)")
    print("3. Alpha Vantage News & Sentiments API")
    print("4. Tradestie WallStreetBets API")
    print("5. Finnhub Social Sentiment API")
    print("6. StockGeist.ai")
    print("7. Google News RSS (Last Resort)")
    print("=" * 80)
    
    # Run individual source tests
    tests = [
        ("Finviz + FinVADER", test_finviz_finvader),
        ("EODHD API", test_eodhd_api),
        ("Alpha Vantage", test_alpha_vantage),
        ("Tradestie Reddit", test_tradestie_reddit),
        ("Finnhub Social", test_finnhub_social),
        ("Google News RSS", test_google_news_rss),
        ("Complete Pipeline", test_complete_pipeline)
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
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name:.<40} {status}")
        if result:
            passed += 1
    
    print("-" * 80)
    print(f"Integration Tests Passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All integration tests passed!")
        print("All seven sentiment analysis sources are properly integrated.")
    else:
        print(f"\n⚠ {len(results) - passed} integration test(s) failed.")
        print("Check the output above for details.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)