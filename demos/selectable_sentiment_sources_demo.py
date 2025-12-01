#!/usr/bin/env python3
"""
Demo script showing how to use selectable sentiment sources
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from news_sentiment import (
    retrieving_news_polarity, 
    SentimentSource,
    finviz_finvader_sentiment,
    eodhd_sentiment,
    alpha_vantage_sentiment,
    reddit_sentiment,
    social_sentiment,
    google_news_sentiment
)
import time

def demo_all_sources():
    """Demo using all sources (default behavior)"""
    print("=" * 60)
    print("DEMO 1: Using All Sentiment Sources (Default)")
    print("=" * 60)
    
    start_time = time.time()
    polarity, titles, label, pos, neg, neu = retrieving_news_polarity("AAPL", 3)
    end_time = time.time()
    
    print(f"Result: Polarity {polarity:.4f} ({label})")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Articles processed: {len(titles)}")
    print(f"Sentiment distribution: +{pos}/-{neg}/0{neu}")

def demo_single_source_finviz():
    """Demo using only Finviz + FinVADER"""
    print("\n" + "=" * 60)
    print("DEMO 2: Using Only Finviz + FinVADER")
    print("=" * 60)
    
    start_time = time.time()
    polarity, titles, label, pos, neg, neu = finviz_finvader_sentiment("AAPL", 3)
    end_time = time.time()
    
    print(f"Result: Polarity {polarity:.4f} ({label})")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Articles processed: {len(titles)}")
    print(f"Sentiment distribution: +{pos}/-{neg}/0{neu}")

def demo_single_source_google_news():
    """Demo using only Google News RSS"""
    print("\n" + "=" * 60)
    print("DEMO 3: Using Only Google News RSS")
    print("=" * 60)
    
    start_time = time.time()
    polarity, titles, label, pos, neg, neu = google_news_sentiment("AAPL", 3)
    end_time = time.time()
    
    print(f"Result: Polarity {polarity:.4f} ({label})")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Articles processed: {len(titles)}")
    print(f"Sentiment distribution: +{pos}/-{neg}/0{neu}")

def demo_custom_source_selection():
    """Demo using custom source selection"""
    print("\n" + "=" * 60)
    print("DEMO 4: Custom Source Selection (Finviz + Google News)")
    print("=" * 60)
    
    start_time = time.time()
    # Use only specific sources
    selected_sources = [SentimentSource.FINVIZ_FINVADER, SentimentSource.GOOGLE_NEWS]
    polarity, titles, label, pos, neg, neu = retrieving_news_polarity(
        "AAPL", 3, selected_sources=selected_sources
    )
    end_time = time.time()
    
    print(f"Result: Polarity {polarity:.4f} ({label})")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Articles processed: {len(titles)}")
    print(f"Sentiment distribution: +{pos}/-{neg}/0{neu}")

def demo_api_source():
    """Demo using an API source (without API key will gracefully skip)"""
    print("\n" + "=" * 60)
    print("DEMO 5: Using Alpha Vantage (API - without key)")
    print("=" * 60)
    
    start_time = time.time()
    polarity, titles, label, pos, neg, neu = alpha_vantage_sentiment("AAPL", 3)
    end_time = time.time()
    
    print(f"Result: Polarity {polarity:.4f} ({label})")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Articles processed: {len(titles)}")
    print(f"Sentiment distribution: +{pos}/-{neg}/0{neu}")
    print("(Note: Without API key, it gracefully skips to other sources)")

def main():
    """Run all demos"""
    print("Selectable Sentiment Sources Demo")
    print("Choose which sentiment analysis sources to use!")
    print("=" * 60)
    
    # Run all demos
    demo_all_sources()
    demo_single_source_finviz()
    demo_single_source_google_news()
    demo_custom_source_selection()
    demo_api_source()
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES:")
    print("=" * 60)
    print("# 1. Use all sources (default)")
    print("polarity, titles, label, pos, neg, neu = retrieving_news_polarity('AAPL', 5)")
    
    print("\n# 2. Use only Finviz + FinVADER")
    print("polarity, titles, label, pos, neg, neu = finviz_finvader_sentiment('AAPL', 5)")
    
    print("\n# 3. Use only Google News RSS")
    print("polarity, titles, label, pos, neg, neu = google_news_sentiment('AAPL', 5)")
    
    print("\n# 4. Custom selection")
    print("from news_sentiment import SentimentSource")
    print("selected = [SentimentSource.FINVIZ_FINVADER, SentimentSource.EODHD_API]")
    print("polarity, titles, label, pos, neg, neu = retrieving_news_polarity('AAPL', 5, selected_sources=selected)")
    
    print("\n# 5. Use API source with key")
    print("polarity, titles, label, pos, neg, neu = alpha_vantage_sentiment('AAPL', 5, api_key='your_key')")

if __name__ == "__main__":
    main()