#!/usr/bin/env python3
"""
Demo script showing how to use API keys with the sentiment analysis system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_sentiment import (
    finviz_finvader_sentiment,
    eodhd_sentiment,
    alpha_vantage_sentiment,
    social_sentiment,
    retrieving_news_polarity,
    SentimentSource
)

def demo_finviz_no_api_key():
    """Demo using Finviz (no API key needed)"""
    print("=" * 60)
    print("DEMO 1: Finviz + FinVADER (No API Key Required)")
    print("=" * 60)
    
    try:
        polarity, titles, label, pos, neg, neu = finviz_finvader_sentiment("AAPL", 3)
        print(f"✓ Success! Polarity: {polarity:.4f} ({label})")
        print(f"  Articles processed: {len(titles)}")
        print(f"  Sentiment distribution: +{pos}/-{neg}/0{neu}")
    except Exception as e:
        print(f"✗ Error: {e}")

def demo_api_without_key():
    """Demo using API sources without providing keys (should gracefully skip)"""
    print("\n" + "=" * 60)
    print("DEMO 2: API Sources Without Keys (Graceful Skip)")
    print("=" * 60)
    
    print("Testing Alpha Vantage without API key...")
    try:
        polarity, titles, label, pos, neg, neu = alpha_vantage_sentiment("AAPL", 3)
        print(f"✓ Success! Polarity: {polarity:.4f} ({label})")
        print(f"  Articles processed: {len(titles)}")
        print(f"  Sentiment distribution: +{pos}/-{neg}/0{neu}")
        print("  (Note: Without API key, it gracefully fell back to other sources)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\nTesting EODHD without API key...")
    try:
        polarity, titles, label, pos, neg, neu = eodhd_sentiment("AAPL", 3)
        print(f"✓ Success! Polarity: {polarity:.4f} ({label})")
        print(f"  Articles processed: {len(titles)}")
        print(f"  Sentiment distribution: +{pos}/-{neg}/0{neu}")
        print("  (Note: Without API key, it gracefully fell back to other sources)")
    except Exception as e:
        print(f"✗ Error: {e}")

def demo_api_with_fake_keys():
    """Demo using API sources with fake keys (should gracefully handle errors)"""
    print("\n" + "=" * 60)
    print("DEMO 3: API Sources With Invalid Keys (Graceful Error Handling)")
    print("=" * 60)
    
    print("Testing Alpha Vantage with invalid API key...")
    try:
        polarity, titles, label, pos, neg, neu = alpha_vantage_sentiment(
            "AAPL", 3, api_key="INVALID_KEY"
        )
        print(f"✓ Success! Polarity: {polarity:.4f} ({label})")
        print(f"  Articles processed: {len(titles)}")
        print(f"  Sentiment distribution: +{pos}/-{neg}/0{neu}")
        print("  (Note: With invalid key, it gracefully fell back to other sources)")
    except Exception as e:
        print(f"✗ Error: {e}")

def demo_mixed_sources():
    """Demo using a mix of sources with and without API keys"""
    print("\n" + "=" * 60)
    print("DEMO 4: Mixed Sources (Some With/Without API Keys)")
    print("=" * 60)
    
    selected_sources = [
        SentimentSource.FINVIZ_FINVADER,    # No key needed
        SentimentSource.ALPHA_VANTAGE,      # Key not provided
        SentimentSource.EODHD_API           # Key not provided
    ]
    
    try:
        polarity, titles, label, pos, neg, neu = retrieving_news_polarity(
            "AAPL", 5,
            selected_sources=selected_sources
        )
        print(f"✓ Success! Polarity: {polarity:.4f} ({label})")
        print(f"  Articles processed: {len(titles)}")
        print(f"  Sentiment distribution: +{pos}/-{neg}/0{neu}")
        print("  (Note: System used available sources and skipped unavailable ones)")
    except Exception as e:
        print(f"✗ Error: {e}")

def demo_how_to_add_real_keys():
    """Demo showing how to add real API keys"""
    print("\n" + "=" * 60)
    print("HOW TO ADD REAL API KEYS")
    print("=" * 60)
    
    print("""
To use real API keys, replace 'YOUR_API_KEY' with your actual key:

# Method 1: Direct parameter
polarity, titles, label, pos, neg, neu = alpha_vantage_sentiment(
    "AAPL", 5, api_key="YOUR_REAL_ALPHA_VANTAGE_KEY"
)

# Method 2: Environment variables (recommended for security)
import os
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
polarity, titles, label, pos, neg, neu = alpha_vantage_sentiment(
    "AAPL", 5, api_key=api_key
)

# Method 3: Multiple API keys
polarity, titles, label, pos, neg, neu = retrieving_news_polarity(
    "AAPL", 10,
    alpha_vantage_api_key="YOUR_ALPHA_VANTAGE_KEY",
    finnhub_api_key="YOUR_FINNHUB_KEY",
    eodhd_api_key="YOUR_EODHD_KEY",
    selected_sources=[
        SentimentSource.FINVIZ_FINVADER,
        SentimentSource.ALPHA_VANTAGE,
        SentimentSource.FINNHUB_SOCIAL
    ]
)
    """)

def main():
    """Run all demos"""
    print("API Keys Demo")
    print("Demonstrating how to use API keys with the sentiment analysis system")
    print("=" * 60)
    
    # Run all demos
    demo_finviz_no_api_key()
    demo_api_without_key()
    demo_api_with_fake_keys()
    demo_mixed_sources()
    demo_how_to_add_real_keys()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print("✓ Finviz + FinVADER: No API key needed")
    print("✓ Tradestie Reddit: No API key needed")
    print("✓ Google News RSS: No API key needed")
    print("⚠ EODHD API: API key required for premium features")
    print("⚠ Alpha Vantage: API key required")
    print("⚠ Finnhub Social: API key required")
    print("⚠ StockGeist: API key required for premium features")
    print("\nThe system gracefully handles missing API keys by skipping those sources.")

if __name__ == "__main__":
    main()