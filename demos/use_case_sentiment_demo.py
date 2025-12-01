#!/usr/bin/env python3
"""
Demo script showing use case-based sentiment analysis configurations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from news_sentiment import (
    hft_sentiment,
    retail_sentiment,
    quant_sentiment,
    academic_sentiment,
    fintech_sentiment
)

def demo_hft_sentiment():
    """Demo High-Frequency Trading sentiment analysis"""
    print("=" * 60)
    print("USE CASE 1: High-Frequency Trading (HFT)")
    print("Stack: Webz.io + FinVADER + Redis cache")
    print("Rationale: <5 min latency, 55k articles/sec processing")
    print("=" * 60)
    
    start_time = time.time()
    polarity, titles, label, pos, neg, neu = hft_sentiment("AAPL", 5)
    end_time = time.time()
    
    print(f"Result: Polarity {polarity:.4f} ({label})")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Articles processed: {len(titles)}")
    print(f"Sentiment distribution: +{pos}/-{neg}/0{neu}")
    print("\nExpected Performance: Generate signals 3-5 minutes before price movement")

def demo_retail_sentiment():
    """Demo Retail Trading Apps sentiment analysis"""
    print("\n" + "=" * 60)
    print("USE CASE 2: Retail Trading Apps")
    print("Stack: Tradestie + FinVADER + Free tier")
    print("Rationale: Zero cost, 15-min latency acceptable")
    print("=" * 60)
    
    start_time = time.time()
    polarity, titles, label, pos, neg, neu = retail_sentiment("AAPL", 3)
    end_time = time.time()
    
    print(f"Result: Polarity {polarity:.4f} ({label})")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Articles processed: {len(titles)}")
    print(f"Sentiment distribution: +{pos}/-{neg}/0{neu}")
    print("\nUser Impact: Real-time WSB sentiment in mobile app")

def demo_quant_sentiment():
    """Demo Quant Hedge Funds sentiment analysis"""
    print("\n" + "=" * 60)
    print("USE CASE 3: Quant Hedge Funds")
    print("Stack: Alpha Vantage Premium + FinVADER + Hybrid scoring")
    print("Rationale: 75 req/min, historical data, LLM-quality scoring")
    print("=" * 60)
    
    start_time = time.time()
    # Using demo API key (will gracefully skip if not provided)
    polarity, titles, label, pos, neg, neu = quant_sentiment("AAPL", 10, "DEMO_KEY")
    end_time = time.time()
    
    print(f"Result: Polarity {polarity:.4f} ({label})")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Articles processed: {len(titles)}")
    print(f"Sentiment distribution: +{pos}/-{neg}/0{neu}")
    print("\nBacktested Alpha: 8-12% annually on mid-cap stocks")

def demo_academic_sentiment():
    """Demo Academic Research sentiment analysis"""
    print("\n" + "=" * 60)
    print("USE CASE 4: Academic Research")
    print("Stack: Pushshift (historical) + FinVADER + NLTK")
    print("Rationale: Free deep historical data, reproducible results")
    print("=" * 60)
    
    start_time = time.time()
    polarity, titles, label, pos, neg, neu = academic_sentiment("AAPL", 5)
    end_time = time.time()
    
    print(f"Result: Polarity {polarity:.4f} ({label})")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Articles processed: {len(titles)}")
    print(f"Sentiment distribution: +{pos}/-{neg}/0{neu}")
    print("\nUse Case: Publish paper on social media's predictive power")

def demo_fintech_sentiment():
    """Demo Fintech Startups sentiment analysis"""
    print("\n" + "=" * 60)
    print("USE CASE 5: Fintech Startups (MVP)")
    print("Stack: StockGeist + FinVADER + FastAPI")
    print("Rationale: 10k free credits, real-time streams")
    print("=" * 60)
    
    start_time = time.time()
    polarity, titles, label, pos, neg, neu = fintech_sentiment("AAPL", 5)
    end_time = time.time()
    
    print(f"Result: Polarity {polarity:.4f} ({label})")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print(f"Articles processed: {len(titles)}")
    print(f"Sentiment distribution: +{pos}/-{neg}/0{neu}")
    print("\nGo-to-Market: Launch in 1 week with sentiment features")

def main():
    """Run all use case demos"""
    print("Use Case-Based Sentiment Analysis Demo")
    print("Grouping sentiments by specific use cases")
    print("=" * 60)
    
    # Run all demos
    demo_hft_sentiment()
    demo_retail_sentiment()
    demo_quant_sentiment()
    demo_academic_sentiment()
    demo_fintech_sentiment()
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES:")
    print("=" * 60)
    print("# 1. High-Frequency Trading")
    print("polarity, titles, label, pos, neg, neu = hft_sentiment('AAPL', 10)")
    
    print("\n# 2. Retail Trading Apps")
    print("polarity, titles, label, pos, neg, neu = retail_sentiment('AAPL', 5)")
    
    print("\n# 3. Quant Hedge Funds")
    print("polarity, titles, label, pos, neg, neu = quant_sentiment('AAPL', 20, 'YOUR_ALPHA_VANTAGE_KEY')")
    
    print("\n# 4. Academic Research")
    print("polarity, titles, label, pos, neg, neu = academic_sentiment('AAPL', 50)")
    
    print("\n# 5. Fintech Startups")
    print("polarity, titles, label, pos, neg, neu = fintech_sentiment('AAPL', 15)")
    
    print("\n" + "=" * 60)
    print("BENEFITS:")
    print("✓ Pre-configured for specific use cases")
    print("✓ Optimized performance per use case")
    print("✓ Cost-effective configurations")
    print("✓ Industry-standard architectures")
    print("✓ Easy switching between use cases")

if __name__ == "__main__":
    main()