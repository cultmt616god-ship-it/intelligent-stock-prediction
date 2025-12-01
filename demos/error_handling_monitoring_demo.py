#!/usr/bin/env python3
"""
Demo script showing error handling and monitoring features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from news_sentiment import (
    robust_finvader_analysis,
    log_sentiment_distribution,
    finviz_finvader_sentiment
)

# Set up logging to see the monitoring output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def demo_robust_finvader():
    """Demo robust FinVADER with retries"""
    print("=" * 60)
    print("DEMO 1: Robust FinVADER with Retries")
    print("=" * 60)
    
    # Test with normal text
    text = "Apple reports strong earnings and exceeds expectations"
    result = robust_finvader_analysis(text)
    print(f"Normal text analysis:")
    print(f"  Text: {text}")
    print(f"  Compound score: {result.get('compound', 0):.4f}")
    
    # Test with edge case text
    edge_text = ""  # Empty text
    result2 = robust_finvader_analysis(edge_text)
    print(f"\nEdge case analysis (empty text):")
    print(f"  Compound score: {result2.get('compound', 0):.4f}")
    print(f"  Fallback to neutral: {result2}")

def demo_sentiment_distribution_logging():
    """Demo sentiment distribution logging"""
    print("\n" + "=" * 60)
    print("DEMO 2: Sentiment Distribution Logging")
    print("=" * 60)
    
    # Get some sample sentiment scores
    polarity, titles, label, pos, neg, neu = finviz_finvader_sentiment("AAPL", 5)
    
    # Create sample scores for logging
    sample_scores = []
    if titles:
        # Simulate some sentiment scores
        sample_scores = [
            {'compound': 0.25, 'pos': 0.3, 'neu': 0.6, 'neg': 0.1},
            {'compound': -0.15, 'pos': 0.1, 'neu': 0.7, 'neg': 0.2},
            {'compound': 0.45, 'pos': 0.5, 'neu': 0.4, 'neg': 0.1},
            {'compound': -0.35, 'pos': 0.05, 'neu': 0.6, 'neg': 0.35},
            {'compound': 0.10, 'pos': 0.2, 'neu': 0.7, 'neg': 0.1}
        ]
    
    print("Logging sentiment distribution for sample scores:")
    log_sentiment_distribution(sample_scores)

def demo_error_recovery():
    """Demo error recovery and graceful degradation"""
    print("\n" + "=" * 60)
    print("DEMO 3: Error Recovery and Graceful Degradation")
    print("=" * 60)
    
    print("The system includes multiple layers of error handling:")
    print("1. Try/Except blocks around all API calls")
    print("2. Retry mechanisms with exponential backoff")
    print("3. Fallback to alternative sources")
    print("4. Neutral sentiment fallback when analysis fails")
    print("5. Detailed logging of all errors")
    
    # Show how errors are handled
    print("\nExample error handling in action:")
    print("- Network timeouts: Retried with exponential backoff")
    print("- API rate limits: Automatic pause and retry")
    print("- Parsing errors: Fallback to neutral sentiment")
    print("- Missing data: Continue with available information")

def main():
    """Run all demos"""
    print("Error Handling and Monitoring Demo")
    print("Showcasing robust error handling and monitoring features")
    print("=" * 60)
    
    # Run all demos
    demo_robust_finvader()
    demo_sentiment_distribution_logging()
    demo_error_recovery()
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES:")
    print("=" * 60)
    print("# 1. Robust FinVADER analysis with retries")
    print("result = robust_finvader_analysis('Company exceeds expectations')")
    print("print(f'Compound score: {result[\"compound\"]:.4f}')")
    
    print("\n# 2. Log sentiment distribution")
    print("scores = [{'compound': 0.25}, {'compound': -0.15}, {'compound': 0.45}]")
    print("log_sentiment_distribution(scores)")
    
    print("\n# 3. Error monitoring in production")
    print("# All errors are logged with detailed information")
    print("# Retry mechanisms handle transient failures")
    print("# System continues operating even when some sources fail")

if __name__ == "__main__":
    main()