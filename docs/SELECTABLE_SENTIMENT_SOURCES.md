# Selectable Sentiment Sources Feature

This document explains the new selectable sentiment sources feature that allows users to choose which sentiment analysis sources to use for their analysis.

## Available Sentiment Sources

### 1. Finviz + FinVADER (Primary Source)
- **Enum**: `SentimentSource.FINVIZ_FINVADER`
- **Description**: Fast HTML scraping from Finviz combined with enhanced financial sentiment analysis using FinVADER
- **Performance**: Fast execution with high accuracy for financial texts
- **Function**: `finviz_finvader_sentiment()`

### 2. EODHD API (API Fallback)
- **Enum**: `SentimentSource.EODHD_API`
- **Description**: Professional API providing pre-calculated sentiment scores
- **Function**: `eodhd_sentiment()`

### 3. Alpha Vantage News & Sentiments API
- **Enum**: `SentimentSource.ALPHA_VANTAGE`
- **Description**: Real-time news ingestion with full article text and composite scoring
- **Function**: `alpha_vantage_sentiment()`

### 4. Tradestie WallStreetBets API
- **Enum**: `SentimentSource.TRADESTIE_REDDIT`
- **Description**: 15-minute updates with raw Reddit comments/posts
- **Function**: `reddit_sentiment()`

### 5. Finnhub Social Sentiment API
- **Enum**: `SentimentSource.FINNHUB_SOCIAL`
- **Description**: Multi-source social sentiment from Reddit, Twitter, Yahoo Finance, StockTwits
- **Function**: `social_sentiment()`

### 6. StockGeist.ai
- **Enum**: `SentimentSource.STOCKGEIST`
- **Description**: Real-time streaming capability with SSE streams or REST API

### 7. Google News RSS (Last Resort)
- **Enum**: `SentimentSource.GOOGLE_NEWS`
- **Description**: RSS feed parsing for wide news coverage when other sources are insufficient
- **Function**: `google_news_sentiment()`

### 8. All Sources (Default)
- **Enum**: `SentimentSource.ALL_SOURCES`
- **Description**: Uses all available sources in priority order

## Usage Options

### 1. Default Behavior (All Sources)
```python
from news_sentiment import retrieving_news_polarity

# Uses all sources in priority order
polarity, titles, label, pos, neg, neu = retrieving_news_polarity("AAPL", 5)
```

### 2. Single Source Functions (Easiest)
```python
from news_sentiment import (
    finviz_finvader_sentiment,
    eodhd_sentiment,
    alpha_vantage_sentiment,
    reddit_sentiment,
    social_sentiment,
    google_news_sentiment
)

# Use only Finviz + FinVADER
polarity, titles, label, pos, neg, neu = finviz_finvader_sentiment("AAPL", 5)

# Use only Google News RSS
polarity, titles, label, pos, neg, neu = google_news_sentiment("AAPL", 5)

# Use only Alpha Vantage (with API key)
polarity, titles, label, pos, neg, neu = alpha_vantage_sentiment("AAPL", 5, api_key="your_key")
```

### 3. Custom Source Selection
```python
from news_sentiment import retrieving_news_polarity, SentimentSource

# Use only specific sources
selected_sources = [
    SentimentSource.FINVIZ_FINVADER,
    SentimentSource.GOOGLE_NEWS
]

polarity, titles, label, pos, neg, neu = retrieving_news_polarity(
    "AAPL", 5, 
    selected_sources=selected_sources
)
```

### 4. Advanced Custom Selection with API Keys
```python
from news_sentiment import retrieving_news_polarity, SentimentSource

# Use specific sources with API keys
selected_sources = [
    SentimentSource.FINVIZ_FINVADER,
    SentimentSource.EODHD_API,
    SentimentSource.ALPHA_VANTAGE
]

polarity, titles, label, pos, neg, neu = retrieving_news_polarity(
    "AAPL", 5,
    eodhd_api_key="your_eodhd_key",
    alpha_vantage_api_key="your_alpha_vantage_key",
    selected_sources=selected_sources
)
```

## Benefits of Selectable Sources

### 1. Performance Optimization
- Choose faster sources for quick analysis
- Skip slower API calls when not needed
- Reduce execution time by selecting only necessary sources

### 2. Cost Control
- Use only free sources to avoid API costs
- Selectively enable premium APIs when needed
- Control API usage based on budget constraints

### 3. Use Case Specificity
- Social sentiment for retail investor analysis
- Professional APIs for institutional use
- News-only sources for traditional analysis

### 4. Reliability Management
- Use proven sources for critical applications
- Avoid experimental APIs in production
- Mix reliable and experimental sources for research

## Source Priority Order

When using multiple sources, they are processed in this priority order:
1. Finviz + FinVADER (Fastest, most reliable)
2. EODHD API (Pre-calculated sentiment)
3. Alpha Vantage (Real-time news)
4. Tradestie Reddit (Social sentiment)
5. Finnhub Social (Multi-platform social)
6. StockGeist (Real-time streaming)
7. Google News RSS (Last resort)

## Examples by Use Case

### Quick Analysis
```python
# Fastest option - only Finviz + FinVADER
from news_sentiment import finviz_finvader_sentiment
polarity, titles, label, pos, neg, neu = finviz_finvader_sentiment("AAPL", 5)
```

### Social Sentiment Focus
```python
# Reddit and social media focus
from news_sentiment import SentimentSource
selected = [
    SentimentSource.TRADESTIE_REDDIT,
    SentimentSource.FINNHUB_SOCIAL
]
polarity, titles, label, pos, neg, neu = retrieving_news_polarity(
    "AAPL", 10, selected_sources=selected
)
```

### Professional Analysis
```python
# Premium sources for professional analysis
from news_sentiment import SentimentSource
selected = [
    SentimentSource.EODHD_API,
    SentimentSource.ALPHA_VANTAGE,
    SentimentSource.FINNHUB_SOCIAL
]
polarity, titles, label, pos, neg, neu = retrieving_news_polarity(
    "AAPL", 10,
    eodhd_api_key="your_key",
    alpha_vantage_api_key="your_key",
    finnhub_api_key="your_key",
    selected_sources=selected
)
```

## Implementation Details

### Source Selection Logic
The system uses a flexible selection mechanism:
- By default, all sources are enabled
- Users can specify exactly which sources to use
- Sources are processed in priority order until article count is satisfied
- API sources gracefully skip when keys aren't provided

### Backward Compatibility
All existing code continues to work unchanged:
```python
# This still works exactly as before
polarity, titles, label, pos, neg, neu = retrieving_news_polarity("AAPL", 5)
```

### Error Handling
- Sources gracefully skip when API keys aren't provided
- System continues with available sources when some fail
- Clear logging shows which sources are being used

## Conclusion

The selectable sentiment sources feature provides maximum flexibility for users to choose exactly which sentiment analysis sources to use based on their specific needs, budget, and performance requirements. Whether you need a quick analysis, social sentiment focus, or professional-grade results, you can now configure the system to use exactly the sources that best fit your use case.