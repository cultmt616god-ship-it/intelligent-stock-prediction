# API Keys Guide for Sentiment Analysis

This guide explains which sentiment sources require API keys, how to obtain them, and how to use them with the sentiment analysis system.

## Sources That Require API Keys

### 1. EODHD API
- **Enum**: `SentimentSource.EODHD_API`
- **Required**: Yes (for premium features)
- **Free Tier**: Limited access
- **Usage**: Pre-calculated sentiment scores

#### How to Get an API Key:
1. Visit [https://eodhd.com/](https://eodhd.com/)
2. Sign up for a free account
3. Navigate to your dashboard to find your API token

#### Usage Example:
```python
from news_sentiment import eodhd_sentiment

# With API key
polarity, titles, label, pos, neg, neu = eodhd_sentiment(
    "AAPL", 5, api_key="your_eodhd_api_key"
)

# Without API key (will gracefully skip to other sources)
polarity, titles, label, pos, neg, neu = eodhd_sentiment("AAPL", 5)
```

### 2. Alpha Vantage News & Sentiments API
- **Enum**: `SentimentSource.ALPHA_VANTAGE`
- **Required**: Yes
- **Free Tier**: 5 API calls per minute, 500 calls per day
- **Usage**: Real-time news ingestion with full article text

#### How to Get an API Key:
1. Visit [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Sign up for a free account
3. You'll receive your API key via email

#### Usage Example:
```python
from news_sentiment import alpha_vantage_sentiment

# With API key
polarity, titles, label, pos, neg, neu = alpha_vantage_sentiment(
    "AAPL", 5, api_key="your_alpha_vantage_key"
)

# Without API key (will gracefully skip to other sources)
polarity, titles, label, pos, neg, neu = alpha_vantage_sentiment("AAPL", 5)
```

### 3. Finnhub Social Sentiment API
- **Enum**: `SentimentSource.FINNHUB_SOCIAL`
- **Required**: Yes
- **Free Tier**: 60 API calls per minute
- **Usage**: Multi-source social sentiment from Reddit, Twitter, Yahoo Finance, StockTwits

#### How to Get an API Key:
1. Visit [https://finnhub.io/dashboard](https://finnhub.io/dashboard)
2. Sign up for a free account
3. Navigate to the dashboard to find your API key

#### Usage Example:
```python
from news_sentiment import social_sentiment

# With API key
polarity, titles, label, pos, neg, neu = social_sentiment(
    "AAPL", 5, api_key="your_finnhub_key"
)

# Without API key (will gracefully skip to other sources)
polarity, titles, label, pos, neg, neu = social_sentiment("AAPL", 5)
```

### 4. StockGeist.ai
- **Enum**: `SentimentSource.STOCKGEIST`
- **Required**: Yes
- **Free Tier**: 10,000 free credits per month
- **Usage**: Real-time streaming capability with SSE streams or REST API

#### How to Get an API Key:
1. Visit [https://stockgeist.ai/](https://stockgeist.ai/)
2. Sign up for an account
3. Navigate to your dashboard to find your API key

#### Usage Example:
```python
from news_sentiment import retrieving_news_polarity, SentimentSource

# With API key
selected_sources = [SentimentSource.STOCKGEIST, SentimentSource.FINVIZ_FINVADER]
polarity, titles, label, pos, neg, neu = retrieving_news_polarity(
    "AAPL", 5,
    stockgeist_api_key="your_stockgeist_key",
    selected_sources=selected_sources
)
```

## Sources That Do NOT Require API Keys

### 1. Finviz + FinVADER
- **Enum**: `SentimentSource.FINVIZ_FINVADER`
- **Required**: No
- **Usage**: Fast HTML scraping from Finviz combined with enhanced financial sentiment analysis

#### Usage Example:
```python
from news_sentiment import finviz_finvader_sentiment

# No API key needed
polarity, titles, label, pos, neg, neu = finviz_finvader_sentiment("AAPL", 5)
```

### 2. Tradestie WallStreetBets API
- **Enum**: `SentimentSource.TRADESTIE_REDDIT`
- **Required**: No
- **Usage**: 15-minute updates with raw Reddit comments/posts

#### Usage Example:
```python
from news_sentiment import reddit_sentiment

# No API key needed
polarity, titles, label, pos, neg, neu = reddit_sentiment("AAPL", 5)
```

### 3. Google News RSS
- **Enum**: `SentimentSource.GOOGLE_NEWS`
- **Required**: No
- **Usage**: RSS feed parsing for wide news coverage

#### Usage Example:
```python
from news_sentiment import google_news_sentiment

# No API key needed
polarity, titles, label, pos, neg, neu = google_news_sentiment("AAPL", 5)
```

## How to Use Multiple API Keys

You can combine multiple API sources in a single analysis:

```python
from news_sentiment import retrieving_news_polarity, SentimentSource

# Using multiple API sources with keys
selected_sources = [
    SentimentSource.FINVIZ_FINVADER,    # No key needed
    SentimentSource.EODHD_API,          # API key required
    SentimentSource.ALPHA_VANTAGE,      # API key required
    SentimentSource.FINNHUB_SOCIAL      # API key required
]

polarity, titles, label, pos, neg, neu = retrieving_news_polarity(
    "AAPL", 10,
    eodhd_api_key="your_eodhd_key",
    alpha_vantage_api_key="your_alpha_vantage_key",
    finnhub_api_key="your_finnhub_key",
    selected_sources=selected_sources
)
```

## Use Case-Specific API Requirements

### High-Frequency Trading (HFT)
- **Primary**: Finviz + FinVADER (No API key needed)
- **Optional**: Webz.io (would require API key if implemented)

### Retail Trading Apps
- **Primary**: Tradestie Reddit (No API key needed)
- **Secondary**: Finviz + FinVADER (No API key needed)

### Quant Hedge Funds
- **Primary**: Alpha Vantage Premium (API key required)
- **Secondary**: Finviz + FinVADER (No API key needed)

### Academic Research
- **Primary**: Google News RSS (No API key needed)
- **Secondary**: Finviz + FinVADER (No API key needed)

### Fintech Startups (MVP)
- **Primary**: StockGeist (API key required for premium features)
- **Secondary**: Finviz + FinVADER (No API key needed)

## Best Practices

### 1. Graceful Degradation
The system is designed to gracefully skip API sources when keys aren't provided:

```python
# This will work even without API keys
# It will use available sources and skip unavailable ones
polarity, titles, label, pos, neg, neu = retrieving_news_polarity(
    "AAPL", 5,
    # API keys not provided - sources will be skipped gracefully
    selected_sources=[
        SentimentSource.EODHD_API,
        SentimentSource.ALPHA_VANTAGE,
        SentimentSource.FINVIZ_FINVADER  # This will work
    ]
)
```

### 2. Environment Variables for Security
Store API keys in environment variables rather than hardcoding them:

```python
import os
from news_sentiment import alpha_vantage_sentiment

# Get API key from environment variable
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

polarity, titles, label, pos, neg, neu = alpha_vantage_sentiment(
    "AAPL", 5, api_key=api_key
)
```

### 3. Rate Limiting Awareness
Be aware of rate limits for each API:

- **Alpha Vantage**: 5 calls/minute (free), 1200 calls/minute (premium)
- **Finnhub**: 60 calls/minute (free)
- **EODHD**: Varies by plan
- **StockGeist**: 10,000 credits/month (free tier)

## Troubleshooting

### API Key Not Working
1. Verify the key is correct and active
2. Check if you've exceeded rate limits
3. Ensure the key is for the correct service
4. Check network connectivity

### "API key not provided" Messages
This is normal behavior when API keys aren't provided. The system will skip those sources and continue with available ones.

### Rate Limit Exceeded
1. Reduce the frequency of API calls
2. Upgrade to a premium plan
3. Implement caching to reduce redundant calls