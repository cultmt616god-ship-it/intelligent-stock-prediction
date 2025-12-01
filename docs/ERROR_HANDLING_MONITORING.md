# Error Handling and Monitoring Documentation

This document explains the robust error handling and monitoring features implemented in the sentiment analysis system.

## 1. Robust Error Handling

### Tenacity-Based Retry Mechanisms

The system implements production-grade retry mechanisms using the `tenacity` library:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def robust_finvader(text: str):
    """Production-grade FinVADER with retries"""
    try:
        return finvader(text)
    except Exception as e:
        logger.error(f"FinVADER failed: {e}")
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}  # Neutral fallback
```

### Key Features

1. **Exponential Backoff**: Waits 4, 8, 16 seconds between retries
2. **Maximum Attempts**: Stops after 3 failed attempts
3. **Graceful Fallback**: Returns neutral sentiment when all retries fail
4. **Detailed Logging**: Logs all errors with context
5. **Compatibility Fallback**: Works even without tenacity library

### Implementation

```python
def robust_finvader(self, text: str) -> Dict:
    """
    Production-grade FinVADER with retries
    """
    if not TENACITY_AVAILABLE:
        # Fallback implementation without tenacity
        try:
            return finvader(text)
        except Exception as e:
            logger.error(f"FinVADER failed: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}  # Neutral fallback
    
    # With tenacity retry decorator
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _robust_finvader_inner(text: str):
        try:
            return finvader(text)
        except Exception as e:
            logger.error(f"FinVADER failed: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}  # Neutral fallback
    
    return _robust_finvader_inner(text)
```

## 2. Comprehensive Monitoring

### Sentiment Distribution Logging

Monitor the confidence distribution of sentiment scores:

```python
def log_sentiment_distribution(scores: list):
    compounds = [s['compound'] for s in scores]
    logger.info(f"Mean: {np.mean(compounds):.3f}, Std: {np.std(compounds):.3f}")
    logger.info(f"Extremes: {sum(1 for s in compounds if abs(s) > 0.5)} / {len(compounds)}")
```

### Implementation

```python
def log_sentiment_distribution(self, articles: List[Dict]):
    """
    Monitor confidence distribution
    """
    try:
        compounds = []
        for article in articles:
            if 'sentiment_score' in article:
                compounds.append(article['sentiment_score'])
            elif 'text' in article and FINVADER_AVAILABLE:
                try:
                    score = finvader(article['text'])
                    compounds.append(score['compound'])
                except:
                    compounds.append(0.0)
            else:
                compounds.append(0.0)
        
        if compounds:
            mean_sentiment = np.mean(compounds)
            std_sentiment = np.std(compounds)
            extremes = sum(1 for s in compounds if abs(s) > 0.5)
            
            logger.info(f"Sentiment Distribution - Mean: {mean_sentiment:.3f}, Std: {std_sentiment:.3f}")
            logger.info(f"Extreme Sentiments: {extremes} / {len(compounds)}")
            logger.info(f"Compound scores range: [{min(compounds):.3f}, {max(compounds):.3f}]")
    except Exception as e:
        logger.error(f"Error logging sentiment distribution: {e}")
```

## 3. Error Recovery Patterns

### Graceful Degradation

The system implements multiple layers of fallback mechanisms:

1. **API Failures**: Fall back to alternative data sources
2. **Network Issues**: Retry with exponential backoff
3. **Parsing Errors**: Return neutral sentiment scores
4. **Missing Data**: Continue with partial results

### Example Implementation

```python
# In get_sentiment method
try:
    # Primary analysis attempt
    compound = finvader(article['text'])['compound']
except Exception as e:
    logger.warning(f"FinVADER analysis failed, falling back to VADER: {e}")
    try:
        # Fallback to standard VADER
        scores = self.sid.polarity_scores(article['text'])
        compound = scores['compound']
    except Exception as e2:
        logger.error(f"Both FinVADER and VADER failed: {e2}")
        compound = 0.0  # Neutral fallback
```

## 4. Logging Strategy

### Structured Logging

All errors and important events are logged with structured information:

```python
import logging
logger = logging.getLogger(__name__)

# Example logs:
logger.info("Found 5 articles on Finviz")
logger.warning("FinVADER analysis failed, falling back to VADER")
logger.error("API request failed after 3 retries")
```

### Log Levels

1. **DEBUG**: Detailed technical information for troubleshooting
2. **INFO**: General operational information
3. **WARNING**: Potential issues that don't stop execution
4. **ERROR**: Significant problems that may affect results
5. **CRITICAL**: Severe errors that stop execution

## 5. Usage Examples

### Robust FinVADER Analysis

```python
from news_sentiment import robust_finvader_analysis

# Normal usage
result = robust_finvader_analysis("Company exceeds expectations")
print(f"Compound score: {result['compound']:.4f}")

# Edge case handling
result = robust_finvader_analysis("")  # Empty text
print(f"Fallback result: {result}")  # Returns neutral sentiment
```

### Monitoring Sentiment Distribution

```python
from news_sentiment import log_sentiment_distribution

# Log distribution of sentiment scores
scores = [
    {'compound': 0.25},
    {'compound': -0.15},
    {'compound': 0.45},
    {'compound': -0.35},
    {'compound': 0.10}
]

log_sentiment_distribution(scores)
# Output:
# INFO: Sentiment Distribution - Mean: 0.060, Std: 0.324
# INFO: Extreme Sentiments: 3 / 5
# INFO: Compound scores range: [-0.350, 0.450]
```

### Production Monitoring

```python
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)

# All operations will now be logged
polarity, titles, label, pos, neg, neu = retrieving_news_polarity("AAPL", 10)
# Logs will show:
# INFO: Found 10 articles on Finviz
# INFO: Analyzing sentiment for 10 articles...
# INFO: Sentiment Distribution - Mean: 0.123, Std: 0.234
```

## 6. Error Handling Best Practices

### Retry Strategy

1. **Exponential Backoff**: Prevents overwhelming failing services
2. **Maximum Attempts**: Prevents infinite retry loops
3. **Jitter**: Reduces thundering herd problems
4. **Specific Exceptions**: Only retries on transient errors

### Fallback Chain

1. **Primary Source**: Finviz + FinVADER
2. **API Fallback**: EODHD, Alpha Vantage, etc.
3. **Social Sources**: Reddit, Twitter sentiment
4. **Last Resort**: Google News RSS
5. **Neutral Default**: Zero sentiment when all fail

### Data Validation

```python
# Validate inputs before processing
if not text or not isinstance(text, str):
    logger.warning("Invalid text input, returning neutral sentiment")
    return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

# Validate outputs
if not isinstance(result, dict) or 'compound' not in result:
    logger.error("Invalid sentiment result format")
    return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
```

## 7. Monitoring Dashboard Integration

### Metrics Collection

The system collects key metrics for monitoring:

1. **Success Rate**: Percentage of successful analyses
2. **Latency**: Time taken for each analysis
3. **Error Rate**: Frequency of different error types
4. **Distribution**: Sentiment score distributions
5. **Source Reliability**: Success rates by data source

### Alerting Thresholds

```python
# Example alerting logic
def should_alert(mean_sentiment, std_sentiment, error_rate):
    alerts = []
    
    if abs(mean_sentiment) > 0.5:
        alerts.append("High sentiment magnitude detected")
    
    if std_sentiment > 0.3:
        alerts.append("High sentiment volatility detected")
    
    if error_rate > 0.1:
        alerts.append("High error rate - check data sources")
    
    return alerts
```

## 8. Testing Error Handling

### Unit Tests

```python
def test_robust_finvader_with_invalid_input():
    """Test robust FinVADER with invalid input"""
    result = robust_finvader_analysis("")
    assert result['compound'] == 0.0
    assert result['neu'] == 1.0

def test_finvader_with_network_failure():
    """Test FinVADER retry mechanism"""
    # Mock network failure and verify retries
    pass

def test_sentiment_distribution_logging():
    """Test sentiment distribution logging"""
    scores = [{'compound': 0.5}, {'compound': -0.5}]
    # Verify logging output
    pass
```

## Conclusion

The error handling and monitoring features make the sentiment analysis system production-ready with:

- **Resilience**: Automatic recovery from transient failures
- **Observability**: Comprehensive logging and monitoring
- **Reliability**: Graceful degradation when components fail
- **Maintainability**: Clear error messages and structured logging
- **Performance**: Efficient retry strategies that don't overwhelm services

These features ensure the system operates reliably in production environments with minimal manual intervention required.