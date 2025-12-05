"""
Test helper utilities for the Stock Market Prediction Web App tests.

This module provides utility functions and classes for testing.
"""

import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock
import random


class MockResponse:
    """Mock HTTP response object for API testing."""
    
    def __init__(self, json_data=None, status_code=200, text=''):
        self.json_data = json_data
        self.status_code = status_code
        self.text = text
        self.content = text.encode()
    
    def json(self):
        """Return JSON data."""
        if self.json_data is None:
            raise ValueError("No JSON data")
        return self.json_data
    
    def raise_for_status(self):
        """Raise exception for bad status codes."""
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def generate_stock_data(symbol='AAPL', days=730, start_price=150.0):
    """
    Generate realistic mock stock data for testing.
    
    Args:
        symbol: Stock symbol
        days: Number of days of data
        start_price: Starting price for the stock
    
    Returns:
        pandas DataFrame with stock data
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic price movements using random walk
    prices = [start_price]
    for _ in range(days - 1):
        change = random.gauss(0, 2)  # Mean 0, std 2
        new_price = max(prices[-1] + change, 1.0)  # Ensure positive
        prices.append(new_price)
    
    data = {
        'Date': dates,
        'Open': [p + random.gauss(0, 0.5) for p in prices],
        'High': [p + abs(random.gauss(1, 0.5)) for p in prices],
        'Low': [p - abs(random.gauss(1, 0.5)) for p in prices],
        'Close': prices,
        'Volume': [random.randint(1000000, 50000000) for _ in range(days)],
        'Adj Close': prices
    }
    
    df = pd.DataFrame(data)
    return df


def generate_sentiment_data(num_articles=10, overall_sentiment='positive'):
    """
    Generate mock sentiment analysis data.
    
    Args:
        num_articles: Number of articles to simulate
        overall_sentiment: 'positive', 'negative', or 'neutral'
    
    Returns:
        Tuple of (polarity, sentiment_list, sentiment_pol, pos, neg, neutral)
    """
    sentiment_map = {
        'positive': (0.5, 1.0),
        'negative': (-1.0, -0.5),
        'neutral': (-0.2, 0.2)
    }
    
    range_min, range_max = sentiment_map.get(overall_sentiment, (0, 0))
    
    sentiment_list = []
    sentiment_pol = []
    pos, neg, neutral = 0, 0, 0
    
    for i in range(num_articles):
        polarity = random.uniform(range_min, range_max)
        sentiment_pol.append(polarity)
        
        if polarity > 0.1:
            sentiment_list.append(f"Positive article {i+1}")
            pos += 1
        elif polarity < -0.1:
            sentiment_list.append(f"Negative article {i+1}")
            neg += 1
        else:
            sentiment_list.append(f"Neutral article {i+1}")
            neutral += 1
    
    overall_polarity = sum(sentiment_pol) / len(sentiment_pol) if sentiment_pol else 0
    
    return (overall_polarity, sentiment_list, sentiment_pol, pos, neg, neutral)


def mock_finviz_response(symbol):
    """Mock Finviz news scraping response."""
    articles = [
        {
            'title': f'{symbol} Stock Surges on Strong Earnings',
            'url': f'https://example.com/news/{symbol}/1',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'sentiment': 0.6
        },
        {
            'title': f'{symbol} Announces New Product Launch',
            'url': f'https://example.com/news/{symbol}/2',
            'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            'sentiment': 0.4
        },
        {
            'title': f'Analysts Upgrade {symbol} Price Target',
            'url': f'https://example.com/news/{symbol}/3',
            'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
            'sentiment': 0.5
        }
    ]
    return articles


def mock_alpha_vantage_response(symbol):
    """Mock Alpha Vantage API response."""
    return {
        'Time Series (Daily)': {
            '2024-01-01': {
                '1. open': '150.00',
                '2. high': '155.00',
                '3. low': '148.00',
                '4. close': '152.50',
                '5. volume': '12345678'
            }
        }
    }


def create_mock_lstm_model():
    """Create a mock LSTM model for testing."""
    model = MagicMock()
    model.fit = MagicMock(return_value=None)
    model.predict = MagicMock(return_value=np.array([[150.0], [151.0], [152.0]]))
    return model


def create_mock_arima_model():
    """Create a mock ARIMA model for testing."""
    model = MagicMock()
    model_fit = MagicMock()
    model_fit.forecast = MagicMock(return_value=[150.0])
    model.fit = MagicMock(return_value=model_fit)
    return model


def assert_decimal_equal(actual, expected, places=2):
    """
    Assert two Decimal values are equal within a number of decimal places.
    
    Args:
        actual: Actual Decimal value
        expected: Expected Decimal value
        places: Number of decimal places to compare
    """
    actual_rounded = Decimal(str(actual)).quantize(Decimal(10) ** -places)
    expected_rounded = Decimal(str(expected)).quantize(Decimal(10) ** -places)
    assert actual_rounded == expected_rounded, \
        f"Expected {expected_rounded}, got {actual_rounded}"


def create_test_user_data(role='user'):
    """Create test user registration data."""
    timestamp = int(datetime.now().timestamp())
    return {
        'email': f'testuser{timestamp}@example.com',
        'username': f'testuser{timestamp}',
        'password': 'TestPass123!',
        'confirm_password': 'TestPass123!',
        'role': role
    }


def create_test_transaction_data(symbol='AAPL', quantity=10, txn_type='BUY'):
    """Create test transaction data."""
    return {
        'symbol': symbol,
        'quantity': str(quantity),
        'txn_type': txn_type
    }


def mock_requests_get(url, *args, **kwargs):
    """
    Mock requests.get for various API endpoints.
    
    Args:
        url: URL being requested
    
    Returns:
        MockResponse object
    """
    if 'finviz.com' in url:
        return MockResponse(
            text='<html><div class="news-link">Stock news here</div></html>',
            status_code=200
        )
    elif 'alphavantage.co' in url:
        return MockResponse(
            json_data=mock_alpha_vantage_response('AAPL'),
            status_code=200
        )
    elif 'eodhd.com' in url:
        return MockResponse(
            json_data={'sentiment': 0.5, 'articles': []},
            status_code=200
        )
    else:
        return MockResponse(status_code=404)


class DatabaseTestHelper:
    """Helper class for database testing operations."""
    
    @staticmethod
    def count_records(model):
        """Count total records for a model."""
        return model.query.count()
    
    @staticmethod
    def get_user_by_email(email):
        """Get user by email."""
        from main import User
        return User.query.filter_by(email=email).first()
    
    @staticmethod
    def get_company_by_symbol(symbol):
        """Get company by symbol."""
        from main import Company
        return Company.query.filter_by(symbol=symbol).first()
    
    @staticmethod
    def clear_all_data(db):
        """Clear all data from database."""
        from main import Dividend, Transaction, PortfolioItem, Broker, Company, User
        
        # Delete in correct order due to foreign keys
        Dividend.query.delete()
        Transaction.query.delete()
        PortfolioItem.query.delete()
        Broker.query.delete()
        Company.query.delete()
        User.query.delete()
        db.session.commit()


def calculate_expected_commission(total_amount, commission_rate):
    """
    Calculate expected commission amount.
    
    Args:
        total_amount: Total transaction amount
        commission_rate: Commission rate percentage
    
    Returns:
        Decimal commission amount
    """
    return (Decimal(str(total_amount)) * Decimal(str(commission_rate)) / Decimal('100')).quantize(Decimal('0.01'))
