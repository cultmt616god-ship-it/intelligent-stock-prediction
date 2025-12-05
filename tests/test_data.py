"""
Sample test data for the Stock Market Prediction Web App.

This module provides realistic test data for various testing scenarios.
"""

from decimal import Decimal
from datetime import datetime, timedelta

# User test data
SAMPLE_USERS = [
    {
        'email': 'john.doe@example.com',
        'username': 'johndoe',
        'password': 'SecurePass123!',
        'role': 'user',
        'wallet_balance': Decimal('10000.00')
    },
    {
        'email': 'jane.smith@example.com',
        'username': 'janesmith',
        'password': 'SecurePass456!',
        'role': 'user',
        'wallet_balance': Decimal('25000.00')
    },
    {
        'email': 'admin@stockapp.com',
        'username': 'admin',
        'password': 'AdminPass789!',
        'role': 'admin',
        'wallet_balance': Decimal('100000.00')
    }
]

# Company test data
SAMPLE_COMPANIES = [
    {
        'symbol': 'AAPL',
        'name': 'Apple Inc.',
        'exchange': 'NASDAQ',
        'sector': 'Technology',
        'is_active': True
    },
    {
        'symbol': 'GOOGL',
        'name': 'Alphabet Inc.',
        'exchange': 'NASDAQ',
        'sector': 'Technology',
        'is_active': True
    },
    {
        'symbol': 'TSLA',
        'name': 'Tesla Inc.',
        'exchange': 'NASDAQ',
        'sector': 'Automotive',
        'is_active': True
    },
    {
        'symbol': 'MSFT',
        'name': 'Microsoft Corporation',
        'exchange': 'NASDAQ',
        'sector': 'Technology',
        'is_active': True
    },
    {
        'symbol': 'NVDA',
        'name': 'NVIDIA Corporation',
        'exchange': 'NASDAQ',
        'sector': 'Technology',
        'is_active': True
    }
]

# Broker test data
SAMPLE_BROKERS = [
    {
        'name': 'Robinhood',
        'email': 'support@robinhood.com',
        'commission_rate': Decimal('0.00'),
        'is_active': True
    },
    {
        'name': 'E*TRADE',
        'email': 'support@etrade.com',
        'commission_rate': Decimal('0.50'),
        'is_active': True
    },
    {
        'name': 'TD Ameritrade',
        'email': 'support@tdameritrade.com',
        'commission_rate': Decimal('0.65'),
        'is_active': True
    },
    {
        'name': 'Interactive Brokers',
        'email': 'support@interactivebrokers.com',
        'commission_rate': Decimal('0.35'),
        'is_active': True
    }
]

# Stock price test data
SAMPLE_STOCK_PRICES = {
    'AAPL': Decimal('175.50'),
    'GOOGL': Decimal('142.30'),
    'TSLA': Decimal('238.45'),
    'MSFT': Decimal('378.90'),
    'NVDA': Decimal('495.25')
}

# Transaction test data
SAMPLE_TRANSACTIONS = [
    {
        'txn_type': 'BUY',
        'symbol': 'AAPL',
        'quantity': 10,
        'price': Decimal('150.00'),
        'total_amount': Decimal('1500.00'),
        'commission_rate': Decimal('0.50')
    },
    {
        'txn_type': 'SELL',
        'symbol': 'AAPL',
        'quantity': 5,
        'price': Decimal('160.00'),
        'total_amount': Decimal('800.00'),
        'commission_rate': Decimal('0.50')
    },
    {
        'txn_type': 'BUY',
        'symbol': 'GOOGL',
        'quantity': 15,
        'price': Decimal('130.00'),
        'total_amount': Decimal('1950.00'),
        'commission_rate': Decimal('0.00')
    }
]

# Portfolio items test data
SAMPLE_PORTFOLIO_ITEMS = [
    {
        'symbol': 'AAPL',
        'quantity': 10,
        'average_buy_price': Decimal('150.00')
    },
    {
        'symbol': 'GOOGL',
        'quantity': 15,
        'average_buy_price': Decimal('130.00')
    },
    {
        'symbol': 'TSLA',
        'quantity': 8,
        'average_buy_price': Decimal('220.00')
    }
]

# Dividend test data
SAMPLE_DIVIDENDS = [
    {
        'symbol': 'AAPL',
        'amount_per_share': Decimal('0.24'),
        'quantity': 10
    },
    {
        'symbol': 'MSFT',
        'amount_per_share': Decimal('0.68'),
        'quantity': 20
    }
]

# Sentiment analysis test data
SAMPLE_SENTIMENT_DATA = {
    'positive': {
        'polarity': 0.65,
        'articles': [
            'Stock hits all-time high on strong earnings',
            'Company announces major product innovation',
            'Analysts upgrade price target significantly'
        ],
        'scores': [0.8, 0.6, 0.55],
        'pos': 3,
        'neg': 0,
        'neutral': 0
    },
    'negative': {
        'polarity': -0.55,
        'articles': [
            'Stock plunges on disappointing earnings',
            'CEO resignation shocks investors',
            'Regulatory concerns weigh on stock'
        ],
        'scores': [-0.7, -0.5, -0.45],
        'pos': 0,
        'neg': 3,
        'neutral': 0
    },
    'neutral': {
        'polarity': 0.05,
        'articles': [
            'Company holds steady in volatile market',
            'Quarterly results meet expectations',
            'Stock trades sideways amid uncertainty'
        ],
        'scores': [0.1, 0.0, 0.05],
        'pos': 1,
        'neg': 0,
        'neutral': 2
    }
}

# ML model test data
ML_MODEL_TEST_PARAMS = {
    'lstm': {
        'timesteps': 7,
        'epochs': 25,
        'batch_size': 32,
        'units': 50,
        'dropout': 0.1,
        'num_layers': 4
    },
    'arima': {
        'order': (6, 1, 0),
        'train_test_split': 0.8
    },
    'linear_regression': {
        'forecast_days': 7,
        'adjustment_factor': 1.04,
        'train_test_split': 0.8
    }
}

# Valid and invalid form data
VALID_REGISTRATION_DATA = {
    'email': 'newuser@example.com',
    'username': 'newuser',
    'password': 'ValidPass123!',
    'confirm_password': 'ValidPass123!'
}

INVALID_REGISTRATION_DATA = [
    {
        'email': '',
        'username': 'testuser',
        'password': 'Pass123!',
        'confirm_password': 'Pass123!',
        'error': 'Email required'
    },
    {
        'email': 'test@example.com',
        'username': '',
        'password': 'Pass123!',
        'confirm_password': 'Pass123!',
        'error': 'Username required'
    },
    {
        'email': 'test@example.com',
        'username': 'testuser',
        'password': 'Pass123!',
        'confirm_password': 'DifferentPass123!',
        'error': 'Passwords do not match'
    },
    {
        'email': 'invalid-email',
        'username': 'testuser',
        'password': 'Pass123!',
        'confirm_password': 'Pass123!',
        'error': 'Invalid email format'
    }
]

# API response mock data
MOCK_YFINANCE_DATA = {
    'AAPL': {
        'Open': [150.0, 151.0, 152.0, 151.5, 153.0],
        'High': [152.0, 153.0, 154.0, 153.5, 155.0],
        'Low': [149.0, 150.0, 151.0, 150.5, 152.0],
        'Close': [151.0, 152.0, 153.0, 152.5, 154.0],
        'Volume': [1000000, 1100000, 1050000, 1075000, 1125000],
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    }
}

MOCK_ALPHA_VANTAGE_RESPONSE = {
    'Time Series (Daily)': {
        '2024-01-05': {
            '1. open': '153.00',
            '2. high': '155.00',
            '3. low': '152.00',
            '4. close': '154.00',
            '5. volume': '1125000'
        }
    }
}

MOCK_FINVIZ_NEWS = [
    {
        'title': 'AAPL: Strong iPhone Sales Drive Revenue Growth',
        'link': 'https://finviz.com/news/1',
        'date': 'Dec 05, 2024'
    },
    {
        'title': 'Apple Announces New AI Features',
        'link': 'https://finviz.com/news/2',
        'date': 'Dec 04, 2024'
    }
]

# Security test data
SQL_INJECTION_ATTEMPTS = [
    "admin' OR '1'='1",
    "'; DROP TABLE users; --",
    "1' UNION SELECT * FROM users--"
]

XSS_ATTEMPTS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "javascript:alert('XSS')"
]

# Performance test data
LOAD_TEST_SCENARIOS = {
    'light': {
        'num_users': 10,
        'num_requests': 100,
        'duration_seconds': 60
    },
    'medium': {
        'num_users': 50,
        'num_requests': 500,
        'duration_seconds': 300
    },
    'heavy': {
        'num_users': 100,
        'num_requests': 1000,
        'duration_seconds': 600
    }
}
