"""
Pytest configuration and shared fixtures for the Stock Market Prediction Web App.

This module provides common fixtures and configuration for all test modules.
"""

import pytest
import sys
import os
from decimal import Decimal
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app, db, User, Company, Broker, PortfolioItem, Transaction, Dividend
from werkzeug.security import generate_password_hash


@pytest.fixture(scope='session')
def test_app():
    """Create and configure a test Flask application instance."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'test-secret-key-for-testing-only'
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture(scope='function')
def client(test_app):
    """Create a test client for the Flask application."""
    return test_app.test_client()


@pytest.fixture(scope='function')
def test_db(test_app):
    """Create a fresh database for each test."""
    with test_app.app_context():
        db.create_all()
        yield db
        db.session.remove()
        db.drop_all()
        db.create_all()


@pytest.fixture
def sample_user(test_db):
    """Create a sample regular user for testing."""
    user = User(
        email='testuser@example.com',
        username='testuser',
        password_hash=generate_password_hash('TestPass123!'),
        role='user',
        wallet_balance=Decimal('10000.00'),
        is_active=True
    )
    test_db.session.add(user)
    test_db.session.commit()
    return user


@pytest.fixture
def sample_admin(test_db):
    """Create a sample admin user for testing."""
    admin = User(
        email='admin@example.com',
        username='admin',
        password_hash=generate_password_hash('AdminPass123!'),
        role='admin',
        wallet_balance=Decimal('50000.00'),
        is_active=True
    )
    test_db.session.add(admin)
    test_db.session.commit()
    return admin


@pytest.fixture
def sample_company(test_db):
    """Create a sample company for testing."""
    company = Company(
        symbol='AAPL',
        name='Apple Inc.',
        exchange='NASDAQ',
        sector='Technology',
        is_active=True
    )
    test_db.session.add(company)
    test_db.session.commit()
    return company


@pytest.fixture
def sample_broker(test_db):
    """Create a sample broker for testing."""
    broker = Broker(
        name='Test Broker',
        email='broker@example.com',
        commission_rate=Decimal('0.50'),
        is_active=True
    )
    test_db.session.add(broker)
    test_db.session.commit()
    return broker


@pytest.fixture
def sample_portfolio_item(test_db, sample_user, sample_company):
    """Create a sample portfolio item for testing."""
    portfolio_item = PortfolioItem(
        user_id=sample_user.id,
        company_id=sample_company.id,
        quantity=10,
        average_buy_price=Decimal('150.00')
    )
    test_db.session.add(portfolio_item)
    test_db.session.commit()
    return portfolio_item


@pytest.fixture
def authenticated_client(client, sample_user):
    """Create an authenticated test client with a logged-in user."""
    with client.session_transaction() as sess:
        sess['user_id'] = sample_user.id
        sess['user_role'] = sample_user.role
    return client


@pytest.fixture
def admin_client(client, sample_admin):
    """Create an authenticated test client with a logged-in admin."""
    with client.session_transaction() as sess:
        sess['user_id'] = sample_admin.id
        sess['user_role'] = sample_admin.role
    return client


@pytest.fixture
def mock_stock_price(monkeypatch):
    """Mock the get_latest_close_price function."""
    def mock_get_price(symbol):
        prices = {
            'AAPL': 175.50,
            'GOOGL': 142.30,
            'TSLA': 238.45,
            'MSFT': 378.90
        }
        return prices.get(symbol, 100.00)
    
    import main
    monkeypatch.setattr(main, 'get_latest_close_price', mock_get_price)
    return mock_get_price


@pytest.fixture
def sample_stock_data():
    """Provide sample stock data for ML model testing."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    data = {
        'Date': dates,
        'Open': np.random.uniform(140, 180, len(dates)),
        'High': np.random.uniform(150, 190, len(dates)),
        'Low': np.random.uniform(130, 170, len(dates)),
        'Close': np.random.uniform(140, 180, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_yfinance(monkeypatch, sample_stock_data):
    """Mock yfinance download function."""
    def mock_download(symbol, start, end):
        return sample_stock_data
    
    import yfinance as yf
    monkeypatch.setattr(yf, 'download', mock_download)


@pytest.fixture
def mock_sentiment_analysis(monkeypatch):
    """Mock sentiment analysis function."""
    def mock_finviz_sentiment(symbol):
        return (
            0.15,  # polarity
            ['Positive news about stock', 'Company reports growth'],  # sentiment_list
            [0.2, 0.1],  # sentiment_pol
            2,  # pos
            0,  # neg
            0   # neutral
        )
    
    import main
    monkeypatch.setattr(main, 'finviz_finvader_sentiment', mock_finviz_sentiment)
    return mock_finviz_sentiment


@pytest.fixture(autouse=True)
def reset_db_session(test_app):
    """Automatically reset database session after each test."""
    yield
    with test_app.app_context():
        db.session.remove()


@pytest.fixture
def sample_transactions(test_db, sample_user, sample_company, sample_broker):
    """Create sample transactions for testing."""
    transactions = []
    
    # Buy transaction
    buy_txn = Transaction(
        user_id=sample_user.id,
        company_id=sample_company.id,
        txn_type='BUY',
        quantity=10,
        price=Decimal('150.00'),
        total_amount=Decimal('1500.00'),
        commission_amount=Decimal('7.50'),
        broker_id=sample_broker.id,
        description='Test buy transaction'
    )
    test_db.session.add(buy_txn)
    transactions.append(buy_txn)
    
    # Sell transaction
    sell_txn = Transaction(
        user_id=sample_user.id,
        company_id=sample_company.id,
        txn_type='SELL',
        quantity=5,
        price=Decimal('160.00'),
        total_amount=Decimal('800.00'),
        commission_amount=Decimal('4.00'),
        broker_id=sample_broker.id,
        description='Test sell transaction'
    )
    test_db.session.add(sell_txn)
    transactions.append(sell_txn)
    
    test_db.session.commit()
    return transactions


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "ml: marks tests for machine learning models"
    )
    config.addinivalue_line(
        "markers", "security: marks tests for security features"
    )
