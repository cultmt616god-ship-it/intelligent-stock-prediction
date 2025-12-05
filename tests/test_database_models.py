"""
Phase 1: Unit Tests for Database Models

Tests for SQLAlchemy models and database operations.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

from main import db, User, Company, Broker, PortfolioItem, Transaction, Dividend


class TestUserModel:
    """Test cases for the User model."""
    
    def test_user_creation(self, test_db):
        """Test creating a new user."""
        user = User(
            email='test@example.com',
            username='testuser',
            password_hash=generate_password_hash('password123'),
            role='user',
            wallet_balance=Decimal('1000.00')
        )
        test_db.session.add(user)
        test_db.session.commit()
        
        assert user.id is not None
        assert user.email == 'test@example.com'
        assert user.username == 'testuser'
        assert user.role == 'user'
        assert user.wallet_balance == Decimal('1000.00')
        assert user.is_active is True
        assert user.created_at is not None
    
    def test_user_password_hashing(self, test_db):
        """Test password hashing and verification."""
        password = 'SecurePassword123!'
        user = User(
            email='test@example.com',
            username='testuser',
            password_hash=generate_password_hash(password),
            role='user'
        )
        test_db.session.add(user)
        test_db.session.commit()
        
        # Verify password works
        assert user.check_password(password) is True
        # Verify wrong password fails
        assert user.check_password('WrongPassword') is False
    
    def test_user_unique_email(self, test_db):
        """Test that email must be unique."""
        user1 = User(
            email='duplicate@example.com',
            username='user1',
            password_hash=generate_password_hash('pass1'),
            role='user'
        )
        test_db.session.add(user1)
        test_db.session.commit()
        
        user2 = User(
            email='duplicate@example.com',
            username='user2',
            password_hash=generate_password_hash('pass2'),
            role='user'
        )
        test_db.session.add(user2)
        
        with pytest.raises(Exception):  # IntegrityError
            test_db.session.commit()
    
    def test_user_unique_username(self, test_db):
        """Test that username must be unique."""
        user1 = User(
            email='user1@example.com',
            username='duplicate',
            password_hash=generate_password_hash('pass1'),
            role='user'
        )
        test_db.session.add(user1)
        test_db.session.commit()
        
        user2 = User(
            email='user2@example.com',
            username='duplicate',
            password_hash=generate_password_hash('pass2'),
            role='user'
        )
        test_db.session.add(user2)
        
        with pytest.raises(Exception):  # IntegrityError
            test_db.session.commit()
    
    def test_user_default_values(self, test_db):
        """Test default values for user model."""
        user = User(
            email='test@example.com',
            username='testuser',
            password_hash=generate_password_hash('password'),
            role='user'
        )
        test_db.session.add(user)
        test_db.session.commit()
        
        assert user.wallet_balance == Decimal('0')
        assert user.is_active is True
        assert user.last_login_at is None


class TestCompanyModel:
    """Test cases for the Company model."""
    
    def test_company_creation(self, test_db):
        """Test creating a new company."""
        company = Company(
            symbol='AAPL',
            name='Apple Inc.',
            exchange='NASDAQ',
            sector='Technology',
            is_active=True
        )
        test_db.session.add(company)
        test_db.session.commit()
        
        assert company.id is not None
        assert company.symbol == 'AAPL'
        assert company.name == 'Apple Inc.'
        assert company.exchange == 'NASDAQ'
        assert company.sector == 'Technology'
        assert company.is_active is True
    
    def test_company_unique_symbol(self, test_db):
        """Test that company symbol must be unique."""
        company1 = Company(symbol='AAPL', name='Apple Inc.')
        test_db.session.add(company1)
        test_db.session.commit()
        
        company2 = Company(symbol='AAPL', name='Another Apple')
        test_db.session.add(company2)
        
        with pytest.raises(Exception):  # IntegrityError
            test_db.session.commit()
    
    def test_company_default_active(self, test_db):
        """Test default is_active value."""
        company = Company(symbol='GOOGL', name='Alphabet Inc.')
        test_db.session.add(company)
        test_db.session.commit()
        
        assert company.is_active is True


class TestBrokerModel:
    """Test cases for the Broker model."""
    
    def test_broker_creation(self, test_db):
        """Test creating a new broker."""
        broker = Broker(
            name='Test Broker',
            email='broker@example.com',
            commission_rate=Decimal('0.50'),
            is_active=True
        )
        test_db.session.add(broker)
        test_db.session.commit()
        
        assert broker.id is not None
        assert broker.name == 'Test Broker'
        assert broker.email == 'broker@example.com'
        assert broker.commission_rate == Decimal('0.50')
        assert broker.is_active is True
    
    def test_broker_default_commission(self, test_db):
        """Test default commission rate."""
        broker = Broker(name='Test Broker')
        test_db.session.add(broker)
        test_db.session.commit()
        
        assert broker.commission_rate == Decimal('0')
    
    def test_broker_default_active(self, test_db):
        """Test default is_active value."""
        broker = Broker(name='Test Broker')
        test_db.session.add(broker)
        test_db.session.commit()
        
        assert broker.is_active is True


class TestPortfolioItemModel:
    """Test cases for the PortfolioItem model."""
    
    def test_portfolio_item_creation(self, test_db, sample_user, sample_company):
        """Test creating a new portfolio item."""
        portfolio_item = PortfolioItem(
            user_id=sample_user.id,
            company_id=sample_company.id,
            quantity=10,
            average_buy_price=Decimal('150.00')
        )
        test_db.session.add(portfolio_item)
        test_db.session.commit()
        
        assert portfolio_item.id is not None
        assert portfolio_item.user_id == sample_user.id
        assert portfolio_item.company_id == sample_company.id
        assert portfolio_item.quantity == 10
        assert portfolio_item.average_buy_price == Decimal('150.00')
        assert portfolio_item.created_at is not None
    
    def test_portfolio_item_relationships(self, test_db, sample_user, sample_company):
        """Test portfolio item relationships."""
        portfolio_item = PortfolioItem(
            user_id=sample_user.id,
            company_id=sample_company.id,
            quantity=10,
            average_buy_price=Decimal('150.00')
        )
        test_db.session.add(portfolio_item)
        test_db.session.commit()
        
        assert portfolio_item.user.email == sample_user.email
        assert portfolio_item.company.symbol == sample_company.symbol
    
    def test_portfolio_item_cascade_delete(self, test_db, sample_user, sample_company):
        """Test cascade delete when user is deleted."""
        portfolio_item = PortfolioItem(
            user_id=sample_user.id,
            company_id=sample_company.id,
            quantity=10,
            average_buy_price=Decimal('150.00')
        )
        test_db.session.add(portfolio_item)
        test_db.session.commit()
        
        portfolio_id = portfolio_item.id
        test_db.session.delete(sample_user)
        test_db.session.commit()
        
        deleted_item = PortfolioItem.query.get(portfolio_id)
        assert deleted_item is None


class TestTransactionModel:
    """Test cases for the Transaction model."""
    
    def test_transaction_creation(self, test_db, sample_user, sample_company, sample_broker):
        """Test creating a new transaction."""
        transaction = Transaction(
            user_id=sample_user.id,
            company_id=sample_company.id,
            txn_type='BUY',
            quantity=10,
            price=Decimal('150.00'),
            total_amount=Decimal('1500.00'),
            commission_amount=Decimal('7.50'),
            broker_id=sample_broker.id,
            description='Test transaction'
        )
        test_db.session.add(transaction)
        test_db.session.commit()
        
        assert transaction.id is not None
        assert transaction.user_id == sample_user.id
        assert transaction.company_id == sample_company.id
        assert transaction.txn_type == 'BUY'
        assert transaction.quantity == 10
        assert transaction.price == Decimal('150.00')
        assert transaction.total_amount == Decimal('1500.00')
        assert transaction.commission_amount == Decimal('7.50')
        assert transaction.created_at is not None
    
    def test_transaction_relationships(self, test_db, sample_user, sample_company, sample_broker):
        """Test transaction relationships."""
        transaction = Transaction(
            user_id=sample_user.id,
            company_id=sample_company.id,
            txn_type='BUY',
            quantity=10,
            price=Decimal('150.00'),
            total_amount=Decimal('1500.00'),
            commission_amount=Decimal('7.50'),
            broker_id=sample_broker.id
        )
        test_db.session.add(transaction)
        test_db.session.commit()
        
        assert transaction.user.email == sample_user.email
        assert transaction.company.symbol == sample_company.symbol
        assert transaction.broker.name == sample_broker.name


class TestDividendModel:
    """Test cases for the Dividend model."""
    
    def test_dividend_creation(self, test_db, sample_portfolio_item):
        """Test creating a new dividend record."""
        dividend = Dividend(
            portfolio_item_id=sample_portfolio_item.id,
            amount_per_share=Decimal('0.25'),
            total_amount=Decimal('2.50'),
            payable_date=datetime.now().date()
        )
        test_db.session.add(dividend)
        test_db.session.commit()
        
        assert dividend.id is not None
        assert dividend.portfolio_item_id == sample_portfolio_item.id
        assert dividend.amount_per_share == Decimal('0.25')
        assert dividend.total_amount == Decimal('2.50')
        assert dividend.created_at is not None
    
    def test_dividend_relationship(self, test_db, sample_portfolio_item):
        """Test dividend relationship with portfolio item."""
        dividend = Dividend(
            portfolio_item_id=sample_portfolio_item.id,
            amount_per_share=Decimal('0.25'),
            total_amount=Decimal('2.50')
        )
        test_db.session.add(dividend)
        test_db.session.commit()
        
        assert dividend.portfolio_item.quantity == sample_portfolio_item.quantity


class TestDatabaseConstraints:
    """Test database constraints and integrity."""
    
    def test_foreign_key_constraints(self, test_db, sample_user):
        """Test foreign key constraints."""
        # Try to create portfolio item with non-existent company
        portfolio_item = PortfolioItem(
            user_id=sample_user.id,
            company_id=99999,  # Non-existent
            quantity=10,
            average_buy_price=Decimal('150.00')
        )
        test_db.session.add(portfolio_item)
        
        with pytest.raises(Exception):  # IntegrityError
            test_db.session.commit()
    
    def test_not_null_constraints(self, test_db):
        """Test NOT NULL constraints."""
        # Try to create user without required fields
        user = User(role='user')  # Missing email and username
        test_db.session.add(user)
        
        with pytest.raises(Exception):  # IntegrityError
            test_db.session.commit()
