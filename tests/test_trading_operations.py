"""
Phase 4: Unit Tests for Trading Operations

Tests for buy/sell transactions, commission calculations, and broker integration.
"""

import pytest
from decimal import Decimal

from main import db, User, Company, Broker, PortfolioItem, Transaction


class TestStockPurchase:
    """Test cases for stock buy operations."""
    
    def test_successful_stock_purchase(self, authenticated_client, test_db, sample_user, mock_stock_price):
        """Test successful stock purchase."""
        initial_balance = sample_user.wallet_balance
        
        data = {
            'symbol': 'AAPL',
            'quantity': '10'
        }
        response = authenticated_client.post('/trade/buy', data=data, follow_redirects=True)
        
        assert response.status_code == 200
        test_db.session.refresh(sample_user)
        
        # Check wallet balance decreased
        assert sample_user.wallet_balance < initial_balance
        
        # Check portfolio item created
        company = Company.query.filter_by(symbol='AAPL').first()
        portfolio_item = PortfolioItem.query.filter_by(
            user_id=sample_user.id,
            company_id=company.id
        ).first()
        assert portfolio_item is not None
        assert portfolio_item.quantity == 10
    
    def test_purchase_insufficient_funds(self, authenticated_client, test_db, sample_user, mock_stock_price):
        """Test stock purchase with insufficient funds."""
        # Set low balance
        sample_user.wallet_balance = Decimal('10.00')
        test_db.session.commit()
        
        data = {
            'symbol': 'AAPL',
            'quantity': '1000'  # Too many shares
        }
        response = authenticated_client.post('/trade/buy', data=data, follow_redirects=True)
        
        assert b'insufficient' in response.data.lower()
    
    def test_purchase_creates_transaction_record(self, authenticated_client, test_db, sample_user, mock_stock_price):
        """Test that purchase creates transaction record."""
        initial_txn_count = Transaction.query.filter_by(user_id=sample_user.id).count()
        
        data = {
            'symbol': 'AAPL',
            'quantity': '5'
        }
        authenticated_client.post('/trade/buy', data=data, follow_redirects=True)
        
        new_txn_count = Transaction.query.filter_by(user_id=sample_user.id).count()
        assert new_txn_count == initial_txn_count + 1
        
        # Check transaction details
        txn = Transaction.query.filter_by(user_id=sample_user.id, txn_type='BUY').first()
        assert txn is not None
        assert txn.quantity == 5
    
    def test_purchase_invalid_quantity(self, authenticated_client, test_db, sample_user):
        """Test purchase with invalid quantity."""
        data = {
            'symbol': 'AAPL',
            'quantity': 'invalid'
        }
        response = authenticated_client.post('/trade/buy', data=data, follow_redirects=True)
        
        assert b'integer' in response.data.lower() or b'invalid' in response.data.lower()
    
    def test_purchase_negative_quantity(self, authenticated_client, test_db, sample_user):
        """Test purchase with negative quantity."""
        data = {
            'symbol': 'AAPL',
            'quantity': '-10'
        }
        response = authenticated_client.post('/trade/buy', data=data, follow_redirects=True)
        
        assert b'greater than zero' in response.data.lower()
    
    def test_purchase_zero_quantity(self, authenticated_client, test_db, sample_user):
        """Test purchase with zero quantity."""
        data = {
            'symbol': 'AAPL',
            'quantity': '0'
        }
        response = authenticated_client.post('/trade/buy', data=data, follow_redirects=True)
        
        assert b'greater than zero' in response.data.lower()
    
    def test_purchase_creates_company_if_not_exists(self, authenticated_client, test_db, sample_user, mock_stock_price):
        """Test that purchasing creates company if it doesn't exist."""
        # Ensure company doesn't exist
        Company.query.filter_by(symbol='NEWSTOCK').delete()
        test_db.session.commit()
        
        data = {
            'symbol': 'NEWSTOCK',
            'quantity': '10'
        }
        authenticated_client.post('/trade/buy', data=data, follow_redirects=True)
        
        company = Company.query.filter_by(symbol='NEWSTOCK').first()
        assert company is not None
    
    def test_purchase_updates_existing_portfolio(self, authenticated_client, test_db, sample_portfolio_item, mock_stock_price):
        """Test that purchasing more of owned stock updates portfolio."""
        initial_quantity = sample_portfolio_item.quantity
        initial_avg_price = sample_portfolio_item.average_buy_price
        
        data = {
            'symbol': sample_portfolio_item.company.symbol,
            'quantity': '5'
        }
        authenticated_client.post('/trade/buy', data=data, follow_redirects=True)
        
        test_db.session.refresh(sample_portfolio_item)
        assert sample_portfolio_item.quantity == initial_quantity + 5
        # Average price should be recalculated
        assert sample_portfolio_item.average_buy_price != initial_avg_price


class TestStockSale:
    """Test cases for stock sell operations."""
    
    def test_successful_stock_sale(self, authenticated_client, test_db, sample_user, sample_portfolio_item, mock_stock_price):
        """Test successful stock sale."""
        initial_balance = sample_user.wallet_balance
        initial_quantity = sample_portfolio_item.quantity
        
        data = {
            'symbol': sample_portfolio_item.company.symbol,
            'quantity': '5'
        }
        response = authenticated_client.post('/trade/sell', data=data, follow_redirects=True)
        
        assert response.status_code == 200
        test_db.session.refresh(sample_user)
        test_db.session.refresh(sample_portfolio_item)
        
        # Wallet balance should increase
        assert sample_user.wallet_balance > initial_balance
        # Quantity should decrease
        assert sample_portfolio_item.quantity == initial_quantity - 5
    
    def test_sell_insufficient_shares(self, authenticated_client, test_db, sample_portfolio_item):
        """Test selling more shares than owned."""
        data = {
            'symbol': sample_portfolio_item.company.symbol,
            'quantity': str(sample_portfolio_item.quantity + 100)
        }
        response = authenticated_client.post('/trade/sell', data=data, follow_redirects=True)
        
        assert b'not enough' in response.data.lower() or b'insufficient' in response.data.lower()
    
    def test_sell_removes_portfolio_item_on_zero(self, authenticated_client, test_db, sample_portfolio_item, mock_stock_price):
        """Test that selling all shares removes portfolio item."""
        portfolio_id = sample_portfolio_item.id
        quantity = sample_portfolio_item.quantity
        
        data = {
            'symbol': sample_portfolio_item.company.symbol,
            'quantity': str(quantity)
        }
        authenticated_client.post('/trade/sell', data=data, follow_redirects=True)
        
        # Portfolio item should be deleted
        deleted_item = PortfolioItem.query.get(portfolio_id)
        assert deleted_item is None
    
    def test_sell_nonexistent_stock(self, authenticated_client, test_db):
        """Test selling stock not in portfolio."""
        data = {
            'symbol': 'NONEXISTENT',
            'quantity': '10'
        }
        response = authenticated_client.post('/trade/sell', data=data, follow_redirects=True)
        
        assert b'no holdings' in response.data.lower()
    
    def test_sell_creates_transaction_record(self, authenticated_client, test_db, sample_portfolio_item, mock_stock_price):
        """Test that sale creates transaction record."""
        data = {
            'symbol': sample_portfolio_item.company.symbol,
            'quantity': '5'
        }
        authenticated_client.post('/trade/sell', data=data, follow_redirects=True)
        
        txn = Transaction.query.filter_by(txn_type='SELL').first()
        assert txn is not None
        assert txn.quantity == 5


class TestCommissionCalculations:
    """Test cases for broker commission calculations."""
    
    def test_commission_calculation(self, test_db, sample_broker):
        """Test commission calculation."""
        from main import calculate_commission
        
        total_amount = Decimal('1000.00')
        commission = calculate_commission(total_amount, sample_broker)
        
        # 0.50% of $1000 = $5.00
        expected = Decimal('5.00')
        assert commission == expected
    
    def test_commission_zero_rate(self, test_db):
        """Test commission with zero rate broker."""
        from main import calculate_commission
        
        broker = Broker(name='Zero Commission', commission_rate=Decimal('0.00'))
        test_db.session.add(broker)
        test_db.session.commit()
        
        total_amount = Decimal('1000.00')
        commission = calculate_commission(total_amount, broker)
        
        assert commission == Decimal('0.00')
    
    def test_commission_deducted_on_buy(self, authenticated_client, test_db, sample_user, sample_broker, mock_stock_price):
        """Test that commission is deducted on buy."""
        initial_balance = sample_user.wallet_balance
        
        data = {
            'symbol': 'AAPL',
            'quantity': '10'
        }
        authenticated_client.post('/trade/buy', data=data, follow_redirects=True)
        
        test_db.session.refresh(sample_user)
        
        # Balance should decrease by (price * quantity + commission)
        assert sample_user.wallet_balance < initial_balance
        
        # Check transaction has commission
        txn = Transaction.query.filter_by(txn_type='BUY').first()
        assert txn.commission_amount > 0
    
    def test_commission_deducted_on_sell(self, authenticated_client, test_db, sample_user, sample_portfolio_item, sample_broker, mock_stock_price):
        """Test that commission is deducted on sell."""
        initial_balance = sample_user.wallet_balance
        
        data = {
            'symbol': sample_portfolio_item.company.symbol,
            'quantity': '5'
        }
        authenticated_client.post('/trade/sell', data=data, follow_redirects=True)
        
        test_db.session.refresh(sample_user)
        
        # Check transaction has commission
        txn = Transaction.query.filter_by(txn_type='SELL').order_by(Transaction.created_at.desc()).first()
        assert txn.commission_amount > 0


class TestBrokerIntegration:
    """Test cases for broker management."""
    
    def test_get_active_broker(self, test_db, sample_broker):
        """Test getting active broker."""
        from main import get_active_broker
        
        broker = get_active_broker()
        assert broker is not None
        assert broker.is_active is True
    
    def test_transaction_records_broker(self, authenticated_client, test_db, sample_broker, mock_stock_price):
        """Test that transactions record broker information."""
        data = {
            'symbol': 'AAPL',
            'quantity': '10'
        }
        authenticated_client.post('/trade/buy', data=data, follow_redirects=True)
        
        txn = Transaction.query.filter_by(txn_type='BUY').first()
        assert txn.broker_id == sample_broker.id
    
    def test_admin_can_add_broker(self, admin_client, test_db):
        """Test admin can add new broker."""
        data = {
            'name': 'New Broker',
            'email': 'new@broker.com',
            'commission_rate': '0.75'
        }
        response = admin_client.post('/admin/brokers', data=data, follow_redirects=True)
        
        assert response.status_code == 200
        broker = Broker.query.filter_by(name='New Broker').first()
        assert broker is not None
        assert broker.commission_rate == Decimal('0.75')


class TestLatestPriceFetching:
    """Test cases for fetching latest stock prices."""
    
    def test_get_latest_price_mock(self, mock_stock_price):
        """Test getting latest stock price (mocked)."""
        from main import get_latest_close_price
        
        price = get_latest_close_price('AAPL')
        assert price is not None
        assert price > 0
    
    def test_get_latest_price_invalid_symbol(self, mock_stock_price):
        """Test getting price for invalid symbol."""
        from main import get_latest_close_price
        
        # Mock returns a default price for unknown symbols
        price = get_latest_close_price('INVALID123')
        assert price is not None  # Mock returns default


class TestTransactionHistory:
    """Test cases for transaction history tracking."""
    
    def test_transaction_has_timestamp(self, test_db, sample_transactions):
        """Test that transactions have timestamps."""
        for txn in sample_transactions:
            assert txn.created_at is not None
    
    def test_transaction_order(self, test_db, sample_transactions):
        """Test transactions are ordered by date."""
        transactions = Transaction.query.order_by(Transaction.created_at.desc()).all()
        
        for i in range(len(transactions) - 1):
            assert transactions[i].created_at >= transactions[i + 1].created_at
    
    def test_user_transaction_history(self, authenticated_client, sample_user):
        """Test viewing user transaction history on dashboard."""
        response = authenticated_client.get('/dashboard')
        
        assert response.status_code == 200
        # Should contain transaction information
        assert b'transaction' in response.data.lower() or b'BUY' in response.data or b'SELL' in response.data
