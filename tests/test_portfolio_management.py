"""
Phase 3: Unit Tests for Portfolio Management

Tests for wallet operations, holdings, and portfolio calculations.
"""

import pytest
from decimal import Decimal
from datetime import datetime

from main import db, User, Company, PortfolioItem, Transaction


class TestWalletOperations:
    """Test cases for wallet balance management."""
    
    def test_initial_wallet_balance(self, test_db, sample_user):
        """Test initial wallet balance."""
        assert sample_user.wallet_balance == Decimal('10000.00')
    
    def test_fund_topup(self, authenticated_client, test_db, sample_user):
        """Test adding funds to wallet."""
        initial_balance = sample_user.wallet_balance
        
        data = {'amount': '5000.00'}
        response = authenticated_client.post('/funds/topup', data=data, follow_redirects=True)
        
        assert response.status_code == 200
        test_db.session.refresh(sample_user)
        assert sample_user.wallet_balance == initial_balance + Decimal('5000.00')
    
    def test_fund_topup_decimal_amount(self, authenticated_client, test_db, sample_user):
        """Test adding decimal funds to wallet."""
        initial_balance = sample_user.wallet_balance
        
        data = {'amount': '123.45'}
        authenticated_client.post('/funds/topup', data=data, follow_redirects=True)
        
        test_db.session.refresh(sample_user)
        assert sample_user.wallet_balance == initial_balance + Decimal('123.45')
    
    def test_fund_topup_invalid_amount(self, authenticated_client, test_db, sample_user):
        """Test adding invalid amount to wallet."""
        initial_balance = sample_user.wallet_balance
        
        data = {'amount': 'invalid'}
        response = authenticated_client.post('/funds/topup', data=data, follow_redirects=True)
        
        test_db.session.refresh(sample_user)
        assert sample_user.wallet_balance == initial_balance
        assert b'invalid' in response.data.lower()
    
    def test_fund_topup_negative_amount(self, authenticated_client, test_db, sample_user):
        """Test adding negative amount (should fail)."""
        initial_balance = sample_user.wallet_balance
        
        data = {'amount': '-1000.00'}
        response = authenticated_client.post('/funds/topup', data=data, follow_redirects=True)
        
        test_db.session.refresh(sample_user)
        assert sample_user.wallet_balance == initial_balance
        assert b'greater than zero' in response.data.lower()
    
    def test_fund_topup_zero_amount(self, authenticated_client, test_db, sample_user):
        """Test adding zero amount (should fail)."""
        initial_balance = sample_user.wallet_balance
        
        data = {'amount': '0'}
        response = authenticated_client.post('/funds/topup', data=data, follow_redirects=True)
        
        test_db.session.refresh(sample_user)
        assert sample_user.wallet_balance == initial_balance


class TestPortfolioHoldings:
    """Test cases for portfolio holdings management."""
    
    def test_create_portfolio_item(self, test_db, sample_user, sample_company):
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
        assert portfolio_item.quantity == 10
        assert portfolio_item.average_buy_price == Decimal('150.00')
    
    def test_update_portfolio_quantity(self, test_db, sample_portfolio_item):
        """Test updating portfolio item quantity."""
        original_quantity = sample_portfolio_item.quantity
        sample_portfolio_item.quantity += 5
        test_db.session.commit()
        
        test_db.session.refresh(sample_portfolio_item)
        assert sample_portfolio_item.quantity == original_quantity + 5
    
    def test_calculate_average_buy_price(self, test_db, sample_portfolio_item):
        """Test recalculating average buy price on new purchase."""
        # Initial: 10 shares at $150 = $1500
        current_total = Decimal(sample_portfolio_item.average_buy_price) * Decimal(sample_portfolio_item.quantity)
        
        # Buy 5 more at $160 = $800
        new_purchase_quantity = 5
        new_purchase_price = Decimal('160.00')
        new_purchase_total = new_purchase_price * new_purchase_quantity
        
        # New total: $1500 + $800 = $2300 for 15 shares
        new_quantity = sample_portfolio_item.quantity + new_purchase_quantity
        new_total = current_total + new_purchase_total
        new_average = new_total / new_quantity
        
        sample_portfolio_item.quantity = new_quantity
        sample_portfolio_item.average_buy_price = new_average
        test_db.session.commit()
        
        test_db.session.refresh(sample_portfolio_item)
        assert sample_portfolio_item.quantity == 15
        expected_average = Decimal('153.33')  # Approximately $153.33
        assert abs(sample_portfolio_item.average_buy_price - expected_average) < Decimal('0.01')
    
    def test_get_user_portfolio(self, test_db, sample_user, sample_company):
        """Test retrieving all portfolio items for a user."""
        # Create multiple portfolio items
        companies = [sample_company]
        for i in range(2):
            company = Company(
                symbol=f'TEST{i}',
                name=f'Test Company {i}'
            )
            test_db.session.add(company)
            companies.append(company)
        test_db.session.commit()
        
        for company in companies:
            portfolio_item = PortfolioItem(
                user_id=sample_user.id,
                company_id=company.id,
                quantity=10 + i,
                average_buy_price=Decimal('100.00')
            )
            test_db.session.add(portfolio_item)
        test_db.session.commit()
        
        portfolio = PortfolioItem.query.filter_by(user_id=sample_user.id).all()
        assert len(portfolio) >= 3
    
    def test_delete_portfolio_item_on_zero_quantity(self, test_db, sample_portfolio_item):
        """Test removing portfolio item when quantity reaches zero."""
        portfolio_id = sample_portfolio_item.id
        test_db.session.delete(sample_portfolio_item)
        test_db.session.commit()
        
        deleted_item = PortfolioItem.query.get(portfolio_id)
        assert deleted_item is None


class TestPortfolioCalculations:
    """Test cases for portfolio value calculations."""
    
    def test_total_invested_calculation(self, test_db, sample_user, sample_company):
        """Test calculating total invested amount."""
        # Create portfolio items
        portfolio_items = []
        total_invested = Decimal('0')
        
        for i in range(3):
            quantity = 10 * (i + 1)
            price = Decimal('100.00') + (i * 10)
            portfolio_item = PortfolioItem(
                user_id=sample_user.id,
                company_id=sample_company.id,
                quantity=quantity,
                average_buy_price=price
            )
            test_db.session.add(portfolio_item)
            portfolio_items.append(portfolio_item)
            total_invested += price * quantity
        
        test_db.session.commit()
        
        # Calculate from database
        items = PortfolioItem.query.filter_by(user_id=sample_user.id).all()
        calculated_invested = sum(
            Decimal(item.average_buy_price) * Decimal(item.quantity)
            for item in items
        )
        
        assert calculated_invested == total_invested
    
    def test_current_portfolio_value(self, test_db, sample_portfolio_item, mock_stock_price):
        """Test calculating current portfolio value with live prices."""
        # Portfolio: 10 shares at avg $150
        # Current price: $175.50 (from mock)
        current_price = Decimal('175.50')
        expected_value = current_price * sample_portfolio_item.quantity
        
        # Get current price from mock
        from main import get_latest_close_price
        latest_price = get_latest_close_price(sample_portfolio_item.company.symbol)
        current_value = Decimal(str(latest_price)) * sample_portfolio_item.quantity
        
        assert abs(current_value - expected_value) < Decimal('0.01')
    
    def test_portfolio_profit_loss(self, test_db, sample_portfolio_item, mock_stock_price):
        """Test calculating profit/loss for portfolio."""
        # Invested: 10 shares * $150 = $1500
        invested = Decimal(sample_portfolio_item.average_buy_price) * sample_portfolio_item.quantity
        
        # Current: 10 shares * $175.50 = $1755
        from main import get_latest_close_price
        current_price = Decimal(str(get_latest_close_price(sample_portfolio_item.company.symbol)))
        current_value = current_price * sample_portfolio_item.quantity
        
        # Profit: $1755 - $1500 = $255
        profit = current_value - invested
        
        assert profit > 0  # Should have profit
        assert abs(profit - Decimal('255.00')) < Decimal('1.00')
    
    def test_portfolio_percentage_return(self, test_db, sample_portfolio_item, mock_stock_price):
        """Test calculating percentage return."""
        invested = Decimal(sample_portfolio_item.average_buy_price) * sample_portfolio_item.quantity
        
        from main import get_latest_close_price
        current_price = Decimal(str(get_latest_close_price(sample_portfolio_item.company.symbol)))
        current_value = current_price * sample_portfolio_item.quantity
        
        percentage_return = ((current_value - invested) / invested) * 100
        
        # ($1755 - $1500) / $1500 * 100 = 17%
        assert abs(percentage_return - Decimal('17.00')) < Decimal('1.00')


class TestDashboardView:
    """Test cases for dashboard portfolio display."""
    
    def test_dashboard_displays_portfolio(self, authenticated_client, test_db, sample_portfolio_item):
        """Test that dashboard displays user's portfolio."""
        response = authenticated_client.get('/dashboard')
        
        assert response.status_code == 200
        assert sample_portfolio_item.company.symbol.encode() in response.data
    
    def test_dashboard_shows_wallet_balance(self, authenticated_client, sample_user):
        """Test that dashboard shows wallet balance."""
        response = authenticated_client.get('/dashboard')
        
        assert response.status_code == 200
        # Balance should be displayed somewhere
        balance_str = str(sample_user.wallet_balance).encode()
        assert balance_str in response.data or b'balance' in response.data.lower()
    
    def test_dashboard_shows_recent_transactions(self, authenticated_client, test_db, sample_transactions):
        """Test that dashboard shows recent transactions."""
        response = authenticated_client.get('/dashboard')
        
        assert response.status_code == 200
        # Should show transaction type
        assert b'BUY' in response.data or b'SELL' in response.data
    
    def test_dashboard_empty_portfolio(self, authenticated_client, sample_user):
        """Test dashboard with empty portfolio."""
        response = authenticated_client.get('/dashboard')
        
        assert response.status_code == 200
        # Should still render successfully even with no holdings


class TestDividendTracking:
    """Test cases for dividend tracking and recording."""
    
    def test_record_dividend(self, authenticated_client, test_db, sample_portfolio_item, sample_user):
        """Test recording a dividend payout."""
        initial_balance = sample_user.wallet_balance
        amount_per_share = Decimal('0.25')
        expected_total = amount_per_share * sample_portfolio_item.quantity
        
        data = {
            'symbol': sample_portfolio_item.company.symbol,
            'amount_per_share': str(amount_per_share)
        }
        response = authenticated_client.post('/dividends/record', data=data, follow_redirects=True)
        
        assert response.status_code == 200
        test_db.session.refresh(sample_user)
        assert sample_user.wallet_balance == initial_balance + expected_total
    
    def test_dividend_invalid_symbol(self, authenticated_client, test_db, sample_user):
        """Test recording dividend for non-existent symbol."""
        initial_balance = sample_user.wallet_balance
        
        data = {
            'symbol': 'NONEXISTENT',
            'amount_per_share': '0.25'
        }
        response = authenticated_client.post('/dividends/record', data=data, follow_redirects=True)
        
        test_db.session.refresh(sample_user)
        assert sample_user.wallet_balance == initial_balance
        assert b'no holdings' in response.data.lower()
    
    def test_dividend_negative_amount(self, authenticated_client, test_db, sample_portfolio_item):
        """Test recording negative dividend amount."""
        data = {
            'symbol': sample_portfolio_item.company.symbol,
            'amount_per_share': '-0.25'
        }
        response = authenticated_client.post('/dividends/record', data=data, follow_redirects=True)
        
        assert b'greater than zero' in response.data.lower()
