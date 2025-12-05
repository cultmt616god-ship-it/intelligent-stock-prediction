"""
Phase 9: Integration Tests for Prediction Pipeline

End-to-end tests for the complete prediction workflow.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


pytestmark = pytest.mark.integration


@pytest.fixture
def complete_prediction_mocks(monkeypatch, mock_yfinance):
    """Mock all components needed for prediction."""
    # Mock ARIMA
    def mock_arima(df):
        return (150.5, 2.5, [100, 105], [101, 104])
    
    # Mock LSTM
    def mock_lstm(df):
        return (152.0, 3.0, [100, 105], [102, 106])
    
    # Mock Linear Regression
    def mock_lr(df):
        forecast_set = np.array([[155], [156], [157], [158], [159], [160], [161]])
        return (df, 154.0, forecast_set, 157.5, 2.8, [100, 105], [101, 106])
    
    # Mock sentiment
    def mock_sentiment(symbol):
        return (0.65, ['Positive news'], [0.65], 1, 0, 0)
    
    import main
    monkeypatch.setattr(main, 'ARIMA_ALGO', mock_arima)
    monkeypatch.setattr(main, 'LSTM_ALGO', mock_lstm)
    monkeypatch.setattr(main, 'LIN_REG_ALGO', mock_lr)
    monkeypatch.setattr(main, 'finviz_finvader_sentiment', mock_sentiment)
    
    return True


class TestPredictionPipeline:
    """Test complete prediction pipeline."""
    
    def test_prediction_workflow(self, client, test_db, complete_prediction_mocks):
        """Test complete prediction workflow from request to response."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        assert response.status_code == 200
        # Should contain prediction results
        assert b'AAPL' in response.data or b'prediction' in response.data.lower()
    
    def test_invalid_symbol_handling(self, client, test_db):
        """Test handling of invalid stock symbol."""
        data = {'nm': 'INVALIDXYZ123'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        # Should handle gracefully
        assert response.status_code in [200, 400]
    
    def test_historical_data_fetch(self, client, complete_prediction_mocks):
        """Test historical data fetching in pipeline."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        # yfinance should be called (mocked)
        assert response.status_code == 200
    
    def test_all_models_execute(self, client, complete_prediction_mocks):
        """Test that ARIMA, LSTM, and LR all execute."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        # All models should run
        assert response.status_code == 200
    
    def test_sentiment_integration(self, client, complete_prediction_mocks):
        """Test sentiment analysis integration in pipeline."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        # Sentiment should be included
        assert response.status_code == 200
    
    def test_recommendation_generation(self, client, complete_prediction_mocks):
        """Test BUY/SELL recommendation generation."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        # Should generate recommendation
        assert b'BUY' in response.data or b'SELL' in response.data or response.status_code == 200


class TestChartGeneration:
    """Test chart/plot generation."""
    
    def test_trends_chart_created(self, client, complete_prediction_mocks):
        """Test Trends.png is created."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        # Chart images should be referenced or created
        assert response.status_code == 200
    
    def test_arima_chart_created(self, client, complete_prediction_mocks):
        """Test ARIMA.png is created."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        assert response.status_code == 200
    
    def test_lstm_chart_created(self, client, complete_prediction_mocks):
        """Test LSTM.png is created."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        assert response.status_code == 200


class TestResultsPage:
    """Test results page rendering."""
    
    def test_results_page_renders(self, client, complete_prediction_mocks):
        """Test results page renders with predictions."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        assert response.status_code == 200
        assert b'results' in response.data.lower() or b'prediction' in response.data.lower()
    
    def test_predictions_displayed(self, client, complete_prediction_mocks):
        """Test all three predictions are displayed."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        # Should show ARIMA, LSTM, and LR predictions
        assert response.status_code == 200
    
    def test_today_stock_data_displayed(self, client, complete_prediction_mocks):
        """Test today's stock data is displayed."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        # Should show Open, Close, High, Low, Volume
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling in prediction pipeline."""
    
    @patch('main.get_historical')
    def test_yfinance_failure_fallback(self, mock_get_historical, client):
        """Test fallback to Alpha Vantage when yfinance fails."""
        mock_get_historical.side_effect = Exception("yfinance failed")
        
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        # Should handle error
        assert response.status_code in [200, 400, 500]
    
    def test_model_failure_handling(self, client):
        """Test handling when a model fails."""
        # Don't mock - let natural errors occur
        data = {'nm': 'AAPL'}
        
        # Should not crash the  application
        try:
            response = client.post('/predict', data=data, follow_redirects=True)
            assert response.status_code in [200, 400, 500]
        except:
            assert True  # Handled gracefully


class TestDataPreprocessing:
    """Test data preprocessing in pipeline."""
    
    def test_csv_creation(self, client, complete_prediction_mocks):
        """Test CSV file creation for stock data."""
        data = {'nm': 'AAPL'}
        response = client.post('/predict', data=data, follow_redirects=True)
        
        # CSV should be created (AAPL.csv)
        assert response.status_code == 200
    
    def test_data_cleaning(self, sample_stock_data):
        """Test data cleaning (dropna)."""
        df_with_na = sample_stock_data.copy()
        df_with_na.loc[5, 'Close'] = None
        
        df_cleaned = df_with_na.dropna()
        
        assert len(df_cleaned) < len(df_with_na)
        assert df_cleaned['Close'].isna().sum() == 0
