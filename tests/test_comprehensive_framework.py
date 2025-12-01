#!/usr/bin/env python3
"""
Comprehensive Testing Framework for news_sentiment.py

This test suite provides complete coverage for all functionality in the 
news_sentiment.py module, including:

1. Core sentiment analysis functionality
2. Source-specific implementations
3. Use case configurations
4. Advanced features (batch processing, hybrid scoring, custom lexicons)
5. Error handling and monitoring
6. Fallback mechanisms
7. Performance optimizations
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import warnings
import tempfile
import json
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_sentiment import (
    ComprehensiveSentimentAnalyzer,
    retrieving_news_polarity,
    # Source-specific functions
    finviz_finvader_sentiment,
    eodhd_sentiment,
    alpha_vantage_sentiment,
    reddit_sentiment,
    social_sentiment,
    google_news_sentiment,
    # Use case functions
    hft_sentiment,
    retail_sentiment,
    quant_sentiment,
    academic_sentiment,
    fintech_sentiment,
    # Advanced features
    batch_sentiment_analysis,
    hybrid_sentiment_analysis,
    custom_lexicon_sentiment,
    # Error handling
    robust_finvader_analysis,
    log_sentiment_distribution,
    # Enums
    SentimentSource,
    UseCase
)


class TestCoreFunctionality(unittest.TestCase):
    """Test core sentiment analysis functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.analyzer = ComprehensiveSentimentAnalyzer(num_articles=5)
    
    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly"""
        self.assertIsInstance(self.analyzer, ComprehensiveSentimentAnalyzer)
        self.assertEqual(self.analyzer.num_articles, 5)
        self.assertIn(SentimentSource.ALL_SOURCES, self.analyzer.selected_sources)
    
    def test_sentiment_source_enum(self):
        """Test that all sentiment sources are defined"""
        expected_sources = [
            SentimentSource.FINVIZ_FINVADER,
            SentimentSource.EODHD_API,
            SentimentSource.ALPHA_VANTAGE,
            SentimentSource.TRADESTIE_REDDIT,
            SentimentSource.FINNHUB_SOCIAL,
            SentimentSource.STOCKGEIST,
            SentimentSource.GOOGLE_NEWS,
            SentimentSource.ALL_SOURCES
        ]
        
        for source in expected_sources:
            self.assertIn(source, SentimentSource)
    
    def test_use_case_enum(self):
        """Test that all use cases are defined"""
        expected_cases = [
            UseCase.HIGH_FREQUENCY_TRADING,
            UseCase.RETAIL_TRADING_APPS,
            UseCase.QUANT_HEDGE_FUNDS,
            UseCase.ACADEMIC_RESEARCH,
            UseCase.FINTECH_STARTUPS
        ]
        
        for case in expected_cases:
            self.assertIn(case, UseCase)


class TestSourceSpecificFunctions(unittest.TestCase):
    """Test source-specific sentiment analysis functions"""
    
    @patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment')
    def test_finviz_finvader_sentiment(self, mock_get_sentiment):
        """Test Finviz + FinVADER sentiment function"""
        mock_get_sentiment.return_value = (0.5, ['Title1'], 'Positive', 1, 0, 0)
        
        result = finviz_finvader_sentiment('AAPL', 5)
        
        self.assertEqual(result, (0.5, ['Title1'], 'Positive', 1, 0, 0))
        mock_get_sentiment.assert_called_once_with('AAPL', 'AAPL')
    
    @patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment')
    def test_google_news_sentiment(self, mock_get_sentiment):
        """Test Google News RSS sentiment function"""
        mock_get_sentiment.return_value = (-0.3, ['Title1'], 'Negative', 0, 1, 0)
        
        result = google_news_sentiment('AAPL', 3)
        
        self.assertEqual(result, (-0.3, ['Title1'], 'Negative', 0, 1, 0))
        mock_get_sentiment.assert_called_once_with('AAPL', 'AAPL')


class TestUseCaseConfigurations(unittest.TestCase):
    """Test use case-based sentiment analysis configurations"""
    
    def test_hft_sentiment_configuration(self):
        """Test High-Frequency Trading configuration"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment') as mock_get_sentiment:
            mock_get_sentiment.return_value = (0.2, [], 'Neutral', 0, 0, 1)
            
            result = hft_sentiment('AAPL', 10)
            
            # Check that the function was called
            self.assertEqual(result, (0.2, [], 'Neutral', 0, 0, 1))
    
    def test_retail_sentiment_configuration(self):
        """Test Retail Trading Apps configuration"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment') as mock_get_sentiment:
            mock_get_sentiment.return_value = (0.1, [], 'Neutral', 0, 0, 1)
            
            result = retail_sentiment('AAPL', 5)
            
            self.assertEqual(result, (0.1, [], 'Neutral', 0, 0, 1))
    
    def test_quant_sentiment_configuration(self):
        """Test Quant Hedge Funds configuration"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment') as mock_get_sentiment:
            mock_get_sentiment.return_value = (0.8, [], 'Positive', 1, 0, 0)
            
            result = quant_sentiment('AAPL', 20, 'test_key')
            
            self.assertEqual(result, (0.8, [], 'Positive', 1, 0, 0))
    
    def test_academic_sentiment_configuration(self):
        """Test Academic Research configuration"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment') as mock_get_sentiment:
            mock_get_sentiment.return_value = (-0.4, [], 'Negative', 0, 1, 0)
            
            result = academic_sentiment('AAPL', 50)
            
            self.assertEqual(result, (-0.4, [], 'Negative', 0, 1, 0))
    
    def test_fintech_sentiment_configuration(self):
        """Test Fintech Startups configuration"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment') as mock_get_sentiment:
            mock_get_sentiment.return_value = (0.3, [], 'Positive', 1, 0, 0)
            
            result = fintech_sentiment('AAPL', 15)
            
            self.assertEqual(result, (0.3, [], 'Positive', 1, 0, 0))


class TestAdvancedFeatures(unittest.TestCase):
    """Test advanced sentiment analysis features"""
    
    def test_batch_sentiment_analysis(self):
        """Test batch processing functionality"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.batch_process_sentiments') as mock_batch:
            import pandas as pd
            mock_df = pd.DataFrame({'symbol': ['AAPL', 'GOOGL'], 'sentiment': [0.5, -0.2]})
            mock_batch.return_value = mock_df
            
            result = batch_sentiment_analysis(['AAPL', 'GOOGL'], 10)
            
            self.assertIsInstance(result, pd.DataFrame)
            mock_batch.assert_called_once()
    
    def test_hybrid_sentiment_analysis(self):
        """Test hybrid scoring functionality"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.hybrid_sentiment') as mock_hybrid:
            mock_hybrid.return_value = {
                'raw_finvader': 0.5,
                'raw_api': 0.3,
                'hybrid': 0.4,
                'confidence': 0.4
            }
            
            result = hybrid_sentiment_analysis(0.6, "Test text", 0.7)
            
            self.assertIsInstance(result, dict)
            self.assertIn('hybrid', result)
            mock_hybrid.assert_called_once()
    
    def test_custom_lexicon_sentiment(self):
        """Test custom lexicon functionality"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.analyze_with_custom_lexicon') as mock_lexicon:
            mock_lexicon.return_value = {'compound': 0.5, 'pos': 0.3, 'neu': 0.2, 'neg': 0.5}
            
            result = custom_lexicon_sentiment("Test text", {"custom": 1.5})
            
            self.assertIsInstance(result, dict)
            self.assertIn('compound', result)
            mock_lexicon.assert_called_once()


class TestErrorHandlingAndMonitoring(unittest.TestCase):
    """Test error handling and monitoring features"""
    
    def test_robust_finvader_analysis(self):
        """Test robust FinVADER with retries"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.robust_finvader') as mock_finvader:
            mock_finvader.return_value = {'compound': 0.5, 'pos': 0.3, 'neu': 0.2, 'neg': 0.5}
            
            result = robust_finvader_analysis("Test text")
            
            self.assertIsInstance(result, dict)
            self.assertIn('compound', result)
            mock_finvader.assert_called_once_with("Test text")
    
    def test_log_sentiment_distribution(self):
        """Test sentiment distribution logging"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.log_sentiment_distribution') as mock_log:
            test_scores = [{'compound': 0.5}, {'compound': -0.3}, {'compound': 0.0}]
            
            log_sentiment_distribution(test_scores)
            
            mock_log.assert_called_once()


class TestFallbackMechanisms(unittest.TestCase):
    """Test fallback mechanisms between different sources"""
    
    def test_should_use_source(self):
        """Test source selection logic"""
        analyzer = ComprehensiveSentimentAnalyzer(
            num_articles=5,
            selected_sources=[SentimentSource.FINVIZ_FINVADER, SentimentSource.GOOGLE_NEWS]
        )
        
        # Test sources that should be used
        self.assertTrue(analyzer._should_use_source(SentimentSource.FINVIZ_FINVADER))
        self.assertTrue(analyzer._should_use_source(SentimentSource.GOOGLE_NEWS))
        
        # Test sources that should not be used
        self.assertFalse(analyzer._should_use_source(SentimentSource.EODHD_API))
        self.assertFalse(analyzer._should_use_source(SentimentSource.ALPHA_VANTAGE))
    
    def test_all_sources_selection(self):
        """Test ALL_SOURCES selection"""
        analyzer = ComprehensiveSentimentAnalyzer(
            num_articles=5,
            selected_sources=[SentimentSource.ALL_SOURCES]
        )
        
        # All sources should be used when ALL_SOURCES is selected
        self.assertTrue(analyzer._should_use_source(SentimentSource.FINVIZ_FINVADER))
        self.assertTrue(analyzer._should_use_source(SentimentSource.EODHD_API))
        self.assertTrue(analyzer._should_use_source(SentimentSource.ALPHA_VANTAGE))


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimization features"""
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        analyzer = ComprehensiveSentimentAnalyzer()
        key = analyzer._get_cache_key("AAPL", "Test text")
        
        self.assertIsInstance(key, str)
        self.assertIn("AAPL", key)
    
    def test_cache_operations(self):
        """Test cache get/set operations"""
        analyzer = ComprehensiveSentimentAnalyzer()
        
        # Test with Redis not available (should not crash)
        key = "test_key"
        value = {"test": "data"}
        
        # These should not raise exceptions even if Redis is not available
        analyzer._set_in_cache(key, value)
        result = analyzer._get_from_cache(key)
        
        # When Redis is not available, should return None
        self.assertIsNone(result)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and end-to-end functionality"""
    
    @patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment')
    def test_retrieving_news_polarity(self, mock_get_sentiment):
        """Test the main sentiment retrieval function"""
        mock_get_sentiment.return_value = (0.25, ['Title1', 'Title2'], 'Positive', 1, 0, 1)
        
        result = retrieving_news_polarity('AAPL', 5)
        
        self.assertEqual(result, (0.25, ['Title1', 'Title2'], 'Positive', 1, 0, 1))
        mock_get_sentiment.assert_called_once()
    
    @patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment')
    def test_retrieving_news_polarity_with_company_name(self, mock_get_sentiment):
        """Test sentiment retrieval with company name resolution"""
        # Mock the CSV reading to return a company name
        with patch('pandas.read_csv') as mock_read_csv:
            import pandas as pd
            mock_df = pd.DataFrame({
                'Ticker': ['AAPL'],
                'Name': ['Apple Inc']
            })
            mock_read_csv.return_value = mock_df
            
            mock_get_sentiment.return_value = (0.3, ['Title1'], 'Positive', 1, 0, 0)
            
            result = retrieving_news_polarity('AAPL', 5)
            
            self.assertEqual(result, (0.3, ['Title1'], 'Positive', 1, 0, 0))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_article_list(self):
        """Test handling of empty article lists"""
        analyzer = ComprehensiveSentimentAnalyzer()
        
        # Mock get_sentiment to return empty results
        with patch.object(analyzer, 'get_finviz_news', return_value=[]):
            with patch.object(analyzer, 'get_google_news', return_value=[]):
                with patch.object(analyzer.selenium_scraper, 'get_news_links', return_value=[]):
                    result = analyzer.get_sentiment('NONEXISTENT', 'Non-existent Corp')
                    
                    # Should return neutral results
                    polarity, titles, label, pos, neg, neu = result
                    self.assertEqual(polarity, 0.0)
                    self.assertEqual(label, "Neutral")
                    self.assertEqual(neu, analyzer.num_articles)  # Should pad with neutral count
    
    def test_invalid_ticker(self):
        """Test handling of invalid tickers"""
        with patch('news_sentiment.ComprehensiveSentimentAnalyzer.get_sentiment') as mock_get_sentiment:
            mock_get_sentiment.return_value = (0.0, [], 'Neutral', 0, 0, 5)
            
            result = retrieving_news_polarity('INVALIDXYZ', 5)
            
            self.assertEqual(result[0], 0.0)  # Neutral polarity
            self.assertEqual(result[2], 'Neutral')  # Neutral label


def create_test_suite():
    """Create and return a test suite with all test cases"""
    suite = unittest.TestSuite()
    
    # Add all test cases to the suite
    test_classes = [
        TestCoreFunctionality,
        TestSourceSpecificFunctions,
        TestUseCaseConfigurations,
        TestAdvancedFeatures,
        TestErrorHandlingAndMonitoring,
        TestFallbackMechanisms,
        TestPerformanceOptimizations,
        TestIntegrationScenarios,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))
    
    return suite


def run_comprehensive_tests():
    """Run all tests and provide detailed reporting"""
    print("Running Comprehensive Test Suite for news_sentiment.py")
    print("=" * 60)
    
    # Create the test suite
    suite = create_test_suite()
    
    # Create a test runner with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    
    # Run the tests
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    # Run the comprehensive test suite
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)