# Comprehensive Testing Framework - README

## Overview

This comprehensive testing framework provides complete test coverage for the Stock Market Prediction Web App across 15 distinct phases.

## Installation

1. Install testing dependencies:
```bash
pip install -r requirements_test.txt
```

2. Install main dependencies:
```bash
pip install -r requirements.txt
```

## Running Tests

### Run All Tests
```bash
python tests/run_all_tests.py
```

### Run Specific Phase
```bash
python tests/run_all_tests.py --phase 1  # Run Phase 1 only
```

### Run with Coverage
```bash
python tests/run_all_tests.py --coverage
```

Or directly with pytest:
```bash
pytest --cov=. --cov-report=html --cov-report=term
```

### Run Unit Tests Only
```bash
python tests/run_all_tests.py --unit
```

### Run Integration Tests Only
```bash
python tests/run_all_tests.py --integration
```

### Run Specific Test File
```bash
pytest tests/test_database_models.py -v
```

## Test Phases

### Phase 1: Database Models
- Tests for all SQLAlchemy models
- Database relationships and constraints
- CRUD operations

### Phase 2: Authentication & Authorization
- User registration and login
- Password hashing and verification
- CSRF protection
- Role-based access control

### Phase 3: Portfolio Management
- Wallet operations
- Portfolio holdings
- Value calculations
- Dividend tracking

### Phase 4: Trading Operations
- Buy/sell transactions
- Commission calculations
- Broker integration

### Phase 5: LSTM Model
- Data preparation and scaling
- Model architecture
- Training and predictions
- Error metrics

### Phase 6: ARIMA Model
- Time series formatting
- Model fitting
- Forecasting
- Accuracy metrics

### Phase 7: Linear Regression Model
- Feature engineering
- Model training
- 7-day forecasting
- Performance evaluation

### Phase 8: Sentiment Analysis
- Multiple source integration
- Polarity calculations
- Fallback mechanisms
- Use case configurations

### Phase 9: Prediction Pipeline
- End-to-end prediction workflow
- Model integration
- Chart generation

### Phase 10: Web Routes
- Flask route testing
- HTTP method handling
- Template rendering

### Phase 11: API Integrations
- yfinance integration
- Alpha Vantage fallback
- Error handling

### Phase 12: E2E Workflows
- Complete user journeys
- Registration to prediction flow
- Full trading cycles

### Phase 13: Performance & Load
- Concurrent user handling
- Database query performance
- Response time benchmarks

### Phase 14: Security
- SQL injection prevention
- XSS protection
- Password security
- Session management

### Phase 15: Deployment Readiness
- Environment configuration
- Error handlers
- Production compatibility
- Health checks

## Test Markers

Tests are categorized with pytest markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.ml` - Machine learning model tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.security` - Security tests

Run specific markers:
```bash
pytest -m unit        # Run only unit tests
pytest -m "not slow"  # Skip slow tests
```

## Coverage Reports

Generate HTML coverage report:
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html  # View in browser
```

## CI/CD Integration

Tests run automatically on:
- Push to main/master/develop branches
- Pull requests
- Daily at 2 AM UTC

See `.github/workflows/tests.yml` for configuration.

## Test Structure

```
tests/
├── conftest.py                     # Pytest fixtures and configuration
├── test_helpers.py                 # Utility functions for testing
├── test_data.py                    # Sample test data
├── run_all_tests.py                # Master test runner
│
├── test_database_models.py         # Phase 1
├── test_authentication.py          # Phase 2
├── test_portfolio_management.py    # Phase 3
├── test_trading_operations.py      # Phase 4
├── test_lstm_model.py              # Phase 5
├── test_arima_model.py             # Phase 6
├── test_linear_regression_model.py # Phase 7
├── test_sentiment_sources.py       # Phase 8
├── test_prediction_pipeline.py     # Phase 9
├── test_web_routes.py              # Phase 10
└── test_e2e_api_perf_security_deployment.py  # Phases 11-15
```

## Fixtures

Common fixtures available in `conftest.py`:

- `test_app` - Flask application instance
- `client` - Test client for HTTP requests
- `test_db` - Fresh database for each test
- `sample_user` - Sample user account
- `sample_admin` - Sample admin account
- `sample_company` - Sample company/stock
- `sample_broker` - Sample broker
- `authenticated_client` - Logged-in test client
- `mock_stock_price` - Mocked stock prices
- `mock_sentiment_analysis` - Mocked sentiment data

## Best Practices

1. **Isolation**: Each test should be independent
2. **Mocking**: Mock external APIs to avoid rate limits
3. **Fast Tests**: Keep unit tests fast (<1s each)
4. **Descriptive Names**: Use clear, descriptive test names
5. **Documentation**: Document complex test scenarios

## Troubleshooting

### Import Errors
Ensure you're in the project root and have installed all dependencies:
```bash
pip install -r requirements.txt requirements_test.txt
```

### Database Errors
Tests use in-memory SQLite, no setup needed. If issues persist, check `conftest.py`.

### ML Model Tests Failing
Some ML tests are marked as `@pytest.mark.slow`. Skip them:
```bash
pytest -m "not slow"
```

### Coverage Not Generated
Install coverage packages:
```bash
pip install pytest-cov coverage
```

## Contributing

When adding new features:

1. Write tests first (TDD)
2. Ensure >80% coverage
3. Run full test suite before PR
4. Update this README if needed

## Support

For issues or questions about the testing framework, check:
- `docs/COMPREHENSIVE_TESTING_FRAMEWORK.md`
- Existing test files for examples
- CI/CD logs for detailed error messages
