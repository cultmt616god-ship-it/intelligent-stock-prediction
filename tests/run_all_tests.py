"""
Master Test Suite Runner

Runs all 15 phases of tests for the Stock Market Prediction Web App.
"""

import sys
import pytest


def run_all_tests():
    """Run all test phases."""
    print("=" * 80)
    print("STOCK MARKET PREDICTION WEB APP - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Test phases
    test_phases = [
        ("Phase 1: Database Models", "tests/test_database_models.py"),
        ("Phase 2: Authentication & Authorization", "tests/test_authentication.py"),
        ("Phase 3: Portfolio Management", "tests/test_portfolio_management.py"),
        ("Phase 4: Trading Operations", "tests/test_trading_operations.py"),
        ("Phase 5: LSTM Model", "tests/test_lstm_model.py"),
        ("Phase 6: ARIMA Model", "tests/test_arima_model.py"),
        ("Phase 7: Linear Regression Model", "tests/test_linear_regression_model.py"),
        ("Phase 8: Sentiment Analysis Sources", "tests/test_sentiment_sources.py"),
        ("Phase 9: Prediction Pipeline Integration", "tests/test_prediction_pipeline.py"),
        ("Phase 10: Web Routes Integration", "tests/test_web_routes.py"),
        ("Phase 11-15: API, E2E, Performance, Security, Deployment", "tests/test_e2e_api_perf_security_deployment.py"),
    ]
    
    # Run all tests
    args = [
        '--verbose',
        '--tb=short',
        '--color=yes',
        '-v'
    ]
    
    # Add all test files
    for phase_name, test_file in test_phases:
        args.append(test_file)
    
    print(f"Running {len(test_phases)} test phases...")
    print()
    
    # Run pytest
    exit_code = pytest.main(args)
    
    print()
    print("=" * 80)
    if exit_code == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)
    
    return exit_code


def run_phase(phase_number):
    """Run a specific test phase."""
    phases = {
        1: "tests/test_database_models.py",
        2: "tests/test_authentication.py",
        3: "tests/test_portfolio_management.py",
        4: "tests/test_trading_operations.py",
        5: "tests/test_lstm_model.py",
        6: "tests/test_arima_model.py",
        7: "tests/test_linear_regression_model.py",
        8: "tests/test_sentiment_sources.py",
        9: "tests/test_prediction_pipeline.py",
        10: "tests/test_web_routes.py",
        11: "tests/test_e2e_api_perf_security_deployment.py",
    }
    
    if phase_number not in phases:
        print(f"Error: Phase {phase_number} not found.")
        print(f"Available phases: 1-11")
        return 1
    
    test_file = phases[phase_number]
    print(f"Running Phase {phase_number}: {test_file}")
    print()
    
    exit_code = pytest.main(['-v', '--tb=short', test_file])
    return exit_code


def run_with_coverage():
    """Run all tests with coverage report."""
    print("Running tests with coverage reporting...")
    print()
    
    exit_code = pytest.main([
        '--cov=.',
        '--cov-report=html',
        '--cov-report=term',
        '--verbose',
        'tests/'
    ])
    
    print()
    print("Coverage report generated in htmlcov/index.html")
    return exit_code


def run_unit_tests_only():
    """Run only unit tests (skip integration and slow tests)."""
    print("Running unit tests only...")
    print()
    
    exit_code = pytest.main([
        '-v',
        '-m', 'unit',
        '--tb=short',
        'tests/'
    ])
    return exit_code


def run_integration_tests_only():
    """Run only integration tests."""
    print("Running integration tests only...")
    print()
    
    exit_code = pytest.main([
        '-v',
        '-m', 'integration',
        '--tb=short',
        'tests/'
    ])
    return exit_code


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Stock Market Prediction App Tests')
    parser.add_argument('--phase', type=int, help='Run specific phase (1-11)')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    
    args = parser.parse_args()
    
    if args.phase:
        exit_code = run_phase(args.phase)
    elif args.coverage:
        exit_code = run_with_coverage()
    elif args.unit:
        exit_code = run_unit_tests_only()
    elif args.integration:
        exit_code = run_integration_tests_only()
    else:
        exit_code = run_all_tests()
    
    sys.exit(exit_code)
