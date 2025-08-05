"""
Test suite for churn prediction system
"""

import os
import sys
import pandas as pd
import pytest
import numpy as np
from datetime import datetime
import logging
import time
from typing import Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_execution.log')
    ]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.utils import validate_data, validate_config
from modeling.exceptions import ConfigurationError, DataValidationError

def test_validate_config() -> None:
    """Test config validation"""
    logger.info("Testing config validation")
    start_time = time.time()
    
    try:
        invalid_config = {"data": {"path": "some/path"}}  
        with pytest.raises(ConfigurationError):
            validate_config(invalid_config)
        logger.info("Config validation test passed")
        
        test_time = time.time() - start_time
        logger.info(f"Config validation test completed: {test_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Config validation test failed: {str(e)}")
        raise

def test_validate_data() -> None:
    """Test data validation"""
    logger.info("Testing data validation")
    start_time = time.time()
    
    try:
        df = pd.DataFrame({
            "monthly_spend": [100, None],
            "churned": [0, 1]
        })
        with pytest.raises(DataValidationError):
            validate_data(df, ["monthly_spend"], "churned")
        logger.info("Data validation test passed")
        
        test_time = time.time() - start_time
        logger.info(f"Data validation test completed: {test_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Data validation test failed: {str(e)}")
        raise

def test_features_and_target_extraction() -> None:
    """Test feature extraction function"""
    logger.info("Testing feature extraction")
    start_time = time.time()
    
    try:
        df = pd.DataFrame({
            "monthly_spend": [100.0, 200.0, 150.0],
            "spend_change": [10.0, -20.0, 5.0],
            "monthly_spend_last_month": [90.0, 220.0, 145.0],
            "logins_last_30_days": [15.0, 25.0, 20.0],
            "logins_last_60_days": [30.0, 50.0, 40.0],
            "logins_per_day": [0.5, 0.83, 0.67],
            "login_trend": [5.0, -5.0, 0.0],
            "support_tickets": [2.0, 5.0, 3.0],
            "tickets_per_month": [0.1, 0.2, 0.15],
            "tickets_per_tenure": [0.05, 0.1, 0.075],
            "tenure_days": [100.0, 200.0, 150.0],
            "tenure_bucket_<3m": [1, 0, 0],
            "tenure_bucket_3-6m": [0, 1, 0],
            "tenure_bucket_6-12m": [0, 0, 1],
            "tenure_bucket_1-2y": [0, 0, 0],
            "tenure_bucket_2y+": [0, 0, 0],
            "churned": [0, 1, 0]
        })
        
        from modeling.features import get_features_and_target
        X, y = get_features_and_target(df)
        
        expected_raw_features = [
            "monthly_spend", "spend_change", "monthly_spend_last_month",
            "logins_last_30_days", "logins_last_60_days", "logins_per_day", "login_trend",
            "support_tickets", "tickets_per_month", "tickets_per_tenure",
            "tenure_days", "tenure_bucket_<3m", "tenure_bucket_3-6m", 
            "tenure_bucket_6-12m", "tenure_bucket_1-2y", "tenure_bucket_2y+"
        ]
        
        expected_engineered_features = [
            "spend_login_interaction", "spend_support_interaction", "tenure_engagement_interaction",
            "login_frequency_trend", "is_high_value", "is_active_user", "is_support_heavy", "customer_segment"
        ]
        
        expected_features = expected_raw_features + expected_engineered_features
        
        logger.info(f"Testing {len(expected_features)} expected features")
        for feature in expected_features:
            assert feature in X.columns, f"Missing feature: {feature}"
            logger.debug(f"Feature '{feature}' found")
        
        assert len(X.columns) == 24
        assert len(y) == 3
        assert y.dtype == np.int32
        
        assert X["monthly_spend"].dtype == np.float64
        assert X["tenure_bucket_<3m"].dtype == np.int32
        assert X["login_trend"].dtype == np.float64
        
        assert X["spend_login_interaction"].dtype == np.float64
        assert X["is_high_value"].dtype == np.int32
        assert X["customer_segment"].dtype == np.int32
        
        logger.info("Feature extraction test passed")
        
        test_time = time.time() - start_time
        logger.info(f"Feature extraction test completed: {test_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Feature extraction test failed: {str(e)}")
        raise

def test_data_quality_validation() -> None:
    """Test data quality validation"""
    logger.info("Testing data quality validation")
    start_time = time.time()
    
    try:
        df_good = pd.DataFrame({
            "monthly_spend": [100.0, 200.0],
            "churned": [0, 1]
        })
        
        df_bad = pd.DataFrame({
            "monthly_spend": [100.0, None],
            "churned": [0, 1]
        })
        
        logger.info("Testing good data validation")
        try:
            validate_data(df_good, ["monthly_spend"], "churned")
            validation_passed = True
            logger.debug("Good data validation passed")
        except DataValidationError:
            validation_passed = False
            logger.error("Good data validation failed")
        
        assert validation_passed, "Good data should pass validation"
        
        logger.info("Testing bad data validation")
        with pytest.raises(DataValidationError):
            validate_data(df_bad, ["monthly_spend"], "churned")
        logger.debug("Bad data validation correctly, missing values")
        
        logger.info("Data quality validation test passed")
        
        test_time = time.time() - start_time
        logger.info(f"Data quality validation test completed: {test_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Data quality validation test failed: {str(e)}")
        raise

def test_config_validation() -> None:
    """Test configuration validation"""
    logger.info("Testing configuration validation")
    start_time = time.time()
    
    try:
        valid_config = {
            "data": {"path": "data/processed/customers_clean.csv"},
            "model": {"test_size": 0.2, "random_state": 42}
        }
        
        logger.info("Testing valid configuration")
        validate_config(valid_config)
        logger.debug("Valid configuration validation passed")
        
        logger.info("Testing invalid configuration")
        invalid_config = {"data": {"path": "some/path"}}
        with pytest.raises(ConfigurationError):
            validate_config(invalid_config)
        logger.debug("Invalid configuration validation,caught missing sections")
        
        logger.info("Testing invalid test size")
        invalid_test_size = {
            "data": {"path": "data/processed/customers_clean.csv"},
            "model": {"test_size": 1.5, "random_state": 42}
        }
        with pytest.raises(ConfigurationError):
            validate_config(invalid_test_size)
        logger.debug("Invalid test size validation, caught invalid value")
        
        logger.info("Configuration validation test passed")
        
        test_time = time.time() - start_time
        logger.info(f"Configuration validation test completed: {test_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Configuration validation test failed: {str(e)}")
        raise

def test_feature_imports() -> None:
    """Test that all required modules can be imported"""
    logger.info("Testing module imports")
    start_time = time.time()
    
    try:
        from modeling.features import get_features_and_target, RAW_FEATURE_COLUMNS, TARGET_COLUMN
        from modeling.utils import validate_data, validate_config
        imports_successful = True
        logger.debug("All required modules imported")
    except ImportError as e:
        imports_successful = False
        logger.error(f"Import error: {e}")
    
    assert imports_successful, "All required modules should import successfully"
    logger.info("Module import test passed")
    
    test_time = time.time() - start_time
    logger.info(f"Module import test completed: {test_time:.3f}s")

def test_feature_constants() -> None:
    """Test that feature constants are correctly defined"""
    logger.info("Testing feature constants")
    start_time = time.time()
    
    try:
        from modeling.features import RAW_FEATURE_COLUMNS, TARGET_COLUMN, FEATURE_DTYPES
        
        assert len(RAW_FEATURE_COLUMNS) == 16, f"Should have 16 raw features, got {len(RAW_FEATURE_COLUMNS)}"
        assert TARGET_COLUMN == "churned", "Target column should be 'churned'"
        
        logger.info(f"Testing {len(RAW_FEATURE_COLUMNS)} raw feature constants")
        for feature in RAW_FEATURE_COLUMNS:
            assert feature in FEATURE_DTYPES, f"Raw feature {feature} should have defined dtype"
            logger.debug(f"Raw feature '{feature}' has defined dtype")
        
        assert TARGET_COLUMN not in RAW_FEATURE_COLUMNS, "Target should not be in raw features"
        
        expected_raw_features = [
            "monthly_spend", "spend_change", "monthly_spend_last_month",
            "logins_last_30_days", "logins_last_60_days", "logins_per_day", "login_trend",
            "support_tickets", "tickets_per_month", "tickets_per_tenure",
            "tenure_days", "tenure_bucket_<3m", "tenure_bucket_3-6m", 
            "tenure_bucket_6-12m", "tenure_bucket_1-2y", "tenure_bucket_2y+"
        ]
        
        logger.info("Testing raw features")
        for feature in expected_raw_features:
            assert feature in RAW_FEATURE_COLUMNS, f"Raw feature {feature} should be in RAW_FEATURE_COLUMNS"
            assert feature in FEATURE_DTYPES, f"Raw feature {feature} should have defined dtype"
            logger.debug(f"Raw feature '{feature}' properly defined")
        
        logger.info("Feature constants test passed")
        
        test_time = time.time() - start_time
        logger.info(f"Feature constants test completed: {test_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Feature constants test failed: {str(e)}")
        raise

if __name__ == "__main__":
    logger.info("Starting tests")
    total_start_time = time.time()
    
    try:
        # Run all tests
        test_functions = [
            test_validate_config,
            test_validate_data,
            test_features_and_target_extraction,
            test_data_quality_validation,
            test_config_validation,
            test_feature_imports,
            test_feature_constants
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_func in test_functions:
            try:
                test_func()
                passed_tests += 1
                logger.info(f"✓ {test_func.__name__} passed")
            except Exception as e:
                failed_tests += 1
                logger.error(f"✗ {test_func.__name__} failed: {str(e)}")
        
        total_time = time.time() - total_start_time
        
        logger.info("=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total tests: {len(test_functions)}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success rate: {passed_tests/len(test_functions)*100:.1f}%")
        logger.info(f"Total execution time: {total_time:.2f}s")
        
        if failed_tests == 0:
            logger.info("All tests passed")
        else:
            logger.warning(f"{failed_tests} test(s) failed")
            
    except Exception as e:
        logger.error(f"Test suite execution failed: {str(e)}")
        raise