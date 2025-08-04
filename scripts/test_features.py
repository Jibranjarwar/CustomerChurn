import os
import sys
import pandas as pd
import pytest
import numpy as np

from datetime import datetime, date

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_gen import add_tenure_bucket, process_customer_data, generate_customers
from modeling.utils import validate_data, validate_config

def test_tenure_bucket():
    """Test tenure bucket creation"""
    df = pd.DataFrame({"tenure_days": [45, 120, 400, 800]})
    df = add_tenure_bucket(df)
    assert set(df["tenure_bucket"].unique()) <= {"<3m", "3-6m", "6-12m", "1-2y", "2y+"}

def test_tickets_per_tenure():
    """Test tickets per tenure calculation"""
    df = pd.DataFrame({"support_tickets": [10, 0], "tenure_days": [100, 0]})
    df["tickets_per_tenure"] = df["support_tickets"] / (df["tenure_days"] + 1)
    assert df["tickets_per_tenure"].iloc[1] == 0

def test_spend_change():
    """Test spend change calculation"""
    df = pd.DataFrame({"monthly_spend": [100, 80], "monthly_spend_last_month": [90, 100]})
    df["spend_change"] = df["monthly_spend"] - df["monthly_spend_last_month"]
    assert (df["spend_change"] == [10, -20]).all()

def test_validate_config():
    """Test config validation"""
    invalid_config = {"data": {"path": "some/path"}}  # Missing model section
    with pytest.raises(ValueError):
        validate_config(invalid_config)

def test_validate_data():
    """Test data validation"""
    df = pd.DataFrame({
        "monthly_spend": [100, None],
        "churned": [0, 1]
    })
    with pytest.raises(ValueError):
        validate_data(df, ["monthly_spend"], "churned")

def test_generate_customers():
    """Test customer data generation"""
    df = generate_customers(n=10, seed=42)
    assert len(df) == 10
    assert set(df.columns) >= {"customer_id", "signup_date", "monthly_spend"}

def test_process_customer_data():
    """Test customer data processing"""
    # Create minimal test data
    test_data = pd.DataFrame({
        "signup_date": pd.to_datetime([date(2024, 1, 1)]),  # Convert to datetime
        "monthly_spend": [100],
        "monthly_spend_last_month": [90],
        "support_tickets": [5],
        "logins_last_30_days": [15]
    })
    
    # Set random seed for reproducible random values
    np.random.seed(42)
    
    processed_df = process_customer_data(test_data)
    
    # Test required columns exist
    assert "tenure_days" in processed_df.columns
    assert "tickets_per_tenure" in processed_df.columns
    assert "login_trend" in processed_df.columns
    
    # Test calculations are correct
    assert processed_df["spend_change"].iloc[0] == 10  # 100 - 90
    assert processed_df["logins_per_day"].iloc[0] == 0.5  # 15 / 30
    assert processed_df["tickets_per_tenure"].iloc[0] > 0  # Should be positive