from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tenure bucket categories to DataFrame.
    
    Args:
        df: DataFrame with 'tenure_days' column
        
    Returns:
        DataFrame with new 'tenure_bucket' column
    """
    bins = [0, 90, 180, 365, 730, float('inf')]
    labels = ['<3m', '3-6m', '6-12m', '1-2y', '2y+']
    df['tenure_bucket'] = pd.cut(df['tenure_days'], bins=bins, labels=labels)
    return df

def generate_customers(n=1000, seed=42):
    """
    Generate synthetic customer data.
    
    Args:
        n: Number of customers to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with customer data
    """
    random.seed(seed)
    np.random.seed(seed)
    
    data = []
    now = datetime.now()
    
    for i in range(n):
        signup = now - timedelta(days=random.randint(1, 1000))
        churned = random.random() < 0.3
        last_active = signup if churned else now - timedelta(days=random.randint(0, 30))
        monthly_spend = round(random.uniform(50, 500), 2)
        monthly_spend_last_month = round(monthly_spend * random.uniform(0.8, 1.2), 2)
        
        data.append({
            "customer_id": i + 1,
            "signup_date": signup.date(),
            "last_active_date": last_active.date(),
            "monthly_spend": monthly_spend,
            "monthly_spend_last_month": monthly_spend_last_month,
            "support_tickets": random.randint(0, 5),
            "logins_last_30_days": random.randint(0, 30),
            "churned": churned
        })
    
    return pd.DataFrame(data)

def process_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process customer data and engineer features.
    
    Args:
        df: Raw customer DataFrame
        
    Returns:
        Processed DataFrame with engineered features
    """
    # Convert dates to datetime if they aren't already
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    now = pd.to_datetime(datetime.now().date())
    
    # Calculate features
    df['tenure_days'] = (now - df['signup_date']).dt.days
    df = add_tenure_bucket(df)
    df['spend_change'] = df['monthly_spend'] - df['monthly_spend_last_month']
    df['tickets_per_tenure'] = df['support_tickets'] / (df['tenure_days'] + 1)
    df['logins_per_day'] = df['logins_last_30_days'] / 30
    df['tickets_per_month'] = df['support_tickets'] / ((df['tenure_days'] / 30).clip(lower=1))
    
    # Simulate 60-day logins for trend calculation
    df['logins_last_60_days'] = df['logins_last_30_days'] + np.random.randint(0, 30, size=len(df))
    df['login_trend'] = df['logins_last_30_days'] - (df['logins_last_60_days'] - df['logins_last_30_days'])
    
    # One-hot encode tenure buckets
    tenure_dummies = pd.get_dummies(df['tenure_bucket'], prefix='tenure_bucket')
    df = pd.concat([df, tenure_dummies], axis=1)
    
    return df

if __name__ == "__main__":
    # Only run when script is executed directly
    load_dotenv()
    
    # Database connection
    engine = create_engine(
        f"postgresql+psycopg://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    
    # Generate and process data
    df = generate_customers()
    df = process_customer_data(df)
    
    # Save to database
    df.to_sql("customers", engine, index=False, if_exists="replace")
    print(f"Inserted {len(df)} customers into 'customers' table.")