import requests
import json

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check Response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_single_prediction():
    """Test single customer prediction"""
    customer_data = {
        "monthly_spend": 150.0,
        "spend_change": -10.0,
        "monthly_spend_last_month": 160.0,
        "logins_last_30_days": 25.0,
        "logins_last_60_days": 45.0,
        "logins_per_day": 0.83,
        "login_trend": -5.0,
        "support_tickets": 3.0,
        "tickets_per_month": 0.15,
        "tickets_per_tenure": 0.08,
        "tenure_days": 180.0,
        "tenure_bucket_less_than_3m": 0,
        "tenure_bucket_3_to_6m": 1,
        "tenure_bucket_6_to_12m": 0,
        "tenure_bucket_1_to_2y": 0,
        "tenure_bucket_2y_plus": 0
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=customer_data,
        params={"customer_id": "CUST_001"}
    )
    
    print("Single Prediction Response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_batch_prediction():
    """Test batch prediction for multiple customers"""
    customers = [
        {
            "customer_id": "CUST_001",
            "monthly_spend": 150.0,
            "spend_change": -10.0,
            "monthly_spend_last_month": 160.0,
            "logins_last_30_days": 25.0,
            "logins_last_60_days": 45.0,
            "logins_per_day": 0.83,
            "login_trend": -5.0,
            "support_tickets": 3.0,
            "tickets_per_month": 0.15,
            "tickets_per_tenure": 0.08,
            "tenure_days": 180.0,
            "tenure_bucket_less_than_3m": 0,
            "tenure_bucket_3_to_6m": 1,
            "tenure_bucket_6_to_12m": 0,
            "tenure_bucket_1_to_2y": 0,
            "tenure_bucket_2y_plus": 0
        },
        {
            "customer_id": "CUST_002",
            "monthly_spend": 300.0,
            "spend_change": 20.0,
            "monthly_spend_last_month": 280.0,
            "logins_last_30_days": 40.0,
            "logins_last_60_days": 75.0,
            "logins_per_day": 1.33,
            "login_trend": 5.0,
            "support_tickets": 1.0,
            "tickets_per_month": 0.05,
            "tickets_per_tenure": 0.03,
            "tenure_days": 365.0,
            "tenure_bucket_less_than_3m": 0,
            "tenure_bucket_3_to_6m": 0,
            "tenure_bucket_6_to_12m": 0,
            "tenure_bucket_1_to_2y": 1,
            "tenure_bucket_2y_plus": 0
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict_batch", json=customers)
    
    print("Batch Prediction Response:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_model_info():
    """Test model information endpoint"""
    response = requests.get(f"{BASE_URL}/model_info")
    print("Model Info Response:")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("Testing Customer Churn Prediction")
    print("=" * 50)
    
    try:
        test_health_check()
        test_single_prediction()
        test_batch_prediction()
        test_model_info()
        
        print("All tests completed")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on http://localhost:8000")
        print("\nTo start the server, run:")
        print("python api.py")
        
    except Exception as e:
        print(f"Error during testing: {e}") 