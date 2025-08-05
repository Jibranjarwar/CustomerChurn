# Customer Churn Prediction System

A machine learning system that predicts customer churn using customer behavior data.

## What This System Does

This system takes customer data (spending patterns, login activity, support tickets, tenure) and predicts whether a customer will churn. It processes raw customer data, builds predictive models, and provides an API.

## How It Works

### Data Processing
- Reads customer data from CSV files
- Creates new features from existing data (interaction features, customer segments)
- Validates data quality and handles missing values
- Splits data into training and testing sets

### Model Training
- Uses Random Forest algorithm for prediction
- Tests different parameter combinations to find the best model
- Uses cross-validation to ensure reliable performance
- Tracks experiments with MLflow to compare different models

### Feature Engineering
- Combines spending and login patterns to create interaction features
- Identifies high-value customers based on spending thresholds
- Categorizes customers as active users or support-heavy users
- Creates customer segments based on risk factors

### Model Evaluation
- Calculates accuracy, precision, recall, and ROC AUC scores
- Generates confusion matrix visualizations
- Creates feature importance charts
- Saves performance metrics and visualizations

## What You Get

### Trained Model
- A saved model file that can predict churn probability for new customers
- Model performance metrics and evaluation results
- Feature importance rankings showing which factors most influence churn

### Visualizations
- Confusion matrix showing prediction accuracy
- Feature importance chart showing which customer behaviors matter most
- Data analysis plots for understanding customer patterns

### API Service
- REST API for making predictions on individual customers
- Batch prediction endpoint for multiple customers
- Health check and model information endpoints
- Risk level classification (high/medium/low risk)

### Experiment Tracking
- MLflow logs of all training runs
- Performance metrics for each model version
- Parameter settings and results comparison

## Project Structure

```
churn_prediction_system/
├── data/
│   ├── raw/           # Original customer data
│   └── processed/     # Cleaned and prepared data
├── modeling/
│   ├── features.py    # Creates new features from raw data
│   ├── train_model.py # Trains models with GridSearch
│   ├── tune_model.py  # Optimizes models with Optuna
│   ├── utils.py       # Evaluates models and creates visualizations
│   └── config.yaml    # Settings for training and validation
├── models/            # Saved trained models
├── artifacts/         # Generated charts and visualizations
├── api.py            # FastAPI service for predictions
├── test_api.py       # Tests for the API endpoints
└── notebooks/        # Data exploration and analysis
```

## Setup and Usage

### Local Development
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train a model:
```bash
python -m modeling.train_model
```

3. Start the API:
```bash
python api.py
```

### Docker Deployment
1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Or build manually:
```bash
docker build -t churn-prediction .
docker run -p 8000:8000 churn-prediction
```

## API Usage

### Single Prediction
```python
import requests

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
    "http://localhost:8000/predict",
    json=customer_data,
    params={"customer_id": "CUST_001"}
)
print(response.json())
```

### API Endpoints
- `GET /` - Health check
- `GET /health` - Detailed system status
- `POST /predict` - Predict churn for one customer
- `POST /predict_batch` - Predict churn for multiple customers
- `GET /model_info` - Get model details and feature importances

## Testing

Test the API:
```bash
python test_api.py
```

Test the feature engineering:
```bash
python scripts/test_features.py
```

## Technologies Used

- **Python 3.9+**
- **Scikit-learn** - Machine learning algorithms
- **Pandas/Numpy** - Data processing
- **MLflow** - Experiment tracking
- **FastAPI** - API framework
- **Docker** - Containerization
- **Matplotlib** - Visualizations
- **pgAdmin4 PostgreSQL** - Database management for customer data generation

## Output Files

After training, you get:
- `models/model.pkl` - Trained model file
- `artifacts/confusion_matrix.png` - Prediction accuracy visualization
- `artifacts/feature_importances.png` - Feature importance chart
- `model_training.log` - Training process logs
- MLflow experiment tracking with all metrics and parameters
