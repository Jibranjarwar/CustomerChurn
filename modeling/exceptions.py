"""
Custom exceptions for churn prediction system.
"""

class ChurnPredictionError(Exception):
    """Base exception for all churn prediction related errors"""
    pass


class ConfigurationError(ChurnPredictionError):
    """Raised when configuration is invalid or missing"""
    pass


class DataValidationError(ChurnPredictionError):
    """Raised when data validation fails"""
    pass


class FeatureEngineeringError(ChurnPredictionError):
    """Raised when feature engineering fails"""
    pass


class ModelTrainingError(ChurnPredictionError):
    """Raised when model training fails"""
    pass


class ModelEvaluationError(ChurnPredictionError):
    """Raised when model evaluation fails"""
    pass


class DataLeakageError(ChurnPredictionError):
    """Raised when data leakage is detected"""
    pass


class InsufficientDataError(ChurnPredictionError):
    """Raised when there is insufficient data for training/evaluation"""
    pass 