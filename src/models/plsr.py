"""
PLSR Model Module
=================
Partial Least Squares Regression (PLSR) implementation with hyperparameter optimization.

PLSR is effective for high-dimensional spectral data with multicollinearity.
It performs dimensionality reduction while considering the target variable.
"""

from sklearn.cross_decomposition import PLSRegression
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PLSRModel:
    """
    Partial Least Squares Regression for spectral soil property prediction.
    
    PLSR is particularly suited for spectral data because:
    - Handles multicollinearity between wavelength bands
    - Reduces dimensionality while preserving predictive information
    - Relates multiple X and y variables simultaneously
    
    Attributes:
        model: Fitted PLSRegression object
        n_components: Number of latent components
        is_trained: Whether model has been trained
    """
    
    def __init__(self, n_components: int = 10, tune_hyperparameters: bool = False,
                 cv_folds: int = 5):
        """
        Initialize PLSR model.
        
        Parameters
        ----------
        n_components : int, default=10
            Number of latent components to extract
        tune_hyperparameters : bool, default=False
            Whether to tune hyperparameters using cross-validation
        cv_folds : int, default=5
            Number of cross-validation folds for tuning
        """
        self.n_components = n_components
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.best_n_components = n_components
        self.model = PLSRegression(n_components=n_components, scale=True)
        self.is_trained = False
        self.tuning_results = None
        logger.info(f"PLSR model initialized with n_components={n_components}, "
                   f"tune_hyperparameters={tune_hyperparameters}")
    
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'PLSRModel':
        """
        Train PLSR model with optional hyperparameter tuning.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features (n_samples, n_features)
        y_train : np.ndarray
            Training target (n_samples,)
            
        Returns
        -------
        self
            Trained model
        """
        try:
            # Perform hyperparameter tuning if enabled
            if self.tune_hyperparameters:
                from .hyperparameter_tuner import HyperparameterTuner
                
                tuner = HyperparameterTuner(
                    'PLSR',
                    cv_folds=self.cv_folds,
                    use_small_grid=True
                )
                
                # Create base model for tuning
                base_model = PLSRegression(scale=True)
                tuning_result = tuner.tune_model(base_model, X_train, y_train, verbose=False)
                
                self.tuning_results = tuning_result
                
                if tuning_result['tuned']:
                    self.best_n_components = tuning_result['best_params'].get('n_components', 10)
                    logger.info(f"Optimal n_components found: {self.best_n_components}")
                    self.model = PLSRegression(n_components=self.best_n_components, scale=True)
                else:
                    logger.warning("Hyperparameter tuning failed, using default parameters")
            
            # Train with selected/tuned parameters
            self.model.fit(X_train, y_train)
            self.is_trained = True
            train_score = self.model.score(X_train, y_train)
            logger.info(f"PLSR model trained. n_components={self.model.n_components}, "
                       f"Training R²={train_score:.4f}")
            return self
        except Exception as e:
            logger.error(f"Error training PLSR model: {str(e)}")
            raise
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Features to predict
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        return predictions.ravel() if predictions.ndim > 1 else predictions
    
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            True values
            
        Returns
        -------
        float
            R² score
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        return self.model.score(X, y)
    
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Get feature importance based on PLS weights.
        
        Parameters
        ----------
        X : np.ndarray
            Feature data
            
        Returns
        -------
        np.ndarray
            Feature importance scores
        """
        # Use absolute values of model weights
        importance = np.abs(self.model.coef_)
        return importance / importance.sum()  # Normalize
    
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'name': 'PLSR',
            'n_components': self.model.n_components,
            'best_n_components': self.best_n_components,
            'tune_hyperparameters': self.tune_hyperparameters,
            'is_trained': self.is_trained,
            'tuning_results': self.tuning_results,
            'model_type': 'Partial Least Squares Regression'
        }
