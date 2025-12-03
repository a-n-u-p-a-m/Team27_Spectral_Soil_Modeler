"""
GBRT Model Module
=================
Gradient Boosting Regression Trees implementation with hyperparameter optimization.

GBRT is excellent for capturing non-linear relationships in spectral data
and provides good feature importance estimates.
"""

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GBRTModel:
    """
    Gradient Boosting Regression Trees for spectral soil property prediction.
    
    GBRT benefits:
    - Captures non-linear relationships
    - Provides feature importance out-of-the-box
    - Robust to outliers
    - Good generalization with proper hyperparameters
    
    Attributes:
        model: Fitted GradientBoostingRegressor object
        is_trained: Whether model has been trained
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 5, random_state: int = 42,
                 tune_hyperparameters: bool = False, cv_folds: int = 5, cv_strategy: str = 'k-fold',
                 search_method: str = 'grid', n_iter: int = 20):
        """
        Initialize GBRT model.
        
        Parameters
        ----------
        n_estimators : int, default=100
            Number of boosting stages
        learning_rate : float, default=0.1
            Shrinkage factor for learning rate
        max_depth : int, default=5
            Maximum tree depth
        random_state : int, default=42
            Random seed for reproducibility
        tune_hyperparameters : bool, default=False
            Whether to tune hyperparameters using cross-validation
        cv_folds : int, default=5
            Number of cross-validation folds for tuning (ignored for LOO)
        cv_strategy : str, default='k-fold'
            Cross-validation strategy: 'k-fold' or 'leave-one-out'
        search_method : str, default='grid'
            Hyperparameter search method: 'grid' for GridSearchCV or 'random' for RandomizedSearchCV
        n_iter : int, default=20
            Number of iterations for RandomizedSearchCV
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.search_method = search_method.lower()
        self.n_iter = n_iter
        
        self.best_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth
        }
        
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbose=0
        )
        self.is_trained = False
        self.tuning_results = None
        logger.info(f"GBRT model initialized: n_estimators={n_estimators}, "
                   f"learning_rate={learning_rate}, max_depth={max_depth}, "
                   f"tune_hyperparameters={tune_hyperparameters}")
    
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'GBRTModel':
        """
        Train GBRT model with optional hyperparameter tuning.
        
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
            # Log tuning status at the start
            logger.info(f"GBRT train() called with tune_hyperparameters={self.tune_hyperparameters}, cv_strategy={self.cv_strategy}")
            
            # Perform hyperparameter tuning if enabled
            if self.tune_hyperparameters:
                logger.info(f"🔧 TUNING INITIATED for GBRT with cv_strategy={self.cv_strategy}")
                from .hyperparameter_tuner import HyperparameterTuner
                
                tuner = HyperparameterTuner(
                    'GBRT',
                    cv_folds=self.cv_folds,
                    search_type=self.search_method,
                    use_small_grid=(self.search_method == 'random'),
                    n_iter=self.n_iter,
                    cv_strategy=self.cv_strategy
                )
                
                # Create base model for tuning
                base_model = GradientBoostingRegressor(random_state=self.random_state, verbose=0)
                tuning_result = tuner.tune_model(base_model, X_train, y_train, verbose=False)
                
                self.tuning_results = tuning_result
                
                if tuning_result['tuned']:
                    self.best_params = tuning_result['best_params']
                    logger.info(f"✅ Optimal hyperparameters found: {self.best_params}")
                    self.model = GradientBoostingRegressor(
                        random_state=self.random_state,
                        verbose=0,
                        **self.best_params
                    )
                else:
                    logger.warning("⚠️ Hyperparameter tuning failed for GBRT, using default parameters")
            else:
                logger.info("ℹ️ GBRT training without hyperparameter tuning (tune_hyperparameters=False)")
            
            # Train with selected/tuned parameters
            self.model.fit(X_train, y_train)
            self.is_trained = True
            train_score = self.model.score(X_train, y_train)
            logger.info(f"GBRT model trained. Training R²: {train_score:.4f}")
            return self
        except Exception as e:
            logger.error(f"Error training GBRT model: {str(e)}")
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
        
        return self.model.predict(X)
    
    
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
    
    
    def get_feature_importance(self, X: np.ndarray = None) -> np.ndarray:
        """
        Get feature importance from GBRT.
        
        Parameters
        ----------
        X : np.ndarray, optional
            Feature data (not used for GBRT)
            
        Returns
        -------
        np.ndarray
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        importance = self.model.feature_importances_
        return importance / importance.sum()  # Normalize
    
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'name': 'GBRT',
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'best_params': self.best_params,
            'tune_hyperparameters': self.tune_hyperparameters,
            'is_trained': self.is_trained,
            'tuning_results': self.tuning_results,
            'model_type': 'Gradient Boosting Regression Trees'
        }
