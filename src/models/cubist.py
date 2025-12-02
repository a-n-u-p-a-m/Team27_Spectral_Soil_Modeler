"""
Cubist Model Module
===================
Cubist algorithm implementation wrapper.

Cubist is a rule-based regression model that creates transparent, 
interpretable rules for predictions while maintaining competitive accuracy.
"""

import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CubistModel:
    """
    Cubist Rule-Based Regression for spectral soil property prediction.
    
    Cubist benefits:
    - Interpretable rule-based predictions
    - Good performance on spectral data
    - Transparent decision logic
    - Effective for handling interactions
    
    Note: This is a wrapper. In production, you'd use:
    - rpy2 with R's Cubist package, or
    - sklearn's tree-based alternatives
    
    For this implementation, we use sklearn's GradientBoostingRegressor
    as a proxy that provides similar tree-based rule learning.
    
    Attributes:
        model: Tree-based regression model
        is_trained: Whether model has been trained
    """
    
    def __init__(self, n_rules: int = 20, neighbors: int = 5,
                 tune_hyperparameters: bool = False, cv_folds: int = 5):
        """
        Initialize Cubist model.
        
        Parameters
        ----------
        n_rules : int, default=20
            Number of rules to generate
        neighbors : int, default=5
            Number of neighbors for instance-based smoothing
        tune_hyperparameters : bool, default=False
            Whether to tune hyperparameters using cross-validation
        cv_folds : int, default=5
            Number of cross-validation folds for tuning
        """
        # Use sklearn's DecisionTreeRegressor as a proxy
        # In production, integrate actual Cubist via rpy2
        from sklearn.tree import DecisionTreeRegressor
        
        self.n_rules = n_rules
        self.neighbors = neighbors
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        
        self.best_params = {
            'n_rules': n_rules,
            'neighbors': neighbors
        }
        
        # Tree depth approximately relates to number of rules
        max_depth = max(3, int(np.log2(n_rules)))
        
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.is_trained = False
        self.tuning_results = None
        logger.info(f"Cubist model initialized: n_rules={n_rules}, neighbors={neighbors}, "
                   f"tune_hyperparameters={tune_hyperparameters}")
    
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'CubistModel':
        """
        Train Cubist model, with optional hyperparameter tuning.
        
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
            if self.tune_hyperparameters:
                logger.info("Performing hyperparameter tuning for Cubist")
                try:
                    from .hyperparameter_tuner import HyperparameterTuner
                    
                    tuner = HyperparameterTuner(
                        model_name='Cubist',
                        use_small_grid=True,
                        cv_folds=self.cv_folds
                    )
                    
                    # Create base model for tuning with sklearn-compatible parameters
                    from sklearn.tree import DecisionTreeRegressor
                    base_model = DecisionTreeRegressor(
                        max_depth=self.n_rules,
                        min_samples_split=5,
                        min_samples_leaf=self.neighbors,
                        random_state=42
                    )
                    
                    # Remap Cubist param grid to DecisionTreeRegressor params
                    original_grid = tuner.param_grid.copy()
                    tuner.param_grid = {
                        'max_depth': [5, 10, 15, 20],
                        'min_samples_leaf': [1, 3, 5]
                    }
                    
                    tuning_results = tuner.tune_model(base_model, X_train, y_train, verbose=False)
                    best_params = tuning_results['best_params']
                    best_score = tuning_results['best_score']
                    self.best_params = best_params
                    self.tuning_results = tuning_results
                    
                    logger.info(f"Best hyperparameters found: {best_params}")
                    if best_score is not None:
                        logger.info(f"Best cross-validation score: {best_score:.4f}")
                    
                    # Train final model with best parameters
                    from sklearn.tree import DecisionTreeRegressor
                    self.model = DecisionTreeRegressor(
                        max_depth=best_params.get('max_depth', self.n_rules),
                        min_samples_split=5,
                        min_samples_leaf=best_params.get('min_samples_leaf', self.neighbors),
                        random_state=42
                    )
                except Exception as e:
                    logger.warning(f"Cubist hyperparameter tuning failed: {str(e)}. Using default parameters.")
                    # Fall back to default parameters
                    from sklearn.tree import DecisionTreeRegressor
                    max_depth = max(3, int(np.log2(self.n_rules)))
                    self.model = DecisionTreeRegressor(
                        max_depth=max_depth,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42
                    )
                    self.best_params = {}
                    self.tuning_results = {'best_score': None, 'error': str(e)}
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            logger.info(f"Cubist model trained. Training R²: {self.model.score(X_train, y_train):.4f}")
            return self
        except Exception as e:
            logger.error(f"Error training Cubist model: {str(e)}")
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
        Get feature importance from Cubist model.
        
        Parameters
        ----------
        X : np.ndarray, optional
            Feature data (not required)
            
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
        """Get model information, including tuning results if available."""
        info = {
            'name': 'Cubist',
            'n_rules': self.n_rules,
            'neighbors': self.neighbors,
            'is_trained': self.is_trained,
            'model_type': 'Rule-Based Regression (Cubist)'
        }
        
        if self.tuning_results:
            info['tuning_enabled'] = True
            info['best_hyperparameters'] = self.best_params
            info['tuning_cv_score'] = self.tuning_results.get('best_score', None)
        else:
            info['tuning_enabled'] = False
        
        return info
