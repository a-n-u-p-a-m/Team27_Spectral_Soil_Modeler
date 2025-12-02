"""
SVR Model Module
================
Support Vector Regression implementation with hyperparameter optimization.

SVR is effective for regression tasks with high-dimensional data and provides
excellent generalization through structural risk minimization.
"""

from sklearn.svm import SVR
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SVRModel:
    """
    Support Vector Regression for spectral soil property prediction.
    
    SVR benefits:
    - Effective in high-dimensional spaces
    - Memory efficient using only support vectors
    - Flexible kernel methods for non-linear relationships
    - Good generalization with proper hyperparameters
    
    Attributes:
        model: Fitted SVR object
        kernel: Kernel type
        is_trained: Whether model has been trained
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 100.0,
                 epsilon: float = 0.1, gamma: str = 'scale',
                 tune_hyperparameters: bool = False, cv_folds: int = 5):
        """
        Initialize SVR model.
        
        Parameters
        ----------
        kernel : str, default='rbf'
            Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
        C : float, default=100.0
            Regularization parameter
        epsilon : float, default=0.1
            Epsilon in epsilon-insensitive loss
        gamma : str or float, default='scale'
            Kernel coefficient: 'scale' = 1/(n_features*X.var())
        tune_hyperparameters : bool, default=False
            Whether to tune hyperparameters using cross-validation
        cv_folds : int, default=5
            Number of cross-validation folds for tuning
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        
        self.best_params = {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'gamma': gamma
        }
        
        self.model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            gamma=gamma
        )
        self.is_trained = False
        self.tuning_results = None
        logger.info(f"SVR model initialized: kernel={kernel}, C={C}, "
                   f"epsilon={epsilon}, gamma={gamma}, "
                   f"tune_hyperparameters={tune_hyperparameters}")
    
    
    def train(self, X_train, y_train):
        """
        Train the SVR model, with optional hyperparameter tuning.
        
        Parameters
        ----------
        X_train : numpy array or pandas DataFrame
            Training features (n_samples, n_features)
        y_train : numpy array or pandas Series
            Training target values
            
        Returns
        -------
        None
        """
        if len(X_train) < 1:
            logger.warning("X_train is empty")
            return
        
        logger.info(f"Training SVR model. Training set size: {len(X_train)}")
        
        if self.tune_hyperparameters:
            logger.info("Performing hyperparameter tuning for SVR")
            from .hyperparameter_tuner import HyperparameterTuner
            
            tuner = HyperparameterTuner(
                model_name='SVR',
                use_small_grid=True,
                cv_folds=self.cv_folds
            )
            
            # Create base model for tuning
            base_model = SVR(
                kernel=self.kernel,
                C=self.C,
                epsilon=self.epsilon,
                gamma=self.gamma
            )
            
            tuning_results = tuner.tune_model(base_model, X_train, y_train, verbose=False)
            best_params = tuning_results['best_params']
            best_score = tuning_results['best_score']
            self.best_params = best_params
            self.tuning_results = tuning_results
            
            logger.info(f"Best hyperparameters found: {best_params}")
            if best_score is not None:
                logger.info(f"Best cross-validation score: {best_score:.4f}")
            
            # Train final model with best parameters
            self.model = SVR(
                kernel=best_params['kernel'],
                C=best_params['C'],
                epsilon=best_params['epsilon'],
                gamma=best_params.get('gamma', 'scale')
            )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("SVR model trained successfully")
    
    
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
    
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Get feature importance for SVR.
        
        For SVR with linear kernels, use coefficient magnitudes.
        For non-linear, use permutation importance approximation.
        
        Parameters
        ----------
        X : np.ndarray
            Feature data
            
        Returns
        -------
        np.ndarray
            Feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        if self.kernel == 'linear':
            # For linear kernel, use coefficients directly
            importance = np.abs(self.model.coef_[0])
        else:
            # For non-linear kernels, use approximate importance
            # based on support vectors and their weights
            support_vectors = self.model.support_vectors_
            dual_coef = np.abs(self.model.dual_coef_[0])
            
            # Compute importance as weighted feature contribution
            importance = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                importance[i] = np.sum(dual_coef * np.abs(support_vectors[:, i]))
        
        return importance / importance.sum()  # Normalize
    
    
    def get_model_info(self) -> dict:
        """
        Get information about the model.
        
        Returns
        -------
        dict
            Model name, parameters, training status, and tuning results
        """
        info = {
            'model_name': 'SVR',
            'kernel': self.model.kernel,
            'C': self.model.C,
            'epsilon': self.model.epsilon,
            'gamma': self.model.gamma,
            'is_trained': self.is_trained,
        }
        
        if self.tuning_results:
            info['tuning_enabled'] = True
            info['best_hyperparameters'] = self.best_params
            info['tuning_cv_score'] = self.tuning_results.get('best_score', None)
        else:
            info['tuning_enabled'] = False
        
        return info
