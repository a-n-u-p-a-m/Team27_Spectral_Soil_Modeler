"""
KRR Model Module
================
Kernel Ridge Regression implementation.

KRR combines ridge regression with kernel methods to handle non-linear relationships
while maintaining computational efficiency through kernel representations.
"""

from sklearn.kernel_ridge import KernelRidge
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class KRRModel:
    """
    Kernel Ridge Regression for spectral soil property prediction.
    
    KRR benefits:
    - Handles non-linear relationships through kernel methods
    - Computationally efficient compared to SVMs
    - Good regularization through ridge penalty
    - Works well with high-dimensional spectral data
    
    Attributes:
        model: Fitted KernelRidge object
        kernel: Kernel type ('rbf', 'linear', 'poly')
        is_trained: Whether model has been trained
    """
    
    def __init__(self, alpha: float = 1.0, kernel: str = 'rbf',
                 gamma: Optional[float] = None, tune_hyperparameters: bool = False,
                 cv_folds: int = 5, cv_strategy: str = 'k-fold', search_method: str = 'grid', n_iter: int = 20):
        """
        Initialize KRR model.
        
        Parameters
        ----------
        alpha : float, default=1.0
            Regularization parameter
        kernel : str, default='rbf'
            Kernel type: 'rbf', 'linear', 'poly', 'sigmoid'
        gamma : float, optional
            Kernel coefficient (for rbf, poly, sigmoid)
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
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        self.search_method = search_method.lower()
        self.n_iter = n_iter
        
        self.best_params = {
            'alpha': alpha,
            'kernel': kernel,
            'gamma': gamma
        }
        
        self.model = KernelRidge(
            alpha=alpha,
            kernel=kernel,
            gamma=gamma
        )
        self.is_trained = False
        self.tuning_results = None
        logger.info(f"KRR model initialized: alpha={alpha}, kernel={kernel}, gamma={gamma}, "
                   f"tune_hyperparameters={tune_hyperparameters}")
    
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'KRRModel':
        """
        Train KRR model, with optional hyperparameter tuning.
        
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
                logger.info("Performing hyperparameter tuning for KRR")
                from .hyperparameter_tuner import HyperparameterTuner
                
                tuner = HyperparameterTuner(
                    model_name='KRR',
                    search_type=self.search_method,
                    use_small_grid=(self.search_method == 'random'),
                    n_iter=self.n_iter,
                    cv_folds=self.cv_folds,
                    cv_strategy=self.cv_strategy
                )
                
                # Create base model for tuning
                base_model = KernelRidge(
                    alpha=self.alpha,
                    kernel=self.kernel,
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
                self.model = KernelRidge(
                    alpha=best_params['alpha'],
                    kernel=best_params['kernel'],
                    gamma=best_params.get('gamma', None)
                )
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            logger.info(f"KRR model trained. Training R²: {self.model.score(X_train, y_train):.4f}")
            return self
        except Exception as e:
            logger.error(f"Error training KRR model: {str(e)}")
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
    
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Get feature importance for KRR using permutation importance concept.
        
        For KRR, we use the absolute values of coefficients in dual space.
        
        Parameters
        ----------
        X : np.ndarray
            Feature data
            
        Returns
        -------
        np.ndarray
            Feature importance scores (approximate)
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        # Use dual coefficients weighted by kernel contributions
        # This is an approximation of importance
        dual_coef = np.abs(self.model.dual_coef_)
        
        # Simple importance: average influence across samples
        importance = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            importance[i] = np.mean(np.abs(X[:, i] * dual_coef))
        
        return importance / importance.sum()  # Normalize
    
    
    def get_model_info(self) -> Dict:
        """Get model information, including tuning results if available."""
        info = {
            'name': 'KRR',
            'alpha': self.model.alpha,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'is_trained': self.is_trained,
            'model_type': 'Kernel Ridge Regression'
        }
        
        if self.tuning_results:
            info['tuning_enabled'] = True
            info['best_hyperparameters'] = self.best_params
            info['tuning_cv_score'] = self.tuning_results.get('best_score', None)
        else:
            info['tuning_enabled'] = False
        
        return info
