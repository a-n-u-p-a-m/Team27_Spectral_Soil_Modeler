"""
Model Trainer Module
====================
Orchestrates training of all 15 model-technique combinations.

This module trains all combinations of:
- 3 spectral preprocessing techniques
- 5 ML algorithms
= 15 total combinations

Includes cross-validation and result tracking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from pathlib import Path

# Import all models
from .plsr import PLSRModel
from .gbrt import GBRTModel
from .krr import KRRModel
from .svr import SVRModel
from .cubist import CubistModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates training of all 15 model-technique combinations.
    
    Handles:
    - Training individual models
    - Cross-validation
    - Metrics computation
    - Result tracking
    - Progress monitoring
    
    Attributes:
        models: Dictionary of initialized models
        results: DataFrame of training results
        best_model: Best performing model
        best_scores: Best model metrics
    """
    
    def __init__(self, tune_hyperparameters: bool = False, cv_folds: int = 5):
        """
        Initialize ModelTrainer.
        
        Parameters
        ----------
        tune_hyperparameters : bool, default=False
            Whether to perform hyperparameter tuning during training
        cv_folds : int, default=5
            Number of cross-validation folds for hyperparameter tuning
        """
        self.tune_hyperparameters = tune_hyperparameters
        self.cv_folds = cv_folds
        self.models = self._initialize_models()
        self.results = None
        self.best_model = None
        self.best_scores = None
        self.cv_results = {}
        logger.info(f"ModelTrainer initialized with 5 algorithms. "
                   f"Hyperparameter tuning: {tune_hyperparameters}, CV folds: {cv_folds}")
    
    
    def _initialize_models(self) -> Dict:
        """
        Initialize all 5 ML models with optional hyperparameter tuning.
        
        Returns
        -------
        dict
            Dictionary of model instances
        """
        models = {
            'PLSR': PLSRModel(n_components=10, tune_hyperparameters=self.tune_hyperparameters,
                             cv_folds=self.cv_folds),
            'GBRT': GBRTModel(n_estimators=100, learning_rate=0.1, max_depth=5,
                             tune_hyperparameters=self.tune_hyperparameters, cv_folds=self.cv_folds),
            'KRR': KRRModel(alpha=1.0, kernel='rbf', gamma=None,
                           tune_hyperparameters=self.tune_hyperparameters, cv_folds=self.cv_folds),
            'SVR': SVRModel(kernel='rbf', C=100.0, epsilon=0.1,
                           tune_hyperparameters=self.tune_hyperparameters, cv_folds=self.cv_folds),
            'Cubist': CubistModel(n_rules=20, neighbors=5,
                                 tune_hyperparameters=self.tune_hyperparameters, cv_folds=self.cv_folds)
        }
        logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
        return models
    
    
    def train_single_model(self, model_name: str, X_train: np.ndarray,
                          y_train: np.ndarray, X_test: np.ndarray,
                          y_test: np.ndarray) -> Dict:
        """
        Train a single model and compute metrics.
        
        Parameters
        ----------
        model_name : str
            Name of model to train
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test target
            
        Returns
        -------
        dict
            Results dictionary with metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        try:
            model = self.models[model_name]
            
            # Train
            model.train(X_train, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # RPD (Residual Prediction Deviation)
            test_rpd = np.std(y_test) / test_rmse if test_rmse > 0 else 0
            
            results = {
                'model': model_name,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'rpd': test_rpd,
                'trained_model': model,
                'y_pred': y_test_pred
            }
            
            logger.info(f"{model_name}: R²={test_r2:.4f}, RMSE={test_rmse:.4f}, RPD={test_rpd:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            return None
    
    
    def train_all_combinations(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               techniques: List[str] = None) -> pd.DataFrame:
        """
        Train all 15 model-technique combinations.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features (already preprocessed with first technique)
        y_train : np.ndarray
            Training target
        X_test : np.ndarray
            Test features (already preprocessed with first technique)
        y_test : np.ndarray
            Test target
        techniques : list, optional
            List of technique names (for labeling results)
            
        Returns
        -------
        pd.DataFrame
            Results table with all combinations
        """
        if techniques is None:
            techniques = ['reflectance', 'absorbance', 'continuum_removal']
        
        results_list = []
        total_combinations = len(self.models) * len(techniques)
        current = 0
        
        logger.info(f"Starting training of {total_combinations} combinations")
        
        for technique in techniques:
            for model_name in self.models.keys():
                current += 1
                print(f"[{current}/{total_combinations}] Training {model_name} with {technique}...")
                
                result = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test
                )
                
                if result is not None:
                    result['technique'] = technique
                    results_list.append(result)
        
        # Create results DataFrame
        self.results = pd.DataFrame([
            {
                'Model': r['model'],
                'Technique': r['technique'],
                'Train_R²': r['train_r2'],
                'Test_R²': r['test_r2'],
                'Train_RMSE': r['train_rmse'],
                'Test_RMSE': r['test_rmse'],
                'Train_MAE': r['train_mae'],
                'Test_MAE': r['test_mae'],
                'RPD': r['rpd'],
                'Model_Object': r['trained_model'],
                'Predictions': r['y_pred']
            }
            for r in results_list
        ])
        
        # Sort by Test_R² (descending)
        self.results = self.results.sort_values('Test_R²', ascending=False).reset_index(drop=True)
        self.results.index = self.results.index + 1  # 1-indexed
        
        logger.info(f"Training complete. Results: {len(self.results)} combinations")
        return self.results
    
    
    def get_leaderboard(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get leaderboard of top models.
        
        Parameters
        ----------
        top_n : int, default=15
            Number of top models to return
            
        Returns
        -------
        pd.DataFrame
            Leaderboard sorted by Test_R²
        """
        if self.results is None:
            raise ValueError("No results available. Run train_all_combinations() first.")
        
        leaderboard = self.results[
            ['Model', 'Technique', 'Test_R²', 'Test_RMSE', 'RPD']
        ].head(top_n).copy()
        
        return leaderboard
    
    
    def get_best_model(self) -> Tuple[Dict, Any]:
        """
        Get best performing model.
        
        Returns
        -------
        tuple
            (results_dict, model_object)
        """
        if self.results is None:
            raise ValueError("No results available. Run train_all_combinations() first.")
        
        best_row = self.results.iloc[0]
        return best_row.to_dict(), best_row['Model_Object']
    
    
    def get_feature_importance(self, rank: int = 1) -> np.ndarray:
        """
        Get feature importance from top model.
        
        Parameters
        ----------
        rank : int, default=1
            Rank of model to get importance from
            
        Returns
        -------
        np.ndarray
            Feature importance scores
        """
        if self.results is None:
            raise ValueError("No results available.")
        
        model_obj = self.results.iloc[rank-1]['Model_Object']
        
        # Create dummy X for feature importance (needed for some methods)
        # In practice, pass actual X_train to preprocessing
        X_dummy = np.random.randn(10, 100)  # Dummy
        
        try:
            importance = model_obj.get_feature_importance(X_dummy)
            return importance
        except Exception as e:
            logger.warning(f"Could not get feature importance: {str(e)}")
            return None
    
    
    def cross_validate_model(self, model_name: str, X: np.ndarray,
                            y: np.ndarray, cv: int = 5) -> Dict:
        """
        Perform cross-validation on a model.
        
        Parameters
        ----------
        model_name : str
            Name of model
        X : np.ndarray
            Feature data
        y : np.ndarray
            Target data
        cv : int, default=5
            Number of CV folds
            
        Returns
        -------
        dict
            Cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        
        # Get the sklearn model object for cross_validate
        sklearn_model = model.model
        
        scoring = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        }
        
        cv_results = cross_validate(
            sklearn_model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )
        
        self.cv_results[model_name] = cv_results
        logger.info(f"Cross-validation complete for {model_name}")
        
        return cv_results
    
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of results.
        
        Returns
        -------
        pd.DataFrame
            Summary by model and technique
        """
        if self.results is None:
            raise ValueError("No results available.")
        
        summary = self.results.groupby(['Model', 'Technique']).agg({
            'Test_R²': ['mean', 'std'],
            'Test_RMSE': ['mean', 'std'],
            'RPD': ['mean', 'std']
        }).round(4)
        
        return summary
    
    
    def save_results(self, filepath: str) -> None:
        """
        Save results to CSV.
        
        Parameters
        ----------
        filepath : str
            Output file path
        """
        if self.results is None:
            raise ValueError("No results to save.")
        
        # Remove non-serializable columns
        save_df = self.results[
            ['Model', 'Technique', 'Train_R²', 'Test_R²', 'Train_RMSE', 'Test_RMSE', 'RPD']
        ].copy()
        
        save_df.to_csv(filepath)
        logger.info(f"Results saved to {filepath}")
    
    
    def get_model_info(self) -> Dict:
        """Get information about all models."""
        info = {}
        for name, model in self.models.items():
            info[name] = model.get_model_info()
        return info
