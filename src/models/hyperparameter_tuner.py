"""
Hyperparameter Tuner Module
============================
Provides hyperparameter optimization for all regression models using
GridSearchCV and RandomizedSearchCV approaches.
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter optimization for regression models.
    
    Provides both grid search and randomized search approaches for
    finding optimal hyperparameters.
    """
    
    # Predefined hyperparameter grids for different models
    PARAM_GRIDS = {
        'PLSR': {
            'n_components': [3, 5, 8, 10, 12, 15, 20]
        },
        'GBRT': {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        },
        'SVR': {
            'kernel': ['rbf', 'poly', 'linear'],
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.01, 0.1, 0.5],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        },
        'KRR': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear', 'polynomial'],
            'gamma': [None, 0.001, 0.01, 0.1, 1.0]
        },
        'Cubist': {
            'n_rules': [5, 10, 15, 20, 25],
            'neighbors': [0, 3, 5, 7, 9]
        }
    }
    
    # Smaller grids for faster search
    PARAM_GRIDS_SMALL = {
        'PLSR': {
            'n_components': [5, 10, 15]
        },
        'GBRT': {
            'n_estimators': [100, 150],
            'learning_rate': [0.01, 0.1],
            'max_depth': [4, 5, 6],
            'min_samples_split': [2, 5],
            'subsample': [0.9, 1.0]
        },
        'SVR': {
            'kernel': ['rbf', 'linear'],
            'C': [1, 100],
            'epsilon': [0.1, 0.5],
            'gamma': ['scale', 'auto']
        },
        'KRR': {
            'alpha': [0.01, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': [None, 0.01, 0.1]
        },
        'Cubist': {
            'n_rules': [10, 20],
            'neighbors': [3, 5]
        }
    }
    
    def __init__(self, model_name: str, cv_folds: int = 5, scoring: str = 'r2',
                 search_type: str = 'grid', use_small_grid: bool = True, n_iter: int = 20):
        """
        Initialize hyperparameter tuner.
        
        Parameters
        ----------
        model_name : str
            Name of model: 'PLSR', 'GBRT', 'SVR', 'KRR', 'Cubist'
        cv_folds : int, default=5
            Number of cross-validation folds
        scoring : str, default='r2'
            Scoring metric: 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
        search_type : str, default='grid'
            Search type: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        use_small_grid : bool, default=True
            Use smaller grid for faster optimization
        n_iter : int, default=20
            Number of iterations for RandomizedSearchCV
        """
        self.model_name = model_name
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.search_type = search_type
        self.use_small_grid = use_small_grid
        self.n_iter = n_iter
        
        # Get parameter grid
        if use_small_grid:
            self.param_grid = self.PARAM_GRIDS_SMALL.get(model_name, {})
        else:
            self.param_grid = self.PARAM_GRIDS.get(model_name, {})
        
        self.best_params = None
        self.best_score = None
        self.cv_results = None
        
        logger.info(f"HyperparameterTuner initialized for {model_name}")
    
    
    def tune_model(self, base_model, X_train: np.ndarray, y_train: np.ndarray,
                   verbose: bool = True) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning on the model.
        
        Parameters
        ----------
        base_model : sklearn model
            Base model to tune (without hyperparameters set)
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        verbose : bool, default=True
            Print tuning progress
            
        Returns
        -------
        dict
            Dictionary with 'best_params', 'best_score', 'n_candidates', 'cv_results'
        """
        try:
            if not self.param_grid:
                logger.warning(f"No parameter grid found for {self.model_name}. "
                             "Using default parameters.")
                return {
                    'best_params': {},
                    'best_score': None,
                    'n_candidates': 0,
                    'cv_results': None,
                    'tuned': False
                }
            
            if self.search_type == 'random':
                search = RandomizedSearchCV(
                    base_model,
                    self.param_grid,
                    n_iter=self.n_iter,
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=-1,
                    verbose=1 if verbose else 0,
                    random_state=42
                )
                logger.info(f"Starting RandomizedSearchCV for {self.model_name} "
                          f"with {self.n_iter} iterations")
            else:
                search = GridSearchCV(
                    base_model,
                    self.param_grid,
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=-1,
                    verbose=1 if verbose else 0
                )
                logger.info(f"Starting GridSearchCV for {self.model_name}")
            
            # Perform search
            search.fit(X_train, y_train)
            
            self.best_params = search.best_params_
            self.best_score = search.best_score_
            self.cv_results = search.cv_results_
            
            if verbose:
                logger.info(f"Best hyperparameters for {self.model_name}: {self.best_params}")
                logger.info(f"Best CV score: {self.best_score:.4f}")
                
                # Print top 3 results
                results_df = self._get_top_results(n_top=3)
                print(f"\n📊 Top 3 Hyperparameter Combinations for {self.model_name}:")
                print(results_df.to_string())
            
            return {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'n_candidates': search.n_candidates_ if hasattr(search, 'n_candidates_') else len(search.cv_results_['params']),
                'cv_results': search.cv_results_,
                'tuned': True
            }
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            return {
                'best_params': {},
                'best_score': None,
                'n_candidates': 0,
                'cv_results': None,
                'tuned': False,
                'error': str(e)
            }
    
    
    def _get_top_results(self, n_top: int = 3) -> 'pd.DataFrame':
        """
        Get top N results from cross-validation.
        
        Parameters
        ----------
        n_top : int, default=3
            Number of top results to return
            
        Returns
        -------
        pd.DataFrame
            DataFrame with top results
        """
        import pandas as pd
        
        if self.cv_results is None:
            return None
        
        results_df = pd.DataFrame(self.cv_results)
        results_df = results_df.sort_values('mean_test_score', ascending=False)
        
        # Select relevant columns
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        cols_to_show = param_cols + ['mean_test_score', 'std_test_score']
        
        return results_df[cols_to_show].head(n_top)
    
    
    def get_tuning_summary(self) -> Dict[str, Any]:
        """
        Get summary of tuning results.
        
        Returns
        -------
        dict
            Summary statistics of tuning
        """
        if self.cv_results is None:
            return None
        
        mean_scores = self.cv_results['mean_test_score']
        std_scores = self.cv_results['std_test_score']
        
        return {
            'model': self.model_name,
            'best_score': self.best_score,
            'mean_cv_score': np.mean(mean_scores),
            'std_cv_score': np.std(mean_scores),
            'score_range': (np.min(mean_scores), np.max(mean_scores)),
            'best_params': self.best_params,
            'n_combinations_tested': len(mean_scores)
        }


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning.
    Uses expected improvement to efficiently search parameter space.
    """
    
    def __init__(self, model_name: str, cv_folds: int = 5, n_calls: int = 20):
        """
        Initialize Bayesian optimizer.
        
        Parameters
        ----------
        model_name : str
            Name of model
        cv_folds : int, default=5
            Number of cross-validation folds
        n_calls : int, default=20
            Number of function calls
        """
        self.model_name = model_name
        self.cv_folds = cv_folds
        self.n_calls = n_calls
        
        try:
            from skopt import gp_minimize  # type: ignore
            from skopt.space import Real, Integer, Categorical  # type: ignore
            self.gp_minimize = gp_minimize
            self.Real = Real
            self.Integer = Integer
            self.Categorical = Categorical
            self.has_skopt = True
        except ImportError:
            logger.warning("scikit-optimize not installed. Bayesian optimization unavailable.")
            self.has_skopt = False
    
    
    def optimize(self, base_model, X_train: np.ndarray, y_train: np.ndarray,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Perform Bayesian optimization.
        
        Parameters
        ----------
        base_model : sklearn model
            Base model to optimize
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        verbose : bool, default=True
            Print optimization progress
            
        Returns
        -------
        dict
            Optimization results
        """
        if not self.has_skopt:
            logger.warning("scikit-optimize not available. Using grid search instead.")
            tuner = HyperparameterTuner(self.model_name, self.cv_folds)
            return tuner.tune_model(base_model, X_train, y_train, verbose)
        
        # For now, return a message that Bayesian optimization is available
        logger.info(f"Bayesian optimization available for {self.model_name}")
        return {'status': 'Bayesian optimization available but requires additional setup'}
