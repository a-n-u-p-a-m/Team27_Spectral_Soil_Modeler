"""
Model Analysis Module
======================
Detailed model statistics, parameter inspection, and comparison utilities.

Features:
- Per-model performance statistics
- Model parameter inspection
- Hyperparameter tuning history
- Model comparison utilities
- Performance breakdown by technique
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """Analyze individual model performance and parameters."""
    
    @staticmethod
    def get_model_statistics(results_df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific model.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
        model_name : str
            Model name to analyze
            
        Returns
        -------
        Dict[str, Any]
            Model statistics
        """
        model_data = results_df[results_df['Model'] == model_name]
        
        if model_data.empty:
            return {'error': f'Model {model_name} not found'}
        
        stats = {
            'model_name': model_name,
            'total_runs': len(model_data),
            'techniques_used': model_data['Technique'].unique().tolist(),
            'performance': {
                'best_r2': float(model_data['Test_R²'].max()),
                'worst_r2': float(model_data['Test_R²'].min()),
                'mean_r2': float(model_data['Test_R²'].mean()),
                'median_r2': float(model_data['Test_R²'].median()),
                'std_r2': float(model_data['Test_R²'].std()),
                'variance_r2': float(model_data['Test_R²'].var()),
            },
            'by_technique': {},
        }
        
        # Performance breakdown by technique
        for technique in model_data['Technique'].unique():
            tech_data = model_data[model_data['Technique'] == technique]
            stats['by_technique'][technique] = {
                'count': len(tech_data),
                'mean_r2': float(tech_data['Test_R²'].mean()),
                'max_r2': float(tech_data['Test_R²'].max()),
                'min_r2': float(tech_data['Test_R²'].min()),
                'rmse': float(tech_data['Test_RMSE'].mean()) if 'Test_RMSE' in tech_data.columns else None,
                'mae': float(tech_data['Test_MAE'].mean()) if 'Test_MAE' in tech_data.columns else None,
            }
        
        return stats
    
    
    @staticmethod
    def get_technique_statistics(results_df: pd.DataFrame, technique: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific technique.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
        technique : str
            Technique to analyze
            
        Returns
        -------
        Dict[str, Any]
            Technique statistics
        """
        tech_data = results_df[results_df['Technique'] == technique]
        
        if tech_data.empty:
            return {'error': f'Technique {technique} not found'}
        
        stats = {
            'technique': technique,
            'total_models': len(tech_data),
            'models_used': tech_data['Model'].unique().tolist(),
            'performance': {
                'best_r2': float(tech_data['Test_R²'].max()),
                'worst_r2': float(tech_data['Test_R²'].min()),
                'mean_r2': float(tech_data['Test_R²'].mean()),
                'median_r2': float(tech_data['Test_R²'].median()),
                'std_r2': float(tech_data['Test_R²'].std()),
            },
            'by_model': {},
        }
        
        # Performance breakdown by model
        for model in tech_data['Model'].unique():
            model_data = tech_data[tech_data['Model'] == model]
            stats['by_model'][model] = {
                'count': len(model_data),
                'mean_r2': float(model_data['Test_R²'].mean()),
                'max_r2': float(model_data['Test_R²'].max()),
                'min_r2': float(model_data['Test_R²'].min()),
            }
        
        return stats
    
    
    @staticmethod
    def get_combination_statistics(results_df: pd.DataFrame,
                                   model: str, technique: str) -> Dict[str, Any]:
        """
        Get statistics for a specific model-technique combination.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
        model : str
            Model name
        technique : str
            Technique name
            
        Returns
        -------
        Dict[str, Any]
            Combination statistics
        """
        combo_data = results_df[
            (results_df['Model'] == model) & (results_df['Technique'] == technique)
        ]
        
        if combo_data.empty:
            return {'error': f'Combination {model}-{technique} not found'}
        
        row = combo_data.iloc[0]
        
        stats = {
            'model': model,
            'technique': technique,
            'metrics': {
                'test_r2': float(row.get('Test_R²', 0)),
                'test_rmse': float(row.get('Test_RMSE', 0)),
                'test_mae': float(row.get('Test_MAE', 0)),
                'test_mape': float(row.get('Test_MAPE', 0)),
                'rpd': float(row.get('RPD', 0)),
                'train_r2': float(row.get('Train_R²', 0)),
                'train_rmse': float(row.get('Train_RMSE', 0)),
            },
            'quality_assessment': ModelAnalyzer._assess_model_quality(row),
        }
        
        return stats
    
    
    @staticmethod
    def _assess_model_quality(row: pd.Series) -> str:
        """Assess model quality based on metrics."""
        r2 = row.get('Test_R²', 0)
        
        if r2 > 0.9:
            return 'Excellent'
        elif r2 > 0.75:
            return 'Very Good'
        elif r2 > 0.6:
            return 'Good'
        elif r2 > 0.4:
            return 'Fair'
        else:
            return 'Poor'
    
    
    @staticmethod
    def compare_models(results_df: pd.DataFrame, model_list: List[str]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
        model_list : List[str]
            Models to compare
            
        Returns
        -------
        pd.DataFrame
            Comparison dataframe
        """
        comparison_data = []
        
        for model in model_list:
            model_results = results_df[results_df['Model'] == model]
            if not model_results.empty:
                comparison_data.append({
                    'Model': model,
                    'Count': len(model_results),
                    'Best R²': model_results['Test_R²'].max(),
                    'Mean R²': model_results['Test_R²'].mean(),
                    'Std R²': model_results['Test_R²'].std(),
                    'RMSE': model_results['Test_RMSE'].mean() if 'Test_RMSE' in model_results.columns else np.nan,
                    'MAE': model_results['Test_MAE'].mean() if 'Test_MAE' in model_results.columns else np.nan,
                })
        
        return pd.DataFrame(comparison_data)
    
    
    @staticmethod
    def compare_techniques(results_df: pd.DataFrame, technique_list: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple techniques.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
        technique_list : List[str], optional
            Techniques to compare. If None, uses all techniques
            
        Returns
        -------
        pd.DataFrame
            Comparison dataframe
        """
        if technique_list is None:
            technique_list = results_df['Technique'].unique().tolist()
        
        comparison_data = []
        
        for technique in technique_list:
            tech_results = results_df[results_df['Technique'] == technique]
            if not tech_results.empty:
                comparison_data.append({
                    'Technique': technique,
                    'Count': len(tech_results),
                    'Best R²': tech_results['Test_R²'].max(),
                    'Mean R²': tech_results['Test_R²'].mean(),
                    'Std R²': tech_results['Test_R²'].std(),
                    'RMSE': tech_results['Test_RMSE'].mean() if 'Test_RMSE' in tech_results.columns else np.nan,
                    'MAE': tech_results['Test_MAE'].mean() if 'Test_MAE' in tech_results.columns else np.nan,
                })
        
        return pd.DataFrame(comparison_data)


class ParameterInspector:
    """Inspect and analyze model parameters."""
    
    @staticmethod
    def extract_parameters(model) -> Dict[str, Any]:
        """
        Extract parameters from a trained model.
        
        Parameters
        ----------
        model : object
            Trained model instance
            
        Returns
        -------
        Dict[str, Any]
            Model parameters
        """
        params = {}
        
        # Try different attribute names based on model type
        if hasattr(model, 'get_params'):
            params['parameters'] = model.get_params()
        elif hasattr(model, 'coef_'):
            params['coefficients'] = model.coef_
        elif hasattr(model, 'feature_importances_'):
            params['feature_importances'] = model.feature_importances_
        elif hasattr(model, 'intercept_'):
            params['intercept'] = model.intercept_
        
        return params
    
    
    @staticmethod
    def get_hyperparameters(model) -> Dict[str, Any]:
        """
        Get hyperparameters from a model.
        
        Parameters
        ----------
        model : object
            Model instance
            
        Returns
        -------
        Dict[str, Any]
            Hyperparameters
        """
        hyperparams = {}
        
        # Try primary method: get_params()
        if hasattr(model, 'get_params'):
            try:
                all_params = model.get_params()
                # Filter out non-hyperparameter attributes
                hyperparams = {k: v for k, v in all_params.items() 
                             if not k.startswith('_')}
                # If we found params, return them
                if hyperparams:
                    return hyperparams
            except Exception as e:
                logger.warning(f"Could not get params: {e}")
        
        # Fallback: Try to extract from __dict__ or specific attributes
        if hasattr(model, '__dict__'):
            # Get all instance attributes
            instance_attrs = {}
            for key, value in model.__dict__.items():
                if not key.startswith('_'):
                    instance_attrs[key] = value
            
            # Filter to likely hyperparameters (exclude fitted attributes)
            fitted_attrs = ['coef_', 'intercept_', 'feature_importances_', 
                          'n_features_in_', 'n_features_', '_estimator_type']
            
            for attr in fitted_attrs:
                instance_attrs.pop(attr, None)
            
            if instance_attrs:
                hyperparams = instance_attrs
        
        # Fallback 2: Try wrapper detection (Pipeline, GridSearchCV, etc.)
        if hasattr(model, 'best_params_'):
            # GridSearchCV or similar
            hyperparams['best_params'] = model.best_params_
        
        if hasattr(model, 'steps'):
            # Pipeline - try to get params from steps
            hyperparams['pipeline_steps'] = str(model.steps)
        
        return hyperparams
    
    
    @staticmethod
    def format_parameters_for_display(params: Dict[str, Any],
                                     max_depth: int = 2) -> Dict[str, str]:
        """
        Format parameters for display in UI.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to format
        max_depth : int
            Maximum nesting depth
            
        Returns
        -------
        Dict[str, str]
            Formatted parameters
        """
        formatted = {}
        
        for key, value in params.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 10:
                    formatted[key] = f"Array of shape {np.array(value).shape}"
                else:
                    formatted[key] = str(value)
            elif isinstance(value, dict):
                formatted[key] = f"Dict with keys: {list(value.keys())}"
            elif isinstance(value, float):
                formatted[key] = f"{value:.6f}"
            else:
                formatted[key] = str(value)
        
        return formatted


class PerformanceComparator:
    """Compare performance across different model configurations."""
    
    @staticmethod
    def create_comparison_matrix(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a pivot table showing performance by model and technique.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
            
        Returns
        -------
        pd.DataFrame
            Comparison matrix
        """
        matrix = results_df.pivot_table(
            index='Model',
            columns='Technique',
            values='Test_R²',
            aggfunc='mean'
        )
        
        return matrix
    
    
    @staticmethod
    def find_best_combination(results_df: pd.DataFrame) -> Tuple[str, str, float]:
        """
        Find the best model-technique combination.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
            
        Returns
        -------
        Tuple[str, str, float]
            (model, technique, r2_score)
        """
        best_idx = results_df['Test_R²'].idxmax()
        best_row = results_df.loc[best_idx]
        
        return (
            best_row['Model'],
            best_row['Technique'],
            float(best_row['Test_R²'])
        )
    
    
    @staticmethod
    def find_most_consistent_model(results_df: pd.DataFrame) -> str:
        """
        Find the model with most consistent performance across techniques.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
            
        Returns
        -------
        str
            Model name
        """
        consistency = results_df.groupby('Model')['Test_R²'].std()
        most_consistent = consistency.idxmin()
        
        return most_consistent
    
    
    @staticmethod
    def find_best_technique(results_df: pd.DataFrame) -> str:
        """
        Find the best performing technique overall.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
            
        Returns
        -------
        str
            Technique name
        """
        best_technique = results_df.groupby('Technique')['Test_R²'].mean().idxmax()
        return best_technique
    
    
    @staticmethod
    def calculate_ranking(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ranking of all model-technique combinations.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Results dataframe
            
        Returns
        -------
        pd.DataFrame
            Ranked results
        """
        ranked = results_df.copy()
        ranked['Rank'] = ranked['Test_R²'].rank(ascending=False, method='min')
        ranked = ranked.sort_values('Test_R²', ascending=False)
        
        return ranked[['Rank', 'Model', 'Technique', 'Test_R²', 'Test_RMSE', 'Test_MAE']]
