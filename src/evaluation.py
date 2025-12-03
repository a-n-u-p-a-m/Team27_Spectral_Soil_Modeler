"""
Evaluation Module
=================
Comprehensive model evaluation, metrics computation, and feature analysis.

Features:
- Detailed performance metrics
- Feature importance analysis
- Residual analysis
- Model comparison
- Leaderboard generation
- Performance visualization data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import LeaveOneOut
from sklearn.base import clone as sklearn_clone
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis.
    
    Computes:
    - Performance metrics (R², RMSE, MAE, MAPE, RPD)
    - Residual statistics (mean, std, distribution)
    - Feature importance analysis
    - Model rankings and comparisons
    - Confidence intervals
    
    Attributes:
        results_df: DataFrame with all model results
        metrics_extended: DataFrame with extended metrics
        feature_importance: DataFrame with feature rankings
    """
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.results_df = None
        self.metrics_extended = None
        self.feature_importance = None
        self.residuals = {}
        logger.info("ModelEvaluator initialized")
    
    
    def compute_extended_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 model_name: str = None) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            Actual values
        y_pred : np.ndarray
            Predicted values
        model_name : str, optional
            Name of model for logging
            
        Returns
        -------
        dict
            Dictionary with all computed metrics
        """
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rpd = np.std(y_true) / rmse if rmse > 0 else 0  # Residual Prediction Deviation
        
        # Residual analysis
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Quantile-based metrics
        q25, q50, q75 = np.percentile(residuals, [25, 50, 75])
        iqr = q75 - q25
        
        # Max absolute error
        max_error = np.max(np.abs(residuals))
        median_absolute_error = np.median(np.abs(residuals))
        
        # Symmetric MAPE
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / 
                              (np.abs(y_true) + np.abs(y_pred)))
        
        # Calculate range-based metrics
        y_range = np.max(y_true) - np.min(y_true)
        nrmse = rmse / y_range if y_range > 0 else 0  # Normalized RMSE
        
        metrics = {
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'SMAPE': smape,
            'RPD': rpd,
            'NRMSE': nrmse,
            'Max_Error': max_error,
            'Median_AE': median_absolute_error,
            'Mean_Residual': mean_residual,
            'Std_Residual': std_residual,
            'Residual_Q25': q25,
            'Residual_Median': q50,
            'Residual_Q75': q75,
            'Residual_IQR': iqr
        }
        
        if model_name:
            self.residuals[model_name] = residuals
            logger.info(f"{model_name}: R²={r2:.4f}, RMSE={rmse:.4f}, RPD={rpd:.2f}")
        
        return metrics
    
    
    def evaluate_models(self, trainer_results: pd.DataFrame,
                       y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all models using trainer results.
        
        Parameters
        ----------
        trainer_results : pd.DataFrame
            Results from ModelTrainer (must have 'Model', 'Technique', 'Predictions' columns)
        y_test : np.ndarray
            Ground truth test values
            
        Returns
        -------
        pd.DataFrame
            Extended evaluation metrics for all models (includes trainer results + extended metrics)
        """
        extended_metrics_list = []
        
        for idx, row in trainer_results.iterrows():
            model_name = row['Model']
            technique = row['Technique']
            y_pred = row['Predictions']
            
            # Compute additional metrics
            metrics = self.compute_extended_metrics(
                y_test, y_pred,
                model_name=f"{model_name}_{technique}"
            )
            
            # Rename metrics to add 'Test_' prefix for clarity
            metrics_renamed = {}
            for key, value in metrics.items():
                metrics_renamed[f'Test_{key}'] = value
            
            # Add identifying info from trainer results
            metrics_renamed['Model'] = model_name
            metrics_renamed['Technique'] = technique
            metrics_renamed['Index'] = idx
            
            # Preserve key trainer metrics
            if 'Train_R²' in row:
                metrics_renamed['Train_R²'] = row['Train_R²']
            if 'Test_RMSE' in row:
                # Use trainer's test RMSE if available, otherwise use our computed one
                metrics_renamed['Test_RMSE'] = row.get('Test_RMSE', metrics_renamed.get('Test_RMSE'))
            if 'Test_MAE' in row:
                metrics_renamed['Test_MAE'] = row.get('Test_MAE', metrics_renamed.get('Test_MAE'))
            if 'RPD' in row:
                metrics_renamed['RPD'] = row['RPD']
            
            extended_metrics_list.append(metrics_renamed)
        
        self.metrics_extended = pd.DataFrame(extended_metrics_list)
        return self.metrics_extended


    def leave_one_out_evaluation(self,
                                 model,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 feature_names: Optional[List[str]] = None,
                                 model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform Leave-One-Out cross-validation for a given estimator.

        Parameters
        ----------
        model : estimator
            A scikit-learn-compatible estimator (implements fit/predict).
        X : np.ndarray or pd.DataFrame
            Feature matrix (n_samples, n_features).
        y : np.ndarray or pd.Series
            Target vector (n_samples,).
        feature_names : list[str], optional
            Optional feature names for downstream reporting.
        model_name : str, optional
            Optional name used when storing residuals.

        Returns
        -------
        dict
            Dictionary containing LOO predictions, true values and aggregated metrics.
        """
        # Ensure numpy arrays
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = np.asarray(X)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = np.asarray(y).ravel()
        else:
            y_arr = np.asarray(y).ravel()

        loo = LeaveOneOut()
        preds = np.zeros_like(y_arr, dtype=float)

        n_splits = 0
        for train_idx, test_idx in loo.split(X_arr):
            n_splits += 1
            X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
            y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

            # Clone the model to avoid state carryover
            try:
                estimator = sklearn_clone(model)
            except Exception:
                estimator = model

            # Fit and predict
            try:
                estimator.fit(X_tr, y_tr)
                p = estimator.predict(X_te)
            except Exception as e:
                logger.error(f"LOO fold training/prediction error: {e}")
                p = np.array([np.nan])

            preds[test_idx] = np.asarray(p).ravel()

        # Compute aggregated metrics
        metrics = self.compute_extended_metrics(y_arr, preds, model_name=model_name)

        result = {
            'model_name': model_name or getattr(model, '__class__', type(model)).__name__,
            'n_splits': n_splits,
            'y_true': y_arr,
            'y_pred': preds,
            'metrics': metrics
        }

        # Store residuals mapping if model_name provided
        if model_name:
            residuals = y_arr - preds
            self.residuals[model_name] = residuals

        return result
    
    
    def get_feature_importance_analysis(self, model_trainer, X_train: np.ndarray,
                                       feature_names: List[str] = None) -> Dict:
        """
        Extract and analyze feature importance from best model.
        
        Parameters
        ----------
        model_trainer : ModelTrainer
            Trained ModelTrainer instance
        X_train : np.ndarray
            Training features (for importance computation)
        feature_names : list, optional
            Names of features (default: Feature_0, Feature_1, ...)
            
        Returns
        -------
        dict
            Feature importance analysis with rankings
        """
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        # Get best model
        best_result, best_model = model_trainer.get_best_model()
        
        try:
            # Get feature importance
            importance_scores = best_model.get_feature_importance(X_train)
            
            if importance_scores is None:
                logger.warning("Could not extract feature importance")
                return None
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance_scores,
                'Rank': np.arange(1, len(importance_scores) + 1)
            })
            
            # Sort by importance (descending)
            importance_df = importance_df.sort_values('Importance', ascending=False)
            importance_df['Rank'] = np.arange(1, len(importance_df) + 1)
            
            # Compute cumulative importance
            importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
            
            # Find number of features for 95% importance
            n_features_95 = np.argmax(importance_df['Cumulative_Importance'] >= 0.95) + 1
            
            self.feature_importance = importance_df
            
            analysis = {
                'best_model': best_result['Model'],
                'best_technique': best_result['Technique'],
                'feature_importance_df': importance_df,
                'top_features': importance_df.head(10),
                'n_features_for_95_importance': n_features_95,
                'total_features': len(feature_names)
            }
            
            logger.info(f"Feature importance extracted: {n_features_95} features explain 95% variance")
            return analysis
            
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return None
    
    
    def get_ranking_comparison(self, metrics_df: pd.DataFrame,
                               primary_metric: str = 'Test_R²',
                               secondary_metrics: List[str] = None) -> pd.DataFrame:
        """
        Generate comprehensive model rankings.
        
        Parameters
        ----------
        metrics_df : pd.DataFrame
            Extended metrics DataFrame
        primary_metric : str, default='Test_R²'
            Primary ranking metric
        secondary_metrics : list, optional
            Secondary ranking metrics for comparison
            
        Returns
        -------
        pd.DataFrame
            Ranking comparison table
        """
        if secondary_metrics is None:
            secondary_metrics = ['Test_RMSE', 'Test_MAE', 'RPD']
        
        # Filter secondary_metrics to only include columns that exist
        available_secondary = [m for m in secondary_metrics if m in metrics_df.columns]
        
        columns_to_select = ['Model', 'Technique', primary_metric] + available_secondary
        # Filter out duplicates and only select columns that exist
        columns_to_select = [c for c in columns_to_select if c in metrics_df.columns]
        
        ranking = metrics_df[columns_to_select].copy()
        
        # Rank by primary metric
        ranking['Rank'] = ranking[primary_metric].rank(ascending=False).astype(int)
        
        # Sort by rank
        ranking = ranking.sort_values('Rank')
        
        return ranking
    
    
    def get_residual_analysis(self, model_name: str = None) -> Dict:
        """
        Analyze residuals for a model.
        
        Parameters
        ----------
        model_name : str, optional
            Model name to analyze. If None, uses first available.
            
        Returns
        -------
        dict
            Residual statistics and analysis
        """
        if not self.residuals:
            raise ValueError("No residuals available. Run evaluate_models() first.")
        
        if model_name is None:
            model_name = list(self.residuals.keys())[0]
        
        if model_name not in self.residuals:
            raise ValueError(f"No residuals for model: {model_name}")
        
        residuals = self.residuals[model_name]
        
        # Normality test (Shapiro-Wilk)
        from scipy import stats
        if len(residuals) > 5000:
            # Sample for large datasets
            sample_residuals = np.random.choice(residuals, 5000, replace=False)
        else:
            sample_residuals = residuals
        
        try:
            statistic, p_value = stats.shapiro(sample_residuals)
            is_normal = p_value > 0.05
        except:
            statistic, p_value, is_normal = None, None, None
        
        analysis = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'normality_p_value': p_value,
            'is_normal': is_normal
        }
        
        return analysis
    
    
    def get_model_comparison_matrix(self, metrics_df: pd.DataFrame,
                                    metrics_to_compare: List[str] = None) -> pd.DataFrame:
        """
        Create comparison matrix for all models and techniques.
        
        Parameters
        ----------
        metrics_df : pd.DataFrame
            Extended metrics DataFrame
        metrics_to_compare : list, optional
            Metrics to include in comparison
            
        Returns
        -------
        pd.DataFrame
            Pivot table for easy comparison
        """
        if metrics_to_compare is None:
            metrics_to_compare = ['Test_R²', 'Test_RMSE', 'Test_MAE', 'RPD']
        
        # Create pivot tables for each metric
        comparison_tables = {}
        for metric in metrics_to_compare:
            if metric in metrics_df.columns:
                pivot = metrics_df.pivot_table(
                    index='Model',
                    columns='Technique',
                    values=metric,
                    aggfunc='mean'
                )
                comparison_tables[metric] = pivot
        
        return comparison_tables
    
    
    def get_best_models_by_metric(self, metrics_df: pd.DataFrame,
                                  metric: str = 'Test_R²',
                                  top_n: int = 5) -> pd.DataFrame:
        """
        Get best models ranked by specific metric.
        
        Parameters
        ----------
        metrics_df : pd.DataFrame
            Extended metrics DataFrame
        metric : str, default='Test_R²'
            Metric to rank by
        top_n : int, default=5
            Number of top models to return
            
        Returns
        -------
        pd.DataFrame
            Top N models by specified metric
        """
        # For metrics like RMSE, MAE (lower is better), reverse order
        ascending = metric in ['Test_RMSE', 'Test_MAE', 'Test_MAPE', 'Test_SMAPE', 'Test_NRMSE', 'Test_Max_Error',
                               'RMSE', 'MAE', 'MAPE', 'SMAPE', 'NRMSE', 'Max_Error']
        
        best = metrics_df.nlargest(top_n, metric) if not ascending \
               else metrics_df.nsmallest(top_n, metric)
        
        return best[['Model', 'Technique', metric, 'RPD']].reset_index(drop=True)
    
    
    def generate_evaluation_report(self, trainer_results: pd.DataFrame,
                                   y_test: np.ndarray,
                                   target_name: str = "Target") -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Parameters
        ----------
        trainer_results : pd.DataFrame
            Results from ModelTrainer
        y_test : np.ndarray
            Ground truth test values
        target_name : str, optional
            Name of target variable
            
        Returns
        -------
        dict
            Complete evaluation report
        """
        # Evaluate all models
        metrics_df = self.evaluate_models(trainer_results, y_test)
        
        # Generate rankings
        ranking = self.get_ranking_comparison(metrics_df)
        
        # Get comparisons
        comparisons = self.get_model_comparison_matrix(metrics_df)
        
        # Best models by different metrics
        best_by_r2 = self.get_best_models_by_metric(metrics_df, 'Test_R²', 5)
        best_by_rmse = self.get_best_models_by_metric(metrics_df, 'Test_RMSE', 5)
        best_by_rpd = self.get_best_models_by_metric(metrics_df, 'RPD', 5)
        
        report = {
            'target_name': target_name,
            'n_models': len(trainer_results),
            'test_samples': len(y_test),
            'metrics_extended': metrics_df,
            'ranking': ranking,
            'comparisons': comparisons,
            'best_by_r2': best_by_r2,
            'best_by_rmse': best_by_rmse,
            'best_by_rpd': best_by_rpd,
            'y_test_stats': {
                'mean': np.mean(y_test),
                'std': np.std(y_test),
                'min': np.min(y_test),
                'max': np.max(y_test)
            }
        }
        
        logger.info("Evaluation report generated successfully")
        return report
    
    
    def export_report_to_csv(self, report: Dict, output_dir: str) -> None:
        """
        Export evaluation report to CSV files.
        
        Parameters
        ----------
        report : dict
            Evaluation report dictionary
        output_dir : str
            Directory to save CSV files
        """
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        report['metrics_extended'].to_csv(
            output_path / 'metrics_extended.csv', index=False
        )
        
        # Save rankings
        report['ranking'].to_csv(
            output_path / 'ranking.csv', index=False
        )
        
        # Save best models
        report['best_by_r2'].to_csv(
            output_path / 'best_by_r2.csv', index=False
        )
        report['best_by_rmse'].to_csv(
            output_path / 'best_by_rmse.csv', index=False
        )
        report['best_by_rpd'].to_csv(
            output_path / 'best_by_rpd.csv', index=False
        )
        
        logger.info(f"Report exported to {output_dir}")
