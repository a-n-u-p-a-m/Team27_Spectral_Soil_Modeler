"""
Context Builder Module
======================
Builds comprehensive context for AI models by aggregating data analytics,
training results, and other relevant information.

Features:
- Data context with full analytics summary
- Training results context with all metrics
- Comparison context for model analysis
- Preprocessing technique context
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build comprehensive context for AI models."""
    
    @staticmethod
    def _extract_paradigm_info(results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract training paradigm information (Standard, Tuned, or Both).
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Training results dataframe
            
        Returns
        -------
        Dict[str, Any]
            Paradigm information with details
        """
        try:
            paradigm_col = None
            
            # Check for paradigm column (case-insensitive)
            for col in results_df.columns:
                if 'paradigm' in col.lower() or 'type' in col.lower():
                    if 'standard' in results_df[col].values or 'tuned' in results_df[col].values:
                        paradigm_col = col
                        break
            
            if paradigm_col is None:
                # Check if there's a hyperparameter column or tuning indicator
                has_tuned = any('tuned' in str(col).lower() or 'param' in str(col).lower() 
                               for col in results_df.columns)
                if has_tuned:
                    return {'paradigm': 'TUNED', 'tuned_details': [], 'both_details': []}
                else:
                    return {'paradigm': 'STANDARD', 'tuned_details': [], 'both_details': []}
            
            # Determine paradigm type
            unique_paradigms = results_df[paradigm_col].unique()
            
            if len(unique_paradigms) == 1:
                paradigm_type = str(unique_paradigms[0]).upper()
                if paradigm_type == 'STANDARD':
                    return {'paradigm': 'STANDARD', 'tuned_details': [], 'both_details': []}
                elif paradigm_type == 'TUNED':
                    tuned_details = ContextBuilder._get_tuned_paradigm_details(results_df, paradigm_col)
                    return {'paradigm': 'TUNED', 'tuned_details': tuned_details, 'both_details': []}
            else:
                # Both paradigms present
                both_details = ContextBuilder._get_both_paradigm_details(results_df, paradigm_col)
                return {'paradigm': 'BOTH', 'tuned_details': [], 'both_details': both_details}
        
        except Exception as e:
            logger.debug(f"Error extracting paradigm info: {e}")
            return {'paradigm': 'STANDARD', 'tuned_details': [], 'both_details': []}
    
    
    @staticmethod
    def _get_tuned_paradigm_details(results_df: pd.DataFrame, paradigm_col: str) -> List[str]:
        """
        Get details about tuned paradigm with hyperparameter variations.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Training results
        paradigm_col : str
            Paradigm column name
            
        Returns
        -------
        List[str]
            Formatted details about tuning
        """
        details = ["TUNED PARADIGM DETAILS:"]
        
        try:
            # Find hyperparameter columns
            param_cols = [col for col in results_df.columns 
                         if 'param' in col.lower() or 'hp' in col.lower() or 'hyperparameter' in col.lower()]
            
            if param_cols:
                details.append(f"  • Hyperparameter Tuning Performed: Yes")
                details.append(f"  • Hyperparameter Columns: {', '.join(param_cols)}")
                details.append("")
                details.append("  Hyperparameter Variations Tested:")
                
                for param_col in param_cols:
                    unique_values = results_df[param_col].dropna().unique()
                    if len(unique_values) > 0:
                        details.append(f"    - {param_col}: {len(unique_values)} variations")
                        if len(unique_values) <= 5:
                            details.append(f"      Values: {', '.join(str(v) for v in unique_values)}")
                
                # Find best tuned model and its hyperparameters
                best_tuned_idx = results_df['Test_R²'].idxmax()
                best_row = results_df.loc[best_tuned_idx]
                
                details.append("")
                details.append("  Best Tuned Configuration:")
                details.append(f"    - Model: {best_row['Model']}")
                details.append(f"    - Technique: {best_row['Technique']}")
                details.append(f"    - R² Score: {best_row['Test_R²']:.6f}")
                
                # Add key metrics for best tuned model
                if 'Test_RMSE' in results_df.columns and pd.notna(best_row['Test_RMSE']):
                    details.append(f"    - Test RMSE: {best_row['Test_RMSE']:.6f}")
                if 'Test_MAE' in results_df.columns and pd.notna(best_row['Test_MAE']):
                    details.append(f"    - Test MAE: {best_row['Test_MAE']:.6f}")
                if 'RPD' in results_df.columns and pd.notna(best_row['RPD']):
                    details.append(f"    - RPD: {best_row['RPD']:.6f}")
                
                for param_col in param_cols:
                    if pd.notna(best_row[param_col]):
                        details.append(f"    - {param_col}: {best_row[param_col]}")
            else:
                # No explicit param columns but marked as tuned
                details.append(f"  • Hyperparameter Tuning Performed: Yes (implicit)")
                details.append(f"  • Total Tuning Variations: {len(results_df)}")
                
                # Group by model to show variations per model
                if 'Model' in results_df.columns:
                    details.append("")
                    details.append("  Tuning Variations per Model:")
                    for model in results_df['Model'].unique():
                        model_data = results_df[results_df['Model'] == model]
                        best_r2 = model_data['Test_R²'].max()
                        details.append(f"    - {model}: {len(model_data)} variations, Best R²={best_r2:.6f}")
            
            details.append("")
            
        except Exception as e:
            logger.debug(f"Error getting tuned paradigm details: {e}")
        
        return details
    
    
    @staticmethod
    def _get_both_paradigm_details(results_df: pd.DataFrame, paradigm_col: str) -> List[str]:
        """
        Get comparison details between Standard and Tuned paradigms.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Training results
        paradigm_col : str
            Paradigm column name
            
        Returns
        -------
        List[str]
            Formatted comparison details
        """
        details = [""]
        
        try:
            standard_df = results_df[results_df[paradigm_col].str.upper() == 'STANDARD']
            tuned_df = results_df[results_df[paradigm_col].str.upper() == 'TUNED']
            
            standard_best_idx = standard_df['Test_R²'].idxmax()
            tuned_best_idx = tuned_df['Test_R²'].idxmax()
            standard_best_row = standard_df.loc[standard_best_idx]
            tuned_best_row = tuned_df.loc[tuned_best_idx]
            
            details.extend([
                "STANDARD PARADIGM:",
                f"  • Models Trained: {len(standard_df)}",
                f"  • Best R²: {standard_df['Test_R²'].max():.6f}",
                f"  • Best Model: {standard_best_row['Model']} ({standard_best_row['Technique']})",
            ])
            
            # Add metrics for standard best model
            if 'Test_RMSE' in standard_df.columns and pd.notna(standard_best_row['Test_RMSE']):
                details.append(f"  • Best RMSE: {standard_best_row['Test_RMSE']:.6f}")
            if 'Test_MAE' in standard_df.columns and pd.notna(standard_best_row['Test_MAE']):
                details.append(f"  • Best MAE: {standard_best_row['Test_MAE']:.6f}")
            if 'RPD' in standard_df.columns and pd.notna(standard_best_row['RPD']):
                details.append(f"  • Best RPD: {standard_best_row['RPD']:.6f}")
            
            details.extend([
                f"  • Mean R²: {standard_df['Test_R²'].mean():.6f}",
                "",
                "TUNED PARADIGM:",
                f"  • Models Trained: {len(tuned_df)}",
                f"  • Best R²: {tuned_df['Test_R²'].max():.6f}",
                f"  • Best Model: {tuned_best_row['Model']} ({tuned_best_row['Technique']})",
            ])
            
            # Add metrics for tuned best model
            if 'Test_RMSE' in tuned_df.columns and pd.notna(tuned_best_row['Test_RMSE']):
                details.append(f"  • Best RMSE: {tuned_best_row['Test_RMSE']:.6f}")
            if 'Test_MAE' in tuned_df.columns and pd.notna(tuned_best_row['Test_MAE']):
                details.append(f"  • Best MAE: {tuned_best_row['Test_MAE']:.6f}")
            if 'RPD' in tuned_df.columns and pd.notna(tuned_best_row['RPD']):
                details.append(f"  • Best RPD: {tuned_best_row['RPD']:.6f}")
            
            details.extend([
                f"  • Mean R²: {tuned_df['Test_R²'].mean():.6f}",
                "",
                "COMPARISON:",
            ])
            
            # Performance improvement
            standard_best = standard_df['Test_R²'].max()
            tuned_best = tuned_df['Test_R²'].max()
            improvement = ((tuned_best - standard_best) / standard_best * 100) if standard_best > 0 else 0
            
            details.extend([
                f"  • Best Tuned vs Best Standard: {improvement:+.2f}% improvement",
                f"  • Mean Tuned vs Mean Standard: {((tuned_df['Test_R²'].mean() - standard_df['Test_R²'].mean()) / standard_df['Test_R²'].mean() * 100):+.2f}% improvement",
            ])
            
            # Find hyperparameter benefits
            param_cols = [col for col in tuned_df.columns 
                         if 'param' in col.lower() or 'hp' in col.lower()]
            
            if param_cols:
                details.append("")
                details.append("  Hyperparameter Tuning Impact:")
                
                for param_col in param_cols:
                    if pd.notna(tuned_df[param_col]).any():
                        unique_vals = tuned_df[param_col].dropna().unique()
                        if len(unique_vals) > 1:
                            details.append(f"    - {param_col}: {len(unique_vals)} values tested")
            
            details.append("")
            
        except Exception as e:
            logger.debug(f"Error getting both paradigm details: {e}")
        
        return details
    
    @staticmethod
    def build_data_context(raw_data: pd.DataFrame, target_col: str,
                          data_profiler: Optional[Any] = None,
                          feature_engineering_config: Optional[Dict[str, Any]] = None,
                          feature_engineering_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Build comprehensive data context with analytics.
        
        Parameters
        ----------
        raw_data : pd.DataFrame
            Raw dataset
        target_col : str
            Target column name
        data_profiler : Optional[Any]
            DataProfiler instance for additional analytics
        feature_engineering_config : Optional[Dict[str, Any]]
            Feature engineering configuration used
        feature_engineering_data : Optional[Dict[str, Any]]
            Calculated feature engineering values and statistics
            
        Returns
        -------
        str
            Formatted context string for AI
        """
        try:
            from interface import DataProfiler
            
            numeric_cols = raw_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Basic statistics
            basic_stats = DataProfiler.get_basic_statistics(raw_data)
            
            context_parts = [
                "=" * 80,
                "DATA ANALYSIS CONTEXT",
                "=" * 80,
                "",
                "DATASET OVERVIEW:",
                f"  • Total Samples: {basic_stats['samples']:,}",
                f"  • Total Features: {basic_stats['features']}",
                f"  • Numeric Features: {basic_stats['numeric_features']}",
                f"  • Memory Usage: {basic_stats['memory_mb']:.2f} MB",
                f"  • Missing Values: {basic_stats['missing_values']} ({basic_stats['missing_percent']:.2f}%)",
                "",
                "TARGET VARIABLE STATISTICS:",
                f"  • Target: {target_col}",
                f"  • Mean: {raw_data[target_col].mean():.6f}",
                f"  • Median: {raw_data[target_col].median():.6f}",
                f"  • Std Dev: {raw_data[target_col].std():.6f}",
                f"  • Min: {raw_data[target_col].min():.6f}",
                f"  • Max: {raw_data[target_col].max():.6f}",
                f"  • Range: {raw_data[target_col].max() - raw_data[target_col].min():.6f}",
                "",
            ]
            
            # Target-feature correlations
            try:
                if len(numeric_cols) > 1:
                    correlations = raw_data[numeric_cols].corrwith(raw_data[target_col]).sort_values(ascending=False)
                    top_corr = correlations.head(10)
                    
                    context_parts.append("TOP CORRELATED FEATURES WITH TARGET:")
                    for idx, (feature, corr_value) in enumerate(top_corr.items(), 1):
                        if feature != target_col:
                            context_parts.append(f"  {idx}. {feature}: {corr_value:.4f}")
                    context_parts.append("")
            except Exception as e:
                logger.debug(f"Could not compute correlations: {e}")
            
            # Distribution analysis
            try:
                target_data = raw_data[target_col].dropna()
                from scipy.stats import skew, kurtosis
                
                context_parts.extend([
                    "TARGET DISTRIBUTION:",
                    f"  • Skewness: {skew(target_data):.4f}",
                    f"  • Kurtosis: {kurtosis(target_data):.4f}",
                    f"  • Q1 (25%): {target_data.quantile(0.25):.6f}",
                    f"  • Q3 (75%): {target_data.quantile(0.75):.6f}",
                    f"  • IQR: {target_data.quantile(0.75) - target_data.quantile(0.25):.6f}",
                    "",
                ])
            except Exception as e:
                logger.debug(f"Could not compute distribution: {e}")
            
            # Feature statistics
            try:
                if numeric_cols:
                    context_parts.append("FEATURE STATISTICS (Numeric Features):")
                    context_parts.append(f"  • Count: {len(numeric_cols)}")
                    context_parts.append(f"  • Mean of means: {raw_data[numeric_cols].mean().mean():.6f}")
                    context_parts.append(f"  • Mean of stds: {raw_data[numeric_cols].std().mean():.6f}")
                    context_parts.append(f"  • Max feature correlation with target: {raw_data[numeric_cols].corrwith(raw_data[target_col]).abs().max():.4f}")
                    context_parts.append("")
            except Exception as e:
                logger.debug(f"Could not compute feature stats: {e}")
            
            # Data quality
            try:
                completeness = (1 - (raw_data.isnull().sum().sum() / (raw_data.shape[0] * raw_data.shape[1]))) * 100
                duplicate_rows = raw_data.duplicated().sum()
                
                context_parts.extend([
                    "DATA QUALITY ASSESSMENT:",
                    f"  • Completeness: {completeness:.2f}%",
                    f"  • Duplicate rows: {duplicate_rows}",
                    f"  • Duplicate percentage: {(duplicate_rows / len(raw_data) * 100):.2f}%",
                    "",
                ])
            except Exception as e:
                logger.debug(f"Could not compute data quality: {e}")
            
            # Outlier detection
            try:
                numeric_data = raw_data[numeric_cols]
                Q1 = numeric_data.quantile(0.25)
                Q3 = numeric_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).sum()
                outlier_percent = (outliers.sum() / (len(numeric_data) * len(numeric_cols)) * 100)
                
                context_parts.extend([
                    "OUTLIER ANALYSIS (IQR Method):",
                    f"  • Total outliers detected: {outliers.sum()}",
                    f"  • Percentage of data: {outlier_percent:.2f}%",
                    f"  • Features with outliers: {(outliers > 0).sum()}",
                    "",
                ])
            except Exception as e:
                logger.debug(f"Could not detect outliers: {e}")
            
            # Add feature engineering context if provided
            if feature_engineering_config and any([
                feature_engineering_config.get('derivatives'),
                feature_engineering_config.get('statistical'),
                feature_engineering_config.get('polynomial'),
                feature_engineering_config.get('spectral_indices'),
                feature_engineering_config.get('pca'),
                feature_engineering_config.get('wavelet')
            ]):
                context_parts.extend([
                    "",
                    "FEATURE ENGINEERING APPLIED:",
                    "",
                ])
                
                fe_techniques = []
                if feature_engineering_config.get('derivatives'):
                    fe_techniques.append("  • Spectral Derivatives (1st-order): Captures rate of change across wavelengths")
                if feature_engineering_config.get('statistical'):
                    window = feature_engineering_config.get('stat_window', 10)
                    fe_techniques.append(f"  • Statistical Features (window={window}): Mean, std, variance, skewness, kurtosis")
                if feature_engineering_config.get('polynomial'):
                    fe_techniques.append("  • Polynomial Features: Interaction terms between spectral bands")
                if feature_engineering_config.get('spectral_indices'):
                    fe_techniques.append("  • Spectral Indices: Custom aggregate metrics (mean, std, slope, curvature)")
                if feature_engineering_config.get('pca'):
                    fe_techniques.append("  • PCA Features: Principal Component Analysis for dimensionality reduction (5 components)")
                if feature_engineering_config.get('wavelet'):
                    fe_techniques.append("  • Wavelet Features: Discrete wavelet transform for multi-scale feature analysis")
                
                context_parts.extend(fe_techniques)
                
                # Add calculated feature engineering statistics if available
                if feature_engineering_data:
                    context_parts.extend([
                        "",
                        "FEATURE ENGINEERING VALUES AND STATISTICS:",
                        ""
                    ])
                    
                    # Derivatives statistics
                    if 'derivatives' in feature_engineering_data:
                        deriv_stats = feature_engineering_data['derivatives']
                        context_parts.append(f"  Spectral Derivatives:")
                        context_parts.append(f"    - Output shape: {deriv_stats.get('shape', 'N/A')}")
                        context_parts.append(f"    - Mean value: {deriv_stats.get('mean', 'N/A')}")
                        context_parts.append(f"    - Std dev: {deriv_stats.get('std', 'N/A')}")
                        context_parts.append(f"    - Range: [{deriv_stats.get('min', 'N/A')}, {deriv_stats.get('max', 'N/A')}]")
                    
                    # Statistical features statistics
                    if 'statistical' in feature_engineering_data:
                        stat_stats = feature_engineering_data['statistical']
                        context_parts.append(f"  Statistical Features:")
                        context_parts.append(f"    - Output shape: {stat_stats.get('shape', 'N/A')}")
                        context_parts.append(f"    - Features: Mean, Std, Variance, Skewness, Kurtosis")
                        context_parts.append(f"    - Window statistics (mean across windows):")
                        context_parts.append(f"      • Mean: {stat_stats.get('mean', 'N/A')}")
                        context_parts.append(f"      • Std dev: {stat_stats.get('std', 'N/A')}")
                    
                    # Polynomial features statistics
                    if 'polynomial' in feature_engineering_data:
                        poly_stats = feature_engineering_data['polynomial']
                        context_parts.append(f"  Polynomial Features (Degree 2):")
                        context_parts.append(f"    - Output shape: {poly_stats.get('shape', 'N/A')}")
                        context_parts.append(f"    - Includes: Original + squares + all pairwise interactions")
                        context_parts.append(f"    - Mean value: {poly_stats.get('mean', 'N/A')}")
                        context_parts.append(f"    - Range: [{poly_stats.get('min', 'N/A')}, {poly_stats.get('max', 'N/A')}]")
                    
                    # Spectral indices statistics
                    if 'spectral_indices' in feature_engineering_data:
                        indices_stats = feature_engineering_data['spectral_indices']
                        context_parts.append(f"  Spectral Indices:")
                        context_parts.append(f"    - Mean Reflectance: {indices_stats.get('mean_reflectance', 'N/A')}")
                        context_parts.append(f"    - Spectral Std Dev: {indices_stats.get('std_reflectance', 'N/A')}")
                        context_parts.append(f"    - Reflectance Slope: {indices_stats.get('slope', 'N/A')}")
                        context_parts.append(f"    - Spectral Curvature: {indices_stats.get('curvature', 'N/A')}")
                    
                    # PCA features statistics
                    if 'pca' in feature_engineering_data:
                        pca_stats = feature_engineering_data['pca']
                        context_parts.append(f"  PCA Features:")
                        context_parts.append(f"    - Output shape: {pca_stats.get('shape', 'N/A')}")
                        context_parts.append(f"    - Explained variance ratio: {pca_stats.get('explained_variance', 'N/A')}")
                        context_parts.append(f"    - Total variance explained: {pca_stats.get('total_variance', 'N/A')}")
                    
                    # Wavelet features statistics
                    if 'wavelet' in feature_engineering_data:
                        wavelet_stats = feature_engineering_data['wavelet']
                        context_parts.append(f"  Wavelet Features:")
                        context_parts.append(f"    - Output shape: {wavelet_stats.get('shape', 'N/A')}")
                        context_parts.append(f"    - Wavelet type: {wavelet_stats.get('wavelet_type', 'N/A')}")
                        context_parts.append(f"    - Mean approximation: {wavelet_stats.get('mean_approx', 'N/A')}")
                    
                    context_parts.append("")
                
                context_parts.extend([
                    "Impact: Feature engineering increased feature space and created derived features",
                    "that capture different aspects of the spectral data."
                ])
            
            context_parts.append("=" * 80)
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building data context: {e}")
            numeric_cols = raw_data.select_dtypes(include=[np.number]).columns.tolist()
            return f"Data context: {len(raw_data)} samples, {len(numeric_cols)} features, target: {target_col}"
    
    
    @staticmethod
    def build_training_context(results_df: pd.DataFrame, 
                              raw_data: pd.DataFrame,
                              target_col: str,
                              paradigm: Optional[str] = None,
                              data_analytics_context: Optional[str] = None,
                              feature_engineering_config: Optional[Dict[str, Any]] = None,
                              feature_engineering_data: Optional[Dict[str, Any]] = None,
                              feature_importance_data: Optional[Dict[str, Any]] = None,
                              cv_strategy: Optional[str] = None,
                              search_method: Optional[str] = None,
                              n_iter: Optional[int] = None,
                              cv_folds: Optional[int] = None) -> str:
        """
        Build comprehensive training results context.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Training results
        raw_data : pd.DataFrame
            Original dataset
        target_col : str
            Target column name
        paradigm : Optional[str]
            Training paradigm ('Standard', 'Tuned', 'Both'). If None, will attempt to infer from data.
        data_analytics_context : Optional[str]
            Data analytics context from earlier analysis
        feature_engineering_config : Optional[Dict[str, Any]]
            Feature engineering configuration used
        feature_engineering_data : Optional[Dict[str, Any]]
            Calculated feature engineering values and statistics
        feature_importance_data : Optional[Dict[str, Any]]
            Feature importance data from all models
        cv_strategy : Optional[str]
            Cross-validation strategy used ('k-fold' or 'leave-one-out')
        search_method : Optional[str]
            Hyperparameter search method ('grid' or 'random')
        n_iter : Optional[int]
            Number of iterations for RandomizedSearch
        cv_folds : Optional[int]
            Number of CV folds for K-Fold strategy
            
        Returns
        -------
        str
            Formatted context string for AI
        """
        try:
            context_parts = []
            
            # Add data analytics context if provided
            if data_analytics_context:
                context_parts.append(data_analytics_context)
                context_parts.append("")
            
            # Add feature engineering context if provided
            if feature_engineering_config and any([
                feature_engineering_config.get('derivatives'),
                feature_engineering_config.get('statistical'),
                feature_engineering_config.get('polynomial'),
                feature_engineering_config.get('spectral_indices'),
                feature_engineering_config.get('pca'),
                feature_engineering_config.get('wavelet')
            ]):
                context_parts.extend([
                    "FEATURE ENGINEERING APPLIED:",
                    "",
                ])
                
                fe_techniques = []
                if feature_engineering_config.get('derivatives'):
                    fe_techniques.append("  • Spectral Derivatives (1st-order): Captures rate of change across wavelengths")
                if feature_engineering_config.get('statistical'):
                    window = feature_engineering_config.get('stat_window', 10)
                    fe_techniques.append(f"  • Statistical Features (window={window}): Mean, std, variance, skewness, kurtosis")
                if feature_engineering_config.get('polynomial'):
                    fe_techniques.append("  • Polynomial Features: Interaction terms between spectral bands")
                if feature_engineering_config.get('spectral_indices'):
                    fe_techniques.append("  • Spectral Indices: Custom aggregate metrics (mean, std, slope, curvature)")
                if feature_engineering_config.get('pca'):
                    fe_techniques.append("  • PCA Features: Principal Component Analysis for dimensionality reduction (5 components)")
                if feature_engineering_config.get('wavelet'):
                    fe_techniques.append("  • Wavelet Features: Discrete wavelet transform for multi-scale feature analysis")
                
                context_parts.extend(fe_techniques)
                
                # Add calculated feature engineering statistics if available
                if feature_engineering_data:
                    context_parts.extend([
                        "",
                        "FEATURE ENGINEERING VALUES AND STATISTICS:",
                        ""
                    ])
                    
                    # Derivatives statistics
                    if 'derivatives' in feature_engineering_data:
                        deriv_stats = feature_engineering_data['derivatives']
                        context_parts.append(f"  Spectral Derivatives:")
                        context_parts.append(f"    - Output shape: {deriv_stats.get('shape', 'N/A')}")
                        context_parts.append(f"    - Mean value: {deriv_stats.get('mean', 'N/A')}")
                        context_parts.append(f"    - Std dev: {deriv_stats.get('std', 'N/A')}")
                        context_parts.append(f"    - Range: [{deriv_stats.get('min', 'N/A')}, {deriv_stats.get('max', 'N/A')}]")
                    
                    # Statistical features statistics
                    if 'statistical' in feature_engineering_data:
                        stat_stats = feature_engineering_data['statistical']
                        context_parts.append(f"  Statistical Features:")
                        context_parts.append(f"    - Output shape: {stat_stats.get('shape', 'N/A')}")
                        context_parts.append(f"    - Features: Mean, Std, Variance, Skewness, Kurtosis")
                        context_parts.append(f"    - Window statistics (mean across windows):")
                        context_parts.append(f"      • Mean: {stat_stats.get('mean', 'N/A')}")
                        context_parts.append(f"      • Std dev: {stat_stats.get('std', 'N/A')}")
                    
                    # Polynomial features statistics
                    if 'polynomial' in feature_engineering_data:
                        poly_stats = feature_engineering_data['polynomial']
                        context_parts.append(f"  Polynomial Features (Degree 2):")
                        context_parts.append(f"    - Output shape: {poly_stats.get('shape', 'N/A')}")
                        context_parts.append(f"    - Includes: Original + squares + all pairwise interactions")
                        context_parts.append(f"    - Mean value: {poly_stats.get('mean', 'N/A')}")
                        context_parts.append(f"    - Range: [{poly_stats.get('min', 'N/A')}, {poly_stats.get('max', 'N/A')}]")
                    
                    # Spectral indices statistics
                    if 'spectral_indices' in feature_engineering_data:
                        indices_stats = feature_engineering_data['spectral_indices']
                        context_parts.append(f"  Spectral Indices:")
                        context_parts.append(f"    - Mean Reflectance: {indices_stats.get('mean_reflectance', 'N/A')}")
                        context_parts.append(f"    - Spectral Std Dev: {indices_stats.get('std_reflectance', 'N/A')}")
                        context_parts.append(f"    - Reflectance Slope: {indices_stats.get('slope', 'N/A')}")
                        context_parts.append(f"    - Spectral Curvature: {indices_stats.get('curvature', 'N/A')}")
                    
                    # PCA features statistics
                    if 'pca' in feature_engineering_data:
                        pca_stats = feature_engineering_data['pca']
                        context_parts.append(f"  PCA Features:")
                        context_parts.append(f"    - Output shape: {pca_stats.get('shape', 'N/A')}")
                        context_parts.append(f"    - Explained variance ratio: {pca_stats.get('explained_variance', 'N/A')}")
                        context_parts.append(f"    - Total variance explained: {pca_stats.get('total_variance', 'N/A')}")
                    
                    # Wavelet features statistics
                    if 'wavelet' in feature_engineering_data:
                        wavelet_stats = feature_engineering_data['wavelet']
                        context_parts.append(f"  Wavelet Features:")
                        context_parts.append(f"    - Output shape: {wavelet_stats.get('shape', 'N/A')}")
                        context_parts.append(f"    - Wavelet type: {wavelet_stats.get('wavelet_type', 'N/A')}")
                        context_parts.append(f"    - Mean approximation: {wavelet_stats.get('mean_approx', 'N/A')}")
                    
                    context_parts.append("")
                
                context_parts.extend([
                    "Impact: Feature engineering increased feature dimensionality and created derived features",
                    "that capture different aspects of the spectral data."
                ])
                context_parts.append("")
            
            # Determine training paradigm (use provided paradigm or infer from data)
            if paradigm:
                paradigm_upper = paradigm.upper()
                paradigm_info = {'paradigm': paradigm_upper, 'tuned_details': [], 'both_details': []}
            else:
                paradigm_info = ContextBuilder._extract_paradigm_info(results_df)
            
            context_parts.extend([
                "=" * 80,
                "TRAINING RESULTS CONTEXT",
                "=" * 80,
                "",
                "OVERALL SUMMARY:",
                f"  • Total Models Trained: {len(results_df)}",
                f"  • Training Paradigm: {paradigm_info['paradigm']}",
                f"  • Best R² Score: {results_df['Test_R²'].max():.6f}",
                f"  • Mean R² Score: {results_df['Test_R²'].mean():.6f}",
                f"  • Median R² Score: {results_df['Test_R²'].median():.6f}",
                f"  • R² Std Dev: {results_df['Test_R²'].std():.6f}",
                f"  • Best Model: {results_df.loc[results_df['Test_R²'].idxmax(), 'Model']}",
                f"  • Best Technique: {results_df.loc[results_df['Test_R²'].idxmax(), 'Technique']}",
                "",
                "TRAINING CONFIGURATION:",
                f"  • Cross-Validation Strategy: {cv_strategy.upper() if cv_strategy else 'K-Fold (default)'}",
                "",
            ])
            
            # Add CV strategy details
            if cv_strategy and cv_strategy.lower() == 'leave-one-out':
                context_parts.extend([
                    "  Cross-Validation Details (Leave-One-Out):",
                    f"    - Each sample left out exactly once for validation",
                    f"    - Number of folds: {len(raw_data)} (one per sample)",
                    f"    - Provides: Unbiased performance estimate on small datasets",
                    f"    - Trade-off: Computationally intensive but most robust CV method",
                    "",
                ])
                
                # Add LOO CV metrics if available
                if any(col for col in results_df.columns if 'LOO_CV' in col):
                    context_parts.extend([
                        "  Leave-One-Out Cross-Validation Metrics (Test Set):",
                        ""
                    ])
                    
                    # Check for LOO CV metrics in results
                    loo_cv_cols = [col for col in results_df.columns if 'LOO_CV_Test' in col]
                    if loo_cv_cols:
                            context_parts.append("  Available LOO CV Metrics:")
                            for col in sorted(loo_cv_cols):
                                # Only process columns with scalar numeric values
                                vals = results_df[col]
                                if pd.notna(vals).any() and vals.apply(lambda x: isinstance(x, (int, float, np.integer, np.floating))).all():
                                    mean_val = vals.mean()
                                    std_val = vals.std()
                                    max_val = vals.max()
                                    context_parts.append(f"    • {col}:")
                                    context_parts.append(f"      - Mean: {mean_val:.6f}")
                                    context_parts.append(f"      - Std Dev: {std_val:.6f}")
                                    context_parts.append(f"      - Best: {max_val:.6f}")
                            context_parts.append("")
                            context_parts.extend([
                                "  Interpretation: LOO CV metrics provide unbiased estimates of generalization",
                                "  performance since each prediction is made on a held-out sample.",
                                ""
                            ])     
            else:
                context_parts.extend([
                    "  Cross-Validation Details (K-Fold):",
                    f"    - Folds: {cv_folds if cv_folds else 5} (default)",
                    f"    - Data split: {100//(cv_folds if cv_folds else 5)}% test per fold",
                    f"    - Trade-off: Faster than LOO CV, good balance of bias and variance",
                    "",
                ])
            
            # Add hyperparameter tuning configuration
            if paradigm_info['paradigm'] == 'TUNED' or paradigm_info['paradigm'] == 'BOTH':
                context_parts.extend([
                    "HYPERPARAMETER TUNING CONFIGURATION:",
                    f"  • Search Method: {'GridSearch (exhaustive)' if search_method and search_method.lower() == 'grid' else 'RandomizedSearch (sampled)'}",
                ])
                
                if search_method and search_method.lower() == 'grid':
                    context_parts.extend([
                        "    - Grid Search: Tests all parameter combinations in large parameter grid",
                        "    - Advantage: Comprehensive exploration of hyperparameter space",
                        "    - Trade-off: Higher computational cost but more thorough",
                    ])
                else:
                    context_parts.extend([
                        f"    - Randomized Search: Tests {n_iter if n_iter else 20} random parameter combinations from small grid",
                        "    - Advantage: Faster than grid search, good for quick exploration",
                        "    - Trade-off: May miss optimal parameters but computationally efficient",
                    ])
                
                context_parts.append("")
            
            context_parts.append("")
            
            # Add paradigm-specific details
            if paradigm_info['paradigm'] == 'TUNED':
                context_parts.extend(paradigm_info['tuned_details'])
            elif paradigm_info['paradigm'] == 'BOTH':
                context_parts.append("TRAINING PARADIGM: Both Standard and Tuned models")
                context_parts.extend(paradigm_info['both_details'])
            
            # Add BEST MODEL HYPERPARAMETERS SECTION
            context_parts.append("")
            context_parts.append("=" * 80)
            context_parts.append("BEST MODEL HYPERPARAMETERS")
            context_parts.append("=" * 80)
            best_idx = results_df['Test_R²'].idxmax()
            best_row = results_df.loc[best_idx]
            best_hyperparams = {}
            
            # Try to get hyperparameters from Hyperparameters column first (most reliable)
            if 'Hyperparameters' in results_df.columns and pd.notna(best_row['Hyperparameters']):
                hyperparams = best_row['Hyperparameters']
                if isinstance(hyperparams, dict) and hyperparams:  # Check if dict is not empty
                    best_hyperparams = hyperparams
            
            # Fallback to Model_Object if Hyperparameters column is empty or not a dict
            if not best_hyperparams and 'Model_Object' in results_df.columns:
                try:
                    from model_analyzer import ParameterInspector
                    best_model_obj = best_row['Model_Object']
                    best_hyperparams = ParameterInspector.get_hyperparameters(best_model_obj)
                except Exception as e:
                    logger.debug(f"Could not extract hyperparameters from Model_Object: {e}")
            
            context_parts.append(f"Model: {best_row['Model']} | Technique: {best_row['Technique']} | R²: {best_row['Test_R²']:.6f}")
            
            # Add key metrics for best model
            metrics_info = []
            if 'Test_RMSE' in results_df.columns and pd.notna(best_row['Test_RMSE']):
                metrics_info.append(f"RMSE: {best_row['Test_RMSE']:.6f}")
            if 'Test_MAE' in results_df.columns and pd.notna(best_row['Test_MAE']):
                metrics_info.append(f"MAE: {best_row['Test_MAE']:.6f}")
            if 'RPD' in results_df.columns and pd.notna(best_row['RPD']):
                metrics_info.append(f"RPD: {best_row['RPD']:.6f}")
            
            if metrics_info:
                context_parts.append("Key Metrics:")
                for metric in metrics_info:
                    context_parts.append(f"  • {metric}")
            
            context_parts.append("")
            
            if best_hyperparams:
                context_parts.append("Hyperparameters:")
                for k, v in best_hyperparams.items():
                    context_parts.append(f"  • {k}: {v}")
            else:
                context_parts.append("Hyperparameters not available")
            context_parts.append("")
            
            context_parts.extend([
                "ALL MODEL RESULTS (Ranked by Test R² Score):",
                "",
            ])
            
            # Add all models sorted by R²
            for idx, row in results_df.nlargest(len(results_df), 'Test_R²').iterrows():
                model_info = [
                    f"  {idx + 1}. {row['Model']} - {row['Technique']}",
                ]
                
                # Add Test metrics
                model_info.append(f"     TEST SET METRICS:")
                if 'Test_R²' in results_df.columns:
                    model_info.append(f"       • R² Score: {row['Test_R²']:.6f}")
                if 'Test_RMSE' in results_df.columns:
                    model_info.append(f"       • RMSE: {row['Test_RMSE']:.6f}")
                if 'Test_MAE' in results_df.columns:
                    model_info.append(f"       • MAE: {row['Test_MAE']:.6f}")
                if 'RPD' in results_df.columns and pd.notna(row['RPD']):
                    model_info.append(f"       • RPD: {row['RPD']:.6f}")
                
                # Add LOO CV metrics if available
                loo_cv_metrics_found = False
                if 'LOO_CV_Test_R²' in results_df.columns and pd.notna(row['LOO_CV_Test_R²']):
                    if not loo_cv_metrics_found:
                        model_info.append(f"     LEAVE-ONE-OUT CV TEST METRICS:")
                        loo_cv_metrics_found = True
                    model_info.append(f"       • LOO CV R² Score: {row['LOO_CV_Test_R²']:.6f}")
                
                if 'LOO_CV_Test_RMSE' in results_df.columns and pd.notna(row['LOO_CV_Test_RMSE']):
                    if not loo_cv_metrics_found:
                        model_info.append(f"     LEAVE-ONE-OUT CV TEST METRICS:")
                        loo_cv_metrics_found = True
                    model_info.append(f"       • LOO CV RMSE: {row['LOO_CV_Test_RMSE']:.6f}")
                
                if 'LOO_CV_Test_MAE' in results_df.columns and pd.notna(row['LOO_CV_Test_MAE']):
                    if not loo_cv_metrics_found:
                        model_info.append(f"     LEAVE-ONE-OUT CV TEST METRICS:")
                        loo_cv_metrics_found = True
                    model_info.append(f"       • LOO CV MAE: {row['LOO_CV_Test_MAE']:.6f}")
                
                # Add hyperparameters if available
                if 'Hyperparameters' in results_df.columns and pd.notna(row['Hyperparameters']):
                    hyperparams = row['Hyperparameters']
                    if isinstance(hyperparams, dict):
                        # Show top 5 hyperparameters
                        hp_items = list(hyperparams.items())[:5]
                        model_info.append(f"     Hyperparameters: {', '.join(f'{k}={v}' for k, v in hp_items)}")
                
                context_parts.extend(model_info)
            
            context_parts.append("")
            
            # Performance by preprocessing technique
            if 'Technique' in results_df.columns:
                context_parts.append("PERFORMANCE BY PREPROCESSING TECHNIQUE:")
                techniques = results_df['Technique'].unique()
                
                for tech in sorted(techniques):
                    tech_data = results_df[results_df['Technique'] == tech]
                    context_parts.append(f"  • {tech}:")
                    context_parts.append(f"    - Models: {len(tech_data)}")
                    context_parts.append(f"    - Test R² - Best: {tech_data['Test_R²'].max():.6f}, Mean: {tech_data['Test_R²'].mean():.6f}, Std Dev: {tech_data['Test_R²'].std():.6f}")
                    
                    # Add LOO CV metrics if available
                    if 'LOO_CV_Test_R²' in tech_data.columns and tech_data['LOO_CV_Test_R²'].notna().any():
                        loo_data = tech_data['LOO_CV_Test_R²'].dropna()
                        context_parts.append(f"    - LOO CV Test R² - Best: {loo_data.max():.6f}, Mean: {loo_data.mean():.6f}, Std Dev: {loo_data.std():.6f}")
                
                context_parts.append("")
            
            # Performance by algorithm - ALWAYS INCLUDE THIS SECTION
            # This section provides per-model consistency metrics across all techniques
            if 'Model' in results_df.columns:
                context_parts.append("PERFORMANCE BY ALGORITHM (Model Consistency Across Techniques):")
                context_parts.append("Use this section to assess which model is most consistent across different preprocessing techniques.")
                context_parts.append("Lower Std Dev = higher consistency; Higher Std Dev = more variable performance across techniques.")
                context_parts.append("")
                
                models = results_df['Model'].unique()
                
                for model in sorted(models):
                    model_data = results_df[results_df['Model'] == model]
                    context_parts.append(f"  • {model}:")
                    context_parts.append(f"    - Techniques tested: {len(model_data)}")
                    context_parts.append(f"    - Test R² - Best: {model_data['Test_R²'].max():.6f}, Mean: {model_data['Test_R²'].mean():.6f}, Std Dev: {model_data['Test_R²'].std():.6f}")
                    
                    # Add LOO CV metrics if available
                    if 'LOO_CV_Test_R²' in model_data.columns and model_data['LOO_CV_Test_R²'].notna().any():
                        loo_data = model_data['LOO_CV_Test_R²'].dropna()
                        context_parts.append(f"    - LOO CV Test R² - Best: {loo_data.max():.6f}, Mean: {loo_data.mean():.6f}, Std Dev: {loo_data.std():.6f}")
                    
                    # Add per-technique breakdown for this model
                    context_parts.append(f"    - Performance by technique:")
                    for technique in sorted(results_df['Technique'].unique()):
                        tech_model_data = model_data[model_data['Technique'] == technique]
                        if len(tech_model_data) > 0:
                            r2_val = tech_model_data['Test_R²'].values[0]
                            metrics_str = f"      • {technique}: Test R²={r2_val:.6f}"
                            
                            # Add LOO CV if available
                            if 'LOO_CV_Test_R²' in results_df.columns and pd.notna(tech_model_data['LOO_CV_Test_R²'].values[0]):
                                loo_r2 = tech_model_data['LOO_CV_Test_R²'].values[0]
                                metrics_str += f", LOO CV R²={loo_r2:.6f}"
                            
                            context_parts.append(metrics_str)
                
                context_parts.append("")
            
            # Model-Technique combinations - ALL combinations
            context_parts.append("=" * 80)
            context_parts.append("ALL MODEL-TECHNIQUE COMBINATIONS (15 Total)")
            context_parts.append("=" * 80)
            
            # Create a matrix view of all combinations
            all_combinations = results_df[['Model', 'Technique', 'Test_R²', 'Test_RMSE', 'Test_MAE']].sort_values('Test_R²', ascending=False) if 'Test_RMSE' in results_df.columns else results_df[['Model', 'Technique', 'Test_R²']].sort_values('Test_R²', ascending=False)
            
            for rank, (idx, row) in enumerate(all_combinations.iterrows(), 1):
                combination_str = f"  {rank:2d}. {row['Model']:15s} + {row['Technique']:20s} → R²: {row['Test_R²']:.6f}"
                
                # Add additional metrics if available
                if 'Test_RMSE' in all_combinations.columns:
                    combination_str += f" | RMSE: {row['Test_RMSE']:.6f}"
                if 'Test_MAE' in all_combinations.columns:
                    combination_str += f" | MAE: {row['Test_MAE']:.6f}"
                
                context_parts.append(combination_str)
            
            context_parts.append("")
            context_parts.append("=" * 80)
            context_parts.append("COMBINATION ANALYSIS ACROSS ALL PREPROCESSING TECHNIQUES")
            context_parts.append("=" * 80)
            context_parts.append("")
            
            # Detailed breakdown by technique showing all models per technique
            if 'Technique' in results_df.columns:
                techniques = sorted(results_df['Technique'].unique())
                
                for tech in techniques:
                    tech_data = results_df[results_df['Technique'] == tech].sort_values('Test_R²', ascending=False)
                    context_parts.append(f"TECHNIQUE: {tech}")
                    context_parts.append(f"  Total Models Tested: {len(tech_data)}")
                    
                    for model_rank, (m_idx, m_row) in enumerate(tech_data.iterrows(), 1):
                        model_line = f"    {model_rank}. {m_row['Model']:15s}: R²={m_row['Test_R²']:.6f}"
                        
                        if 'Test_RMSE' in tech_data.columns and pd.notna(m_row['Test_RMSE']):
                            model_line += f" | RMSE={m_row['Test_RMSE']:.6f}"
                        if 'Test_MAE' in tech_data.columns and pd.notna(m_row['Test_MAE']):
                            model_line += f" | MAE={m_row['Test_MAE']:.6f}"
                        if 'RPD' in tech_data.columns and pd.notna(m_row['RPD']):
                            model_line += f" | RPD={m_row['RPD']:.6f}"
                        
                        context_parts.append(model_line)
                    
                    context_parts.append("")
            
            # Add feature importance methods explanation
            context_parts.extend([
                "=" * 80,
                "FEATURE IMPORTANCE ANALYSIS METHODS",
                "=" * 80,
                "",
                "Feature Importance Calculation Methods Used:",
                "",
                "Tree-Based Models (GBRT, Cubist):",
                "  • Method: Gini/Entropy impurity-based importance from decision trees",
                "  • Interpretation: Higher values = features more critical for splits",
                "  • Characteristics: Fast computation, biased toward high-cardinality features",
                "",
                "Linear Models (PLSR):",
                "  • Method: Standardized regression coefficient absolute values",
                "  • Interpretation: Shows relative weight of each feature in predictions",
                "  • Characteristics: Directly interpretable, accounts for multicollinearity",
                "",
                "Kernel Methods (KRR, SVR):",
                "  • Method: Permutation importance (feature shuffling impact on error)",
                "  • Interpretation: Decrease in model performance when feature is randomized",
                "  • Characteristics: Model-agnostic, reliable across all model types",
                "",
                "Overall Context:",
                "  • Feature importance helps identify key spectral bands for soil properties",
                "  • Different models may emphasize different features (ensemble approach)",
                "  • Use feature importance to guide dimensionality reduction or feature engineering",
                "",
            ])
            
            # Add feature importance information if provided
            # Handle both single model importance and multiple models importance dict
            if feature_importance_data and isinstance(feature_importance_data, dict) and len(feature_importance_data) > 0:
                try:
                    context_parts.extend([
                        "=" * 80,
                        "FEATURE IMPORTANCE ANALYSIS FOR ALL MODEL-TECHNIQUE COMBINATIONS",
                        "=" * 80,
                        ""
                    ])
                    
                    # Check if it's a dictionary of multiple models or single model
                    # If it has 'top_features' key, it's old format (single model)
                    if 'top_features' in feature_importance_data:
                        context_parts.extend([
                            f"Model: {feature_importance_data.get('model_name', 'Unknown')}",
                            f"Importance Type: {feature_importance_data.get('importance_type', 'unknown')}",
                            f"Total Features: {feature_importance_data.get('n_features', 0)}",
                            "",
                            "Top 10 Most Important Spectral Bands:",
                            ""
                        ])
                        
                        top_features = feature_importance_data.get('top_features', [])
                        if isinstance(top_features, list):
                            for i, feat in enumerate(top_features[:10], 1):
                                if isinstance(feat, dict):
                                    band_idx = feat.get('index', 0)
                                    importance = feat.get('importance', 0)
                                    context_parts.append(f"  {i:2d}. Band {band_idx:3d}: {importance:.6f}")
                    else:
                        # Multiple models format - iterate through all models
                        for combo_key, fi_info in sorted(feature_importance_data.items(), 
                                                         key=lambda x: x[1].get('r2_score', 0) if isinstance(x[1], dict) else 0, reverse=True):
                            if not isinstance(fi_info, dict):
                                continue
                                
                            model_name = fi_info.get('model', 'Unknown')
                            technique = fi_info.get('technique', 'Unknown')
                            r2_score = fi_info.get('r2_score', 0)
                            importance_type = fi_info.get('importance_type', 'unknown')
                            top_features = fi_info.get('top_features', [])
                            n_features = fi_info.get('n_features', 0)
                            
                            context_parts.extend([
                                f"Model: {model_name} | Technique: {technique} | R²: {r2_score:.6f}",
                                f"  Importance Type: {importance_type}",
                                f"  Total Features: {n_features}",
                                f"  Top 10 Most Important Spectral Bands:",
                                ""
                            ])
                            
                            if isinstance(top_features, list):
                                for i, feat in enumerate(top_features[:10], 1):
                                    if isinstance(feat, dict):
                                        band_idx = feat.get('index', 0)
                                        importance = feat.get('importance', 0)
                                        context_parts.append(f"    {i:2d}. Band {band_idx:3d}: {importance:.6f}")
                            
                            context_parts.append("")
                except Exception as fe:
                    logger.warning(f"Error processing feature importance data: {fe}")
                    # Continue with rest of context building
            
            context_parts.append("=" * 80)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building training context: {e}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Training context: {len(results_df)} models trained"
    
    
    @staticmethod
    def build_prediction_context(model_info: Dict[str, Any],
                                 predictions: pd.Series,
                                 actuals: Optional[pd.Series] = None) -> str:
        """
        Build context for prediction analysis.
        
        Parameters
        ----------
        model_info : Dict[str, Any]
            Model information
        predictions : pd.Series
            Predictions
        actuals : Optional[pd.Series]
            Actual values (if available)
            
        Returns
        -------
        str
            Formatted context string for AI
        """
        try:
            context_parts = [
                "=" * 80,
                "PREDICTION CONTEXT",
                "=" * 80,
                "",
                "MODEL INFORMATION:",
                f"  • Model: {model_info.get('model', 'Unknown')}",
                f"  • Technique: {model_info.get('technique', 'Unknown')}",
                f"  • Training R²: {model_info.get('train_r2', 'N/A')}",
                f"  • Test R²: {model_info.get('test_r2', 'N/A')}",
                "",
                "PREDICTION STATISTICS:",
                f"  • Total Predictions: {len(predictions)}",
                f"  • Mean Prediction: {predictions.mean():.6f}",
                f"  • Std Dev: {predictions.std():.6f}",
                f"  • Min: {predictions.min():.6f}",
                f"  • Max: {predictions.max():.6f}",
                "",
            ]
            
            if actuals is not None:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                
                r2 = r2_score(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                mae = mean_absolute_error(actuals, predictions)
                
                context_parts.extend([
                    "PREDICTION PERFORMANCE:",
                    f"  • R² Score: {r2:.6f}",
                    f"  • RMSE: {rmse:.6f}",
                    f"  • MAE: {mae:.6f}",
                    "",
                ])
            
            context_parts.append("=" * 80)
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building prediction context: {e}")
            return "Prediction context available"
"""
AI Explanation Module
=====================
Integrates Gemini and ChatGPT APIs to provide AI-powered explanations of results.

Features:
- Support for both Google Gemini and OpenAI ChatGPT
- Explanations for training results
- Explanations for predictions
- Model performance interpretation
- Feature importance analysis explanations
- Easy provider switching
"""

import os
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class AIExplainer(ABC):
    """Base class for AI explanation providers."""
    
    @abstractmethod
    def explain_training_results(self, results_dict: Dict) -> str:
        """Explain training results."""
        pass
    
    @abstractmethod
    def explain_model_performance(self, model_metrics: Dict) -> str:
        """Explain model performance metrics."""
        pass
    
    @abstractmethod
    def explain_prediction(self, prediction: float, features: Dict, 
                          model_info: Dict) -> str:
        """Explain a prediction."""
        pass
    
    @abstractmethod
    def explain_feature_importance(self, importance_df: Any) -> str:
        """Explain feature importance."""
        pass


class GeminiExplainer(AIExplainer):
    """Google Gemini-based explanations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini Explainer.
        
        Parameters
        ----------
        api_key : str, optional
            Google API key. If None, reads from GOOGLE_API_KEY env var
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key not provided. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        try:
            import google.generativeai as genai
            self.genai = genai
            genai.configure(api_key=self.api_key)
            # Try gemini-1.5-pro first (stable model), fallback to gemini-pro
            try:
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("Gemini Explainer initialized with gemini-1.5-pro")
            except Exception:
                self.model = genai.GenerativeModel('gemini-pro')
                logger.info("Gemini Explainer initialized with gemini-pro fallback")
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
    
    
    def explain_training_results(self, results_dict: Dict) -> str:
        """Explain training results using Gemini."""
        prompt = self._build_training_prompt(results_dict)
        
        try:
            response = self.model.generate_content(prompt)
            logger.info("Gemini training explanation generated")
            return response.text
        except Exception as e:
            logger.error(f"Error generating training explanation: {str(e)}")
            return self._fallback_training_explanation(results_dict)
    
    
    def explain_model_performance(self, model_metrics: Dict) -> str:
        """Explain model performance using Gemini."""
        prompt = self._build_performance_prompt(model_metrics)
        
        try:
            response = self.model.generate_content(prompt)
            logger.info("Gemini performance explanation generated")
            return response.text
        except Exception as e:
            logger.error(f"Error generating performance explanation: {str(e)}")
            return self._fallback_performance_explanation(model_metrics)
    
    
    def explain_prediction(self, prediction: float, features: Dict,
                          model_info: Dict) -> str:
        """Explain a prediction using Gemini."""
        prompt = self._build_prediction_prompt(prediction, features, model_info)
        
        try:
            response = self.model.generate_content(prompt)
            logger.info("Gemini prediction explanation generated")
            return response.text
        except Exception as e:
            logger.error(f"Error generating prediction explanation: {str(e)}")
            return self._fallback_prediction_explanation(prediction, model_info)
    
    
    def explain_feature_importance(self, importance_df: Any) -> str:
        """Explain feature importance using Gemini."""
        prompt = self._build_importance_prompt(importance_df)
        
        try:
            response = self.model.generate_content(prompt)
            logger.info("Gemini feature importance explanation generated")
            return response.text
        except Exception as e:
            logger.error(f"Error generating importance explanation: {str(e)}")
            return self._fallback_importance_explanation(importance_df)
    
    
    def answer_user_query(self, question: str, context_str: str) -> str:
        """Answer user queries about model results using context and own knowledge."""
        prompt = f"""You are an expert soil scientist and machine learning specialist helping users understand their model training results and ML methodologies.

You have access to the following context about the training results:

{context_str}

User Question: {question}

IMPORTANT INSTRUCTIONS:
1. ALWAYS use the specific numbers and metrics from the context provided above.
2. When mentioning R², RMSE, MAE, RPD, or any other metrics - REFERENCE THE ACTUAL VALUES from the context.
3. DO NOT say metrics are "N/A" or "missing" if they are shown in the context above.
4. Be specific with numbers and metrics from the results.
5. Reference the results directly with actual values.
6. If explaining general ML concepts, relate them back to the specific results shown.
7. Keep responses concise but thorough (200-300 words).
8. Be helpful, informative, and professional.

Answer the question now, using the actual values from the context:
"""
        
        try:
            response = self.model.generate_content(prompt)
            result = response.text if hasattr(response, 'text') else str(response)
            logger.info(f"Gemini query response generated for: {question}")
            return result
        except Exception as e:
            logger.error(f"Error generating query response: {str(e)}")
            return None
    
    
    @staticmethod
    def _format_value(val: Any, decimal_places: int = 4) -> str:
        """Safely format a value, handling strings and numbers."""
        if isinstance(val, (int, float)):
            return f"{val:.{decimal_places}f}"
        return str(val)
    
    @staticmethod
    def _build_training_prompt(results_dict: Dict) -> str:
        """Build prompt for training results explanation."""
        best_model = results_dict.get('best_model', 'Unknown')
        best_technique = results_dict.get('best_technique', 'Unknown')
        test_r2 = GeminiExplainer._format_value(results_dict.get('test_r2', 'N/A'), 4)
        test_rmse = GeminiExplainer._format_value(results_dict.get('test_rmse', 'N/A'), 4)
        rpd = GeminiExplainer._format_value(results_dict.get('rpd', 'N/A'), 2)
        n_models = results_dict.get('n_models', 'Unknown')
        
        return f"""
You are an expert soil scientist and machine learning specialist.
Explain the following model training results to a user in simple, actionable terms.

Training Results Summary:
- Best Model: {best_model}
- Best Technique: {best_technique}
- Test R² Score: {test_r2}
- Test RMSE: {test_rmse}
- RPD (Residual Prediction Deviation): {rpd}
- Total Models Trained: {n_models}

Please provide:
1. Interpretation of the R² score and what it means
2. Assessment of model quality (poor/fair/good/excellent)
3. Key insights about model performance
4. Recommendations for use of this model
5. Suggestions for improvement if applicable

Keep the explanation concise but informative (200-300 words).
"""
    
    @staticmethod
    def _build_performance_prompt(model_metrics: Dict) -> str:
        """Build prompt for performance explanation."""
        r2_score = GeminiExplainer._format_value(model_metrics.get('R²', 'N/A'), 4)
        rmse = GeminiExplainer._format_value(model_metrics.get('RMSE', 'N/A'), 4)
        mae = GeminiExplainer._format_value(model_metrics.get('MAE', 'N/A'), 4)
        mape = GeminiExplainer._format_value(model_metrics.get('MAPE', 'N/A'), 4)
        rpd = GeminiExplainer._format_value(model_metrics.get('RPD', 'N/A'), 2)
        model = model_metrics.get('Model', 'Unknown')
        
        return f"""
Explain these model performance metrics to a soil scientist:

Metrics:
- R² Score: {r2_score}
- RMSE: {rmse}
- MAE: {mae}
- MAPE: {mape}
- RPD: {rpd}
- Model: {model}

Provide:
1. What each metric means
2. Whether the model is performing well
3. Any concerning metrics
4. Practical implications for soil property prediction

Keep it practical and actionable (150-250 words).
"""
    
    @staticmethod
    def _build_prediction_prompt(prediction: float, features: Dict,
                                model_info: Dict) -> str:
        """Build prompt for prediction explanation."""
        pred_val = GeminiExplainer._format_value(prediction, 2)
        model_name = model_info.get('model_name', 'Unknown')
        technique = model_info.get('technique', 'Unknown')
        r2_conf = GeminiExplainer._format_value(model_info.get('r2', 'N/A'), 4)
        
        return f"""
A machine learning model has made a soil property prediction.
Explain this result to the user.

Prediction: {pred_val}
Model Used: {model_name}
Model Technique: {technique}
Model Confidence (R²): {r2_conf}
Number of Input Features: {len(features)}

Explain:
1. What this prediction value means
2. How confident we can be in this prediction
3. What factors the model considered
4. Practical implications for soil science

Keep it user-friendly (150-200 words).
"""
    
    @staticmethod
    def _build_importance_prompt(importance_df: Any) -> str:
        """Build prompt for feature importance explanation."""
        top_features = importance_df.head(5).to_string() if hasattr(importance_df, 'head') else str(importance_df)
        
        return f"""
These are the most important spectral features for predicting soil properties:

Top Features:
{top_features}

Explain:
1. What these features represent in spectral analysis
2. Why these features are important for soil property prediction
3. How to interpret their relative importance
4. Practical recommendations based on this importance ranking

Keep it educational and practical (150-250 words).
"""
    
    @staticmethod
    def _fallback_training_explanation(results_dict: Dict) -> str:
        """Fallback explanation if API fails."""
        r2 = results_dict.get('test_r2', 0)
        rpd = results_dict.get('rpd', 0)
        
        if r2 > 0.9:
            quality = "excellent"
        elif r2 > 0.7:
            quality = "good"
        elif r2 > 0.5:
            quality = "fair"
        else:
            quality = "poor"
        
        return f"""
**Training Results Summary**

Best Model: {results_dict.get('best_model', 'Unknown')}
Performance: {quality.upper()}

Key Metrics:
- R² Score: {r2:.4f} (explains {r2*100:.1f}% of variance)
- RPD: {rpd:.2f} (ratio of std dev to RMSE)

The model achieved {quality} performance on the test set.
An R² of {r2:.4f} suggests the model can reasonably predict soil properties.
RPD of {rpd:.2f} indicates acceptable prediction capability.
"""
    
    @staticmethod
    def _fallback_performance_explanation(model_metrics: Dict) -> str:
        """Fallback performance explanation if API fails."""
        r2 = model_metrics.get('R²', 0)
        rmse = model_metrics.get('RMSE', 0)
        rpd = model_metrics.get('RPD', 0)
        model = model_metrics.get('Model', 'Unknown')
        
        if r2 > 0.9:
            assessment = "excellent"
        elif r2 > 0.7:
            assessment = "good"
        elif r2 > 0.5:
            assessment = "fair"
        else:
            assessment = "poor"
        
        return f"""
**Model Performance Assessment**

Model: {model}
Assessment: {assessment.upper()}

Key Metrics:
- R²: {r2:.4f} - Explains {r2*100:.1f}% of variance
- RMSE: {rmse:.4f} - Average prediction error
- RPD: {rpd:.2f} - Ratio of std dev to prediction error

This model shows {assessment} predictive ability for soil property estimation.
"""
    
    @staticmethod
    def _fallback_prediction_explanation(prediction: float, model_info: Dict) -> str:
        """Fallback prediction explanation if API fails."""
        r2 = model_info.get('r2', 0)
        model = model_info.get('model_name', 'Unknown')
        
        confidence = "high" if r2 > 0.8 else "moderate" if r2 > 0.6 else "low"
        
        return f"""
**Prediction Result**

Predicted Value: {prediction:.2f}
Model: {model}
Confidence Level: {confidence}

Based on the model's R² of {r2:.4f}, this prediction has {confidence} confidence.
The value represents the estimated soil property based on the input spectral data.
"""
    
    @staticmethod
    def _fallback_importance_explanation(importance_df: Any) -> str:
        """Fallback importance explanation if API fails."""
        return f"""
**Feature Importance Summary**

The model identifies certain spectral features as most important for predicting soil properties.
These top features capture the greatest variance in the prediction target.

Key insights:
- Features with higher importance scores have greater influence on predictions
- Spectral regions with high importance represent significant soil property signatures
- These features are recommended for focused data collection and analysis
"""


class ChatGPTExplainer(AIExplainer):
    """OpenAI ChatGPT-based explanations."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize ChatGPT Explainer.
        
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If None, reads from OPENAI_API_KEY env var
        model : str, default='gpt-3.5-turbo'
            Model to use (gpt-3.5-turbo or gpt-4)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"ChatGPT Explainer initialized with model {model}")
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install openai"
            )
    
    
    def explain_training_results(self, results_dict: Dict) -> str:
        """Explain training results using ChatGPT."""
        prompt = self._build_training_prompt(results_dict)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            logger.info("ChatGPT training explanation generated")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating training explanation: {str(e)}")
            return GeminiExplainer._fallback_training_explanation(results_dict)
    
    
    def explain_model_performance(self, model_metrics: Dict) -> str:
        """Explain model performance using ChatGPT."""
        prompt = self._build_performance_prompt(model_metrics)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            logger.info("ChatGPT performance explanation generated")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating performance explanation: {str(e)}")
            return GeminiExplainer._fallback_performance_explanation(model_metrics)
    
    
    def explain_prediction(self, prediction: float, features: Dict,
                          model_info: Dict) -> str:
        """Explain a prediction using ChatGPT."""
        prompt = self._build_prediction_prompt(prediction, features, model_info)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            logger.info("ChatGPT prediction explanation generated")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating prediction explanation: {str(e)}")
            return GeminiExplainer._fallback_prediction_explanation(prediction, model_info)
    
    
    def explain_feature_importance(self, importance_df: Any) -> str:
        """Explain feature importance using ChatGPT."""
        prompt = self._build_importance_prompt(importance_df)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            logger.info("ChatGPT feature importance explanation generated")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating importance explanation: {str(e)}")
            return GeminiExplainer._fallback_importance_explanation(importance_df)
    
    
    def answer_user_query(self, question: str, context_str: str) -> str:
        """Answer user query using ChatGPT."""
        prompt = f"""
You are an expert AI assistant helping analyze machine learning model results and soil science data.

USER CONTEXT:
{context_str}

USER QUESTION:
{question}

Please provide a clear, concise, and helpful answer based on the context provided.
Focus on practical insights and avoid unnecessary technical jargon when possible.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )
            logger.info("ChatGPT user query answered")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error answering user query: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    
    # Reuse prompts from GeminiExplainer
    @staticmethod
    def _build_training_prompt(results_dict: Dict) -> str:
        return GeminiExplainer._build_training_prompt(results_dict)
    
    @staticmethod
    def _build_performance_prompt(model_metrics: Dict) -> str:
        return GeminiExplainer._build_performance_prompt(model_metrics)
    
    @staticmethod
    def _build_prediction_prompt(prediction: float, features: Dict,
                                model_info: Dict) -> str:
        return GeminiExplainer._build_prediction_prompt(prediction, features, model_info)
    
    @staticmethod
    def _build_importance_prompt(importance_df: Any) -> str:
        return GeminiExplainer._build_importance_prompt(importance_df)


class AIExplainerFactory:
    """Factory for creating AI explainer instances."""
    
    @staticmethod
    def create_explainer(provider: str = "gemini",
                        api_key: Optional[str] = None,
                        **kwargs) -> AIExplainer:
        """
        Create an AI explainer instance.
        
        Parameters
        ----------
        provider : str, default='gemini'
            Provider to use: 'gemini' or 'chatgpt'
        api_key : str, optional
            API key for the provider
        **kwargs : dict
            Additional arguments for the provider
            
        Returns
        -------
        AIExplainer
            Explainer instance
        """
        provider = provider.lower()
        
        if provider == "gemini":
            return GeminiExplainer(api_key=api_key)
        elif provider == "chatgpt":
            return ChatGPTExplainer(api_key=api_key, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")


class ExplanationCache:
    """Cache for AI explanations to reduce API calls."""
    
    def __init__(self, max_cache_size: int = 100):
        """
        Initialize cache.
        
        Parameters
        ----------
        max_cache_size : int
            Maximum number of cached explanations
        """
        self.cache = {}
        self.max_size = max_cache_size
        self.hits = 0
        self.misses = 0
    
    
    def _get_key(self, explanation_type: str, data: Any) -> str:
        """Generate cache key."""
        import hashlib
        import json
        
        try:
            data_str = json.dumps(str(data), sort_keys=True)
        except:
            data_str = str(data)
        
        key = f"{explanation_type}:{hashlib.md5(data_str.encode()).hexdigest()}"
        return key
    
    
    def get(self, explanation_type: str, data: Any) -> Optional[str]:
        """Get explanation from cache."""
        key = self._get_key(explanation_type, data)
        
        if key in self.cache:
            self.hits += 1
            logger.info(f"Cache hit for {explanation_type}")
            return self.cache[key]
        
        self.misses += 1
        return None
    
    
    def put(self, explanation_type: str, data: Any, explanation: str) -> None:
        """Store explanation in cache."""
        key = self._get_key(explanation_type, data)
        
        # Simple eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest (first) entry
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = explanation
        logger.info(f"Cached explanation for {explanation_type}")
    
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


class ExplainerWithCache:
    """AI Explainer with caching capability."""
    
    def __init__(self, provider: str = "gemini",
                 api_key: Optional[str] = None,
                 cache_enabled: bool = True,
                 **kwargs):
        """
        Initialize explainer with optional caching.
        
        Parameters
        ----------
        provider : str
            AI provider to use
        api_key : str, optional
            API key
        cache_enabled : bool
            Whether to enable caching
        **kwargs
            Additional arguments
        """
        self.explainer = AIExplainerFactory.create_explainer(
            provider=provider,
            api_key=api_key,
            **kwargs
        )
        self.cache = ExplanationCache() if cache_enabled else None
        self.cache_enabled = cache_enabled
        logger.info(f"ExplainerWithCache initialized with caching: {cache_enabled}")
    
    
    def explain_training_results(self, results_dict: Dict) -> str:
        """Explain training results with caching."""
        if self.cache_enabled:
            cached = self.cache.get('training_results', str(results_dict))
            if cached:
                return cached
        
        explanation = self.explainer.explain_training_results(results_dict)
        
        if self.cache_enabled:
            self.cache.put('training_results', str(results_dict), explanation)
        
        return explanation
    
    
    def explain_model_performance(self, model_metrics: Dict) -> str:
        """Explain model performance with caching."""
        if self.cache_enabled:
            cached = self.cache.get('model_performance', str(model_metrics))
            if cached:
                return cached
        
        explanation = self.explainer.explain_model_performance(model_metrics)
        
        if self.cache_enabled:
            self.cache.put('model_performance', str(model_metrics), explanation)
        
        return explanation
    
    
    def explain_prediction(self, prediction: float, features: Dict,
                          model_info: Dict) -> str:
        """Explain prediction with caching."""
        cache_key = f"{prediction}:{str(model_info)}"
        
        if self.cache_enabled:
            cached = self.cache.get('prediction', cache_key)
            if cached:
                return cached
        
        explanation = self.explainer.explain_prediction(prediction, features, model_info)
        
        if self.cache_enabled:
            self.cache.put('prediction', cache_key, explanation)
        
        return explanation
    
    
    def explain_feature_importance(self, importance_df: Any) -> str:
        """Explain feature importance with caching."""
        if self.cache_enabled:
            cached = self.cache.get('feature_importance', str(importance_df))
            if cached:
                return cached
        
        explanation = self.explainer.explain_feature_importance(importance_df)
        
        if self.cache_enabled:
            self.cache.put('feature_importance', str(importance_df), explanation)
        
        return explanation
    
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return None
    
    
    def answer_user_query(self, question: str, context_str: str) -> str:
        """Answer user query with optional caching."""
        # For queries, we cache based on question + context hash to avoid duplicate API calls
        # but NOT when the context changes (different data = different cache key)
        if self.cache_enabled:
            cache_key = f"{question}:{hash(context_str) % 10000}"  # Use hash to keep key manageable
            cached = self.cache.get('user_query', cache_key)
            if cached:
                logger.info("Using cached response for user query")
                return cached
        
        answer = self.explainer.answer_user_query(question, context_str)
        
        if self.cache_enabled and answer:
            cache_key = f"{question}:{hash(context_str) % 10000}"
            self.cache.put('user_query', cache_key, answer)
        
        return answer
"""
Conversational AI Chat Interface
=================================
Interactive chat for querying models, results, and getting insights.

Features:
- Conversational Q&A about training results
- Model parameter queries
- Performance comparison questions
- Recommendation generation
- Chat history management (global persistent + cycle-specific)
- Context-aware responses
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class GlobalChatHistory:
    """Manage global persistent chat history."""
    
    HISTORY_FILE = Path("./logs/global_chat_history.json")
    
    @classmethod
    def initialize(cls):
        """Initialize history file."""
        cls.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not cls.HISTORY_FILE.exists():
            cls.HISTORY_FILE.write_text(json.dumps({"messages": []}, indent=2))
    
    @classmethod
    def add_message(cls, role: str, content: str, context_type: str = "general"):
        """Add message to global history."""
        cls.initialize()
        try:
            data = json.loads(cls.HISTORY_FILE.read_text())
            data["messages"].append({
                "timestamp": datetime.now().isoformat(),
                "role": role,
                "content": content,
                "context_type": context_type
            })
            cls.HISTORY_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error saving to global history: {e}")
    
    @classmethod
    def get_messages(cls, context_type: Optional[str] = None) -> List[Dict]:
        """Get messages from global history."""
        cls.initialize()
        try:
            data = json.loads(cls.HISTORY_FILE.read_text())
            messages = data.get("messages", [])
            
            if context_type:
                messages = [m for m in messages if m.get("context_type") == context_type]
            
            return messages
        except Exception as e:
            logger.error(f"Error reading global history: {e}")
            return []
    
    @classmethod
    def clear_history(cls):
        """Clear all history."""
        cls.initialize()
        cls.HISTORY_FILE.write_text(json.dumps({"messages": []}, indent=2))


class ChatInterface:
    """Conversational interface for model insights."""
    
    def __init__(self, ai_provider: str = "gemini", api_key: Optional[str] = None):
        """
        Initialize chat interface.
        
        Parameters
        ----------
        ai_provider : str
            AI provider ('gemini' or 'chatgpt')
        api_key : str, optional
            API key
        """
        self.ai_provider = ai_provider.lower()
        self.api_key = api_key
        self.cycle_chat_history = []  # Non-persistent, cycle-specific
        
        try:
            self.explainer = AIExplainerFactory.create_explainer(
                provider=self.ai_provider,
                api_key=api_key
            )
            self.ai_available = True
        except Exception as e:
            logger.warning(f"AI not available: {e}")
            self.ai_available = False
            self.explainer = None
        
        # Initialize global history
        GlobalChatHistory.initialize()
    
    
    def add_message(self, role: str, content: str, timestamp: Optional[str] = None):
        """
        Add message to chat history.
        
        Parameters
        ----------
        role : str
            'user' or 'assistant'
        content : str
            Message content
        timestamp : str, optional
            Message timestamp
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        message = {
            'role': role,
            'content': content,
            'timestamp': timestamp
        }
        
        # Add to cycle-specific history
        self.cycle_chat_history.append(message)
        
        # Add to global persistent history
        GlobalChatHistory.add_message(role, content, context_type="general")
    
    
    def get_chat_history(self, persistent: bool = False) -> List[Dict]:
        """
        Get chat history.
        
        Parameters
        ----------
        persistent : bool
            If True, get global persistent history; else get cycle history
            
        Returns
        -------
        List[Dict]
            Chat history
        """
        if persistent:
            return GlobalChatHistory.get_messages()
        return self.cycle_chat_history
    
    
    def clear_chat_history(self, persistent: bool = False):
        """
        Clear chat history.
        
        Parameters
        ----------
        persistent : bool
            If True, clear global history; else clear cycle history only
        """
        if persistent:
            GlobalChatHistory.clear_history()
        else:
            self.cycle_chat_history = []
    
    
    def query_results(self, question: str, context: Dict[str, Any]) -> str:
        """
        Query results with context.
        Returns the response string WITHOUT adding to history internally.
        History is managed by the UI layer (StreamlitChatUI).
        
        Parameters
        ----------
        question : str
            User question
        context : Dict[str, Any]
            Context data (results, models, etc.)
            
        Returns
        -------
        str
            Response
        """
        try:
            # Use comprehensive training context if available
            if 'training_context_str' in context and context['training_context_str']:
                context_str = context['training_context_str']
            else:
                # Fallback to building context from dictionary
                context_str = self._build_context_string(context)
            
            # Try AI first if available - ALWAYS use AI for comprehensive context
            if self.ai_available and hasattr(self.explainer, 'answer_user_query'):
                ai_response = self.explainer.answer_user_query(question, context_str)
                if ai_response:
                    # Return response WITHOUT adding to history (UI layer handles this)
                    return ai_response
                else:
                    logger.warning("AI returned None")
                    # If AI fails, return a generic message asking to retry
                    return "I encountered an issue generating a response. Please try again or rephrase your question."
            
            # If AI is not available, inform user
            logger.warning("AI not available, cannot process query")
            return "AI features are not available. Please ensure API keys are configured (GOOGLE_API_KEY or OPENAI_API_KEY)"
        
        except Exception as e:
            logger.error(f"Error in query_results: {e}")
            response = self._generate_fallback_response(question, context)
            return response
    
    
    def _build_context_string(self, context: Dict[str, Any]) -> str:
        """Build context string from data."""
        context_parts = []
        
        if 'summary' in context:
            summary = context['summary']
            best_r2 = summary.get('best_r2', 'N/A')
            mean_r2 = summary.get('mean_r2', 'N/A')
            best_r2_str = f"{best_r2:.4f}" if isinstance(best_r2, (int, float)) else str(best_r2)
            mean_r2_str = f"{mean_r2:.4f}" if isinstance(mean_r2, (int, float)) else str(mean_r2)
            
            # Ensure best_model and best_technique are proper strings
            best_model = summary.get('best_model', 'N/A')
            best_model_str = str(best_model) if best_model not in (None, 'N/A', '') else 'Unknown'
            
            best_technique = summary.get('best_technique', 'N/A')
            best_technique_str = str(best_technique) if best_technique not in (None, 'N/A', '') else 'Unknown'
            
            paradigm = summary.get('paradigm', 'Unknown')
            
            context_parts.append(f"""
Training Summary:
- Total Models: {summary.get('total_models', 'N/A')}
- Training Paradigm: {paradigm}
- Best R² Score: {best_r2_str}
- Mean R² Score: {mean_r2_str}
- Best Model: {best_model_str}
- Best Technique: {best_technique_str}
""")
            
            # Add hyperparameters if available
            hyperparams = summary.get('hyperparameters', {})
            if hyperparams:
                context_parts.append("- Key Hyperparameters: " + ", ".join([f"{k}={v}" for k, v in list(hyperparams.items())[:5]]))
        
        if 'top_models' in context:
            context_parts.append("\nTop Performing Models:")
            for i, model in enumerate(context['top_models'][:3], 1):
                # Handle both capitalized and lowercase key variants
                r2 = model.get('Test_R²', model.get('r2', 'N/A'))
                r2_str = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
                model_name = model.get('Model', model.get('model', 'N/A'))
                technique = model.get('Technique', model.get('technique', 'N/A'))
                context_parts.append(
                    f"{i}. {model_name} ({technique}): R²={r2_str}"
                )
        
        if 'statistics' in context and 'by_technique' in context['statistics']:
            context_parts.append("\nPerformance by Technique:")
            for tech, stats in context['statistics']['by_technique'].items():
                mean_r2_tech = stats.get('mean_r2', 'N/A')
                mean_r2_tech_str = f"{mean_r2_tech:.4f}" if isinstance(mean_r2_tech, (int, float)) else str(mean_r2_tech)
                context_parts.append(f"- {tech}: Mean R²={mean_r2_tech_str}")
        
        return '\n'.join(context_parts)
    
    
    def _generate_fallback_response(self, question: str, context: Dict[str, Any]) -> str:
        """Generate fallback response without AI."""
        q = question.lower()
        summary = context.get('summary', {})
        stats = context.get('statistics', {})
        top_models = context.get('top_models', [])
        
        # Best model query
        if 'best' in q and 'model' in q:
            best_model = summary.get('best_model', 'Unknown')
            best_technique = summary.get('best_technique', 'Unknown')
            best_r2 = summary.get('best_r2', 0)
            
            # Ensure they're proper strings
            best_model = str(best_model) if best_model not in (None, 'N/A', '') else 'Unknown'
            best_technique = str(best_technique) if best_technique not in (None, 'N/A', '') else 'Unknown'
            
            if isinstance(best_r2, (int, float)) and best_r2 > 0:
                return f"The best performing model is {best_model} using the {best_technique} technique with an R² score of {best_r2:.4f}."
            return f"The best performing model is {best_model} using the {best_technique} technique."
        
        # R² score query
        elif 'r2' in q or 'r squared' in q or 'variance' in q:
            best_r2 = summary.get('best_r2', 0)
            mean_r2 = summary.get('mean_r2', 0)
            if isinstance(best_r2, (int, float)) and isinstance(mean_r2, (int, float)):
                return f"The best R² score achieved is {best_r2:.4f}, which explains {best_r2*100:.1f}% of the variance. The mean R² across all models is {mean_r2:.4f}."
            return f"The best R² score is {best_r2}. The mean R² is {mean_r2}."
        
        # Technique comparison
        elif 'technique' in q and ('compare' in q or 'comparison' in q or 'different' in q):
            if 'by_technique' in stats:
                techs = stats['by_technique']
                tech_summaries = []
                for tech, tech_stats in sorted(techs.items(), key=lambda x: x[1].get('mean_r2', 0), reverse=True):
                    mean_r2 = tech_stats.get('mean_r2', 'N/A')
                    if isinstance(mean_r2, (int, float)):
                        tech_summaries.append(f"{tech}: Mean R²={mean_r2:.4f}")
                    else:
                        tech_summaries.append(f"{tech}: Mean R²={mean_r2}")
                if tech_summaries:
                    return "Preprocessing technique comparison:\n" + "\n".join(tech_summaries)
            return "The different preprocessing techniques show varying performance across models."
        
        # Production quality query
        elif ('quality' in q or 'production' in q or 'acceptable' in q) and 'model' in q:
            best_r2 = summary.get('best_r2', 0)
            if isinstance(best_r2, (int, float)):
                if best_r2 > 0.85:
                    return f"The best model with R²={best_r2:.4f} is excellent and suitable for production use. It explains over 85% of variance."
                elif best_r2 > 0.75:
                    return f"The best model with R²={best_r2:.4f} is good and can be considered for production with proper validation."
                elif best_r2 > 0.60:
                    return f"The best model with R²={best_r2:.4f} is fair but may need further optimization before production deployment."
                else:
                    return f"The best model with R²={best_r2:.4f} needs significant improvement before production use."
            return "Model quality assessment: Check the R² score and other metrics for suitability."
        
        # Consistency query
        elif 'consistent' in q or 'consistency' in q:
            if top_models:
                best_model = top_models[0].get('Model', top_models[0].get('model', 'N/A'))
                return f"The {best_model} model appears to be the most consistent performer across different techniques."
            return "Consistency analysis: Review the top performing models across techniques."
        
        # Recommendations query
        elif 'recommend' in q or 'suggestion' in q:
            best_model = summary.get('best_model', 'Unknown')
            best_technique = summary.get('best_technique', 'Unknown')
            best_r2 = summary.get('best_r2', 0)
            mean_r2 = summary.get('mean_r2', 0)
            
            # Ensure they're proper strings
            best_model = str(best_model) if best_model not in (None, 'N/A', '') else 'Unknown'
            best_technique = str(best_technique) if best_technique not in (None, 'N/A', '') else 'Unknown'
            
            if isinstance(best_r2, (int, float)) and isinstance(mean_r2, (int, float)) and best_r2 > 0 and mean_r2 > 0:
                improvement = ((best_r2 - mean_r2) / mean_r2 * 100) if mean_r2 > 0 else 0
                return f"Recommended: Deploy {best_model} with {best_technique} preprocessing (R²={best_r2:.4f}). This represents a {improvement:.1f}% improvement over average performance."
            return f"Recommended model: {best_model} with {best_technique} technique for deployment."
        
        # Improvement query
        elif 'improve' in q or 'hyperparameter' in q or 'tuning' in q:
            total_models = summary.get('total_models', 0)
            if total_models:
                return f"Total models trained: {total_models}. Hyperparameter tuning helps select the best configuration for each algorithm-technique combination."
            return "Hyperparameter tuning optimizes model performance by searching the parameter space."
        
        # Default response
        else:
            return "Based on the training results, the best model is " + summary.get('best_model', 'N/A') + ". For more detailed analysis, please refer to the statistics and visualization sections."


class QueryTemplate:
    """Predefined query templates for common questions."""
    
    TEMPLATES = {
        'best_model': "What is the best performing model and technique combination?",
        'technique_comparison': "How do the different preprocessing techniques compare?",
        'model_quality': "Is the best model of acceptable quality for production use?",
        'improvement': "How much did hyperparameter tuning improve the results?",
        'consistency': "Which model is most consistent across techniques?",
        'recommendations': "What are your recommendations for model selection?",
        'performance_gaps': "Why is there a performance gap between models?",
        'next_steps': "What steps would improve model performance further?",
    }
    
    @staticmethod
    def get_template_questions() -> List[str]:
        """Get list of template questions."""
        return list(QueryTemplate.TEMPLATES.values())
    
    @staticmethod
    def get_template_id(question: str) -> Optional[str]:
        """Get template ID from question text."""
        for template_id, template_text in QueryTemplate.TEMPLATES.items():
            if question.lower() == template_text.lower():
                return template_id
        return None


class StreamlitChatUI:
    """Streamlit UI components for chat interface."""
    
    @staticmethod
    def render_chat_interface(chat_interface: ChatInterface,
                            context: Dict[str, Any],
                            key_prefix: str = "chat",
                            show_global_history: bool = False):
        """
        Render complete chat interface in Streamlit.
        
        Parameters
        ----------
        chat_interface : ChatInterface
            Chat interface instance
        context : Dict[str, Any]
            Context data
        key_prefix : str
            Key prefix for Streamlit widgets
        show_global_history : bool
            If True, show global persistent history; else show cycle-specific history
        """
        import streamlit as st
        import os
        
        st.markdown("### 💬 AI Assistant - Ask Questions")
        
        # AI Provider Selection
        st.markdown("**Select AI Provider:**")
        col1, col2 = st.columns(2)
        with col1:
            ai_provider_option = st.radio(
                "Choose AI provider:",
                ["🤖 Gemini", "🔷 ChatGPT"],
                index=0 if st.session_state.ai_provider == 'gemini' else 1,
                horizontal=True,
                key=f"{key_prefix}_ai_provider_select",
                disabled=st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
            )
            new_provider = 'gemini' if 'Gemini' in ai_provider_option else 'chatgpt'
            if new_provider != st.session_state.ai_provider:
                st.session_state.ai_provider = new_provider
                st.rerun()
        
        with col2:
            api_status = "✅ Available" if st.session_state.ai_provider == 'gemini' and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "✅ Available" if st.session_state.ai_provider == 'openai' and os.getenv("OPENAI_API_KEY") else "❌ Not Configured"
            st.info(f"**Status:** {api_status}")
        
        st.markdown("---")
        
        # Initialize session state for history view toggle
        if f"{key_prefix}_show_global_hist" not in st.session_state:
            st.session_state[f"{key_prefix}_show_global_hist"] = False
        
        # Chat history display (collapsible)
        with st.expander("📚 Chat History", expanded=False):
            # History source toggle
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📝 Current Session", key=f"{key_prefix}_show_session_btn"):
                    st.session_state[f"{key_prefix}_show_global_hist"] = False
            
            with col2:
                if st.button("📚 All-Time History", key=f"{key_prefix}_show_global_btn"):
                    st.session_state[f"{key_prefix}_show_global_hist"] = True
            
            st.markdown("---")
            
            # Display appropriate history
            if st.session_state[f"{key_prefix}_show_global_hist"]:
                st.markdown("**All-Time Conversation History (Persistent)**")
                history_source = chat_interface.get_chat_history(persistent=True)
            else:
                st.markdown("**Current Session Chat**")
                history_source = chat_interface.get_chat_history(persistent=False)
            
            if history_source:
                for message in history_source:
                    if message['role'] == 'user':
                        st.write(f"**You**: {message['content']}")
                    else:
                        st.write(f"**Assistant**: {message['content']}")
            else:
                st.info("No messages yet. Ask a question to start!")
            
            # Clear button (only for current session)
            if not st.session_state[f"{key_prefix}_show_global_hist"]:
                if st.button("🗑️ Clear Session History", key=f"{key_prefix}_clear_session"):
                    chat_interface.clear_chat_history(persistent=False)
                    st.success("Session history cleared")
                    st.rerun()
        
        # Input section
        st.markdown("---")
        
        # Quick templates
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Quick Questions:**")
            template_questions = QueryTemplate.get_template_questions()
            selected_template = st.selectbox(
                "Or ask a predefined question",
                template_questions,
                key=f"{key_prefix}_template",
                disabled=st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
            )
        
        with col2:
            st.markdown("**Custom Question:**")
            custom_question = st.text_input(
                "Ask your own question about the results",
                key=f"{key_prefix}_custom",
                placeholder="e.g., Explain PLSR methodology, Why is PLSR best?, ...",
                disabled=st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
            )
        
        # Send button
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            pass
        
        with col2:
            send_template = st.button(
                "📤 Use Template",
                key=f"{key_prefix}_send_template",
                disabled=st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
            )
        
        with col3:
            send_custom = st.button(
                "📤 Send Custom",
                key=f"{key_prefix}_send_custom",
                disabled=st.session_state.get('report_generating', False) or st.session_state.get('ai_thinking', False)
            )
        
        # Process input - check if buttons were clicked
        question = None
        if send_template and selected_template:
            question = selected_template
        elif send_custom and custom_question:
            question = custom_question
        
        # Display latest messages first (reverse chronological)
        st.markdown("#### 💭 Latest Exchange")
        if chat_interface.cycle_chat_history:
            latest_pair = chat_interface.cycle_chat_history[-2:]
            for message in latest_pair:
                if message['role'] == 'user':
                    st.write(f"**You**: {message['content']}")
                else:
                    st.info(f"**Assistant**: {message['content']}")
        
        # Process question if one was provided
        if question:
            try:
                # Add user message
                chat_interface.add_message('user', question)
                
                # Get response with spinner
                with st.spinner("🤔 Thinking... This may take a moment."):
                    try:
                        response = chat_interface.query_results(question, context)
                    except Exception as query_error:
                        error_msg = str(query_error)
                        if "insufficient_quota" in error_msg or "quota" in error_msg.lower():
                            response = "❌ **API Quota Exceeded**\n\nThe AI service has reached its quota limit. Please check your API billing and quota settings.\n\nFor OpenAI: https://platform.openai.com/account/billing/overview\n\nPlease try again later or switch to a different AI provider."
                        elif "429" in error_msg or "rate" in error_msg.lower():
                            response = "❌ **API Rate Limited**\n\nToo many requests sent to the AI service. Please wait a moment and try again."
                        else:
                            response = f"❌ **Error**: {error_msg}\n\nPlease try again or check your API configuration."
                
                # Add assistant message
                chat_interface.add_message('assistant', response)
                
                # Display response
                if response.startswith("❌"):
                    st.error(response)
                else:
                    st.success("✅ Response:")
                    st.markdown(response)
                
                # Clear input fields by rerunning
                st.session_state[f"{key_prefix}_pending_question"] = None
                st.rerun()
            except Exception as e:
                logger.error(f"Error in query processing: {e}")
                st.error(f"Error: {str(e)}")
        
        st.markdown("---")
"""
Report Generation Module
========================
AI-powered report generation for training results and model performance.

Features:
- AI-generated insights for results
- Comparison reports
- Recommendation generation
- Custom prompt handling for user queries
- Report formatting and export
"""

import streamlit as st
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate AI-powered reports on training results."""
    
    def __init__(self, ai_provider: str = "gemini", api_key: Optional[str] = None,
                 raw_data: Optional[pd.DataFrame] = None, target_col: Optional[str] = None,
                 cv_strategy: Optional[str] = None, search_method: Optional[str] = None,
                 n_iter: Optional[int] = None, cv_folds: Optional[int] = None,
                 data_analytics_context: Optional[str] = None,
                 feature_importance_data: Optional[Dict[str, Any]] = None,
                 feature_engineering_config: Optional[Dict[str, Any]] = None,
                 feature_engineering_data: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator.
        
        Parameters
        ----------
        ai_provider : str
            AI provider ('gemini' or 'chatgpt')
        api_key : str, optional
            API key for AI provider
        raw_data : pd.DataFrame, optional
            Raw dataset for context
        target_col : str, optional
            Target column name
        cv_strategy : Optional[str]
            Cross-validation strategy used ('k-fold' or 'leave-one-out')
        search_method : Optional[str]
            Hyperparameter search method ('grid' or 'random')
        n_iter : Optional[int]
            Number of iterations for RandomizedSearch
        cv_folds : Optional[int]
            Number of CV folds for K-Fold strategy
        data_analytics_context : Optional[str]
            Data analytics context string
        feature_importance_data : Optional[Dict[str, Any]]
            Feature importance data from all models
        feature_engineering_config : Optional[Dict[str, Any]]
            Feature engineering configuration
        feature_engineering_data : Optional[Dict[str, Any]]
            Feature engineering statistics and values
        """
        self.ai_provider = ai_provider.lower()
        self.api_key = api_key
        self.raw_data = raw_data
        self.target_col = target_col
        self.cv_strategy = cv_strategy
        self.search_method = search_method
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.data_analytics_context = data_analytics_context
        self.feature_importance_data = feature_importance_data
        self.feature_engineering_config = feature_engineering_config
        self.feature_engineering_data = feature_engineering_data
        
        # Try to initialize AI explainer
        try:
            self.explainer = ExplainerWithCache(
                provider=self.ai_provider,
                api_key=api_key,
                cache_enabled=True
            )
            self.ai_available = True
        except Exception as e:
            logger.warning(f"AI not available: {e}")
            self.ai_available = False
            self.explainer = None
    
    
    def generate_training_report(self, results_df: pd.DataFrame,
                                paradigm: str = "Standard",
                                include_ai_insights: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive training report.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            Training results
        paradigm : str
            Training paradigm (Standard, Tuned, etc.)
        include_ai_insights : bool
            Whether to include AI-generated insights
            
        Returns
        -------
        Dict[str, Any]
            Report dictionary
        """
        report = {
            'title': f'{paradigm} Training Report',
            'timestamp': datetime.now().isoformat(),
            'paradigm': paradigm,
            'summary': self._generate_summary(results_df),
            'statistics': self._calculate_statistics(results_df),
            'top_models': self._get_top_models(results_df),
            'technique_analysis': self._analyze_techniques(results_df),
        }
        
        if include_ai_insights and self.ai_available:
            report['ai_insights'] = self._generate_ai_insights(results_df, paradigm)
        
        return report
    
    
    def generate_comparison_report(self, standard_results: pd.DataFrame,
                                  tuned_results: pd.DataFrame,
                                  include_ai_insights: bool = True) -> Dict[str, Any]:
        """
        Generate comparison report between standard and tuned models.
        
        Parameters
        ----------
        standard_results : pd.DataFrame
            Standard training results
        tuned_results : pd.DataFrame
            Tuned training results
        include_ai_insights : bool
            Whether to include AI insights
            
        Returns
        -------
        Dict[str, Any]
            Comparison report
        """
        standard_summary = self._generate_summary(standard_results)
        tuned_summary = self._generate_summary(tuned_results)
        
        report = {
            'title': 'Standard vs Tuned Paradigm Comparison Report',
            'timestamp': datetime.now().isoformat(),
            'standard_summary': standard_summary,
            'tuned_summary': tuned_summary,
            'improvements': self._calculate_improvements(standard_results, tuned_results),
            'best_overall': self._find_best_overall(standard_results, tuned_results),
            'standard_top_models': self._get_top_models(standard_results, top_n=5),
            'tuned_top_models': self._get_top_models(tuned_results, top_n=5),
        }
        
        if include_ai_insights and self.ai_available:
            report['ai_recommendations'] = self._generate_recommendations(
                standard_results, tuned_results
            )
        
        return report
    
    
    def _generate_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics."""
        try:
            # Ensure Test_R² is numeric
            r2_values = pd.to_numeric(results_df['Test_R²'], errors='coerce')
            
            return {
                'total_models': len(results_df),
                'best_r2': float(r2_values.max()),
                'worst_r2': float(r2_values.min()),
                'mean_r2': float(r2_values.mean()),
                'median_r2': float(r2_values.median()),
                'std_r2': float(r2_values.std()),
                'best_model': results_df.loc[r2_values.idxmax(), 'Model'],
                'best_technique': results_df.loc[r2_values.idxmax(), 'Technique'],
            }
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                'total_models': len(results_df),
                'best_r2': 0.0,
                'worst_r2': 0.0,
                'mean_r2': 0.0,
                'median_r2': 0.0,
                'std_r2': 0.0,
                'best_model': 'Unknown',
                'best_technique': 'Unknown',
            }
    
    
    def _calculate_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed statistics by technique and model."""
        stats = {}
        
        # By technique
        stats['by_technique'] = {}
        for technique in results_df['Technique'].unique():
            tech_data = results_df[results_df['Technique'] == technique]
            stats['by_technique'][technique] = {
                'count': len(tech_data),
                'mean_r2': float(tech_data['Test_R²'].mean()),
                'max_r2': float(tech_data['Test_R²'].max()),
                'min_r2': float(tech_data['Test_R²'].min()),
            }
        
        # By model
        stats['by_model'] = {}
        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]
            stats['by_model'][model] = {
                'count': len(model_data),
                'mean_r2': float(model_data['Test_R²'].mean()),
                'max_r2': float(model_data['Test_R²'].max()),
                'min_r2': float(model_data['Test_R²'].min()),
            }
        
        return stats
    
    
    def _get_top_models(self, results_df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Get top N performing models."""
        top = results_df.nlargest(top_n, 'Test_R²')
        return [
            {
                'rank': i + 1,
                'model': row['Model'],
                'technique': row['Technique'],
                'r2': float(row['Test_R²']),
                'rmse': float(row['Test_RMSE']) if 'Test_RMSE' in results_df.columns and pd.notna(row['Test_RMSE']) else 0.0,
                'mae': float(row['Test_MAE']) if 'Test_MAE' in results_df.columns and pd.notna(row['Test_MAE']) else 0.0,
                'rpd': float(row['RPD']) if 'RPD' in results_df.columns and pd.notna(row['RPD']) else 0.0,
            }
            for i, (_, row) in enumerate(top.iterrows())
        ]
    
    
    def _analyze_techniques(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by technique."""
        analysis = {}
        
        for technique in results_df['Technique'].unique():
            tech_data = results_df[results_df['Technique'] == technique]
            analysis[technique] = {
                'performance': 'good' if tech_data['Test_R²'].mean() > 0.7 else 'fair' if tech_data['Test_R²'].mean() > 0.5 else 'poor',
                'avg_r2': float(tech_data['Test_R²'].mean()),
                'consistency': float(tech_data['Test_R²'].std()),
                'best_model': tech_data.loc[tech_data['Test_R²'].idxmax(), 'Model'],
            }
        
        return analysis
    
    
    def _calculate_improvements(self, standard: pd.DataFrame,
                               tuned: pd.DataFrame) -> Dict[str, Any]:
        """Calculate improvements from tuning."""
        try:
            # Ensure numeric columns
            std_r2 = pd.to_numeric(standard['Test_R²'], errors='coerce')
            tuned_r2 = pd.to_numeric(tuned['Test_R²'], errors='coerce')
            
            std_rmse = pd.to_numeric(standard['Test_RMSE'], errors='coerce') if 'Test_RMSE' in standard.columns else pd.Series([0])
            tuned_rmse = pd.to_numeric(tuned['Test_RMSE'], errors='coerce') if 'Test_RMSE' in tuned.columns else pd.Series([0])
            
            std_mae = pd.to_numeric(standard['Test_MAE'], errors='coerce') if 'Test_MAE' in standard.columns else pd.Series([0])
            tuned_mae = pd.to_numeric(tuned['Test_MAE'], errors='coerce') if 'Test_MAE' in tuned.columns else pd.Series([0])
            
            # Calculate best values
            std_best_r2 = std_r2.max()
            tuned_best_r2 = tuned_r2.max()
            
            std_best_rmse = std_rmse.min() if len(std_rmse) > 0 else 0
            tuned_best_rmse = tuned_rmse.min() if len(tuned_rmse) > 0 else 0
            
            std_best_mae = std_mae.min() if len(std_mae) > 0 else 0
            tuned_best_mae = tuned_mae.min() if len(tuned_mae) > 0 else 0
            
            # Calculate mean values
            std_mean_r2 = std_r2.mean()
            tuned_mean_r2 = tuned_r2.mean()
            
            # Calculate improvements
            best_r2_improvement = tuned_best_r2 - std_best_r2
            mean_r2_improvement = tuned_mean_r2 - std_mean_r2
            rmse_improvement = std_best_rmse - tuned_best_rmse  # Lower is better
            mae_improvement = std_best_mae - tuned_best_mae  # Lower is better
            
            # Calculate percentage improvements
            best_r2_improvement_percent = ((tuned_best_r2 - std_best_r2) / abs(std_best_r2) * 100) if std_best_r2 != 0 else 0
            
            return {
                'standard_best_r2': float(std_best_r2),
                'tuned_best_r2': float(tuned_best_r2),
                'standard_mean_r2': float(std_mean_r2),
                'tuned_mean_r2': float(tuned_mean_r2),
                'best_r2_improvement': float(best_r2_improvement),
                'mean_r2_improvement': float(mean_r2_improvement),
                'rmse_improvement': float(rmse_improvement),
                'mae_improvement': float(mae_improvement),
                'improvement_percent': float(best_r2_improvement_percent),
                'tuning_beneficial': best_r2_improvement > 0,
            }
        except Exception as e:
            logger.error(f"Error calculating improvements: {e}")
            return {
                'standard_best_r2': 0.0,
                'tuned_best_r2': 0.0,
                'standard_mean_r2': 0.0,
                'tuned_mean_r2': 0.0,
                'best_r2_improvement': 0.0,
                'mean_r2_improvement': 0.0,
                'rmse_improvement': 0.0,
                'mae_improvement': 0.0,
                'improvement_percent': 0.0,
                'tuning_beneficial': False,
            }
    
    
    def _find_best_overall(self, standard: pd.DataFrame,
                          tuned: pd.DataFrame) -> Dict[str, Any]:
        """Find the best overall model."""
        try:
            # Ensure both dataframes have the 'Model' and 'Technique' columns
            if 'Model' not in standard.columns or 'Model' not in tuned.columns:
                logger.warning("Model column not found in results")
                return {
                    'model': 'Unknown',
                    'technique': 'Unknown',
                    'paradigm': 'Unknown',
                    'r2': 0.0,
                    'rmse': 0.0,
                }
            
            # Create copies and assign paradigm
            std_copy = standard.copy()
            std_copy['paradigm'] = 'Standard'
            
            tuned_copy = tuned.copy()
            tuned_copy['paradigm'] = 'Tuned'
            
            all_results = pd.concat([std_copy, tuned_copy], ignore_index=True)
            
            # Ensure Test_R² is numeric
            all_results['Test_R²'] = pd.to_numeric(all_results['Test_R²'], errors='coerce')
            
            # Drop NaN values in Test_R²
            valid_results = all_results.dropna(subset=['Test_R²'])
            
            if len(valid_results) == 0:
                logger.warning("No valid R² values found in comparison")
                return {
                    'model': 'Unknown',
                    'technique': 'Unknown',
                    'paradigm': 'Unknown',
                    'r2': 0.0,
                    'rmse': 0.0,
                }
            
            # Find the best based on Test_R²
            best_idx = valid_results['Test_R²'].idxmax()
            best_row = valid_results.loc[best_idx]
            
            return {
                'model': str(best_row['Model']),
                'technique': str(best_row['Technique']),
                'paradigm': str(best_row['paradigm']),
                'r2': float(best_row['Test_R²']),
                'rmse': float(best_row.get('Test_RMSE', 0)),
            }
        except Exception as e:
            logger.error(f"Error finding best overall: {e}", exc_info=True)
            return {
                'model': 'Unknown',
                'technique': 'Unknown',
                'paradigm': 'Unknown',
                'r2': 0.0,
                'rmse': 0.0,
            }
    
    
    def _generate_ai_insights(self, results_df: pd.DataFrame, paradigm: str = "Standard") -> str:
        """Generate AI insights using explainer."""
        if not self.explainer:
            return "AI insights not available."
        
        try:
            # Build comprehensive context using context builder with all training configuration
            try:
                context_str = ContextBuilder.build_training_context(
                    results_df,
                    self.raw_data if self.raw_data is not None else pd.DataFrame(),
                    self.target_col if self.target_col else 'target',
                    paradigm=paradigm,
                    data_analytics_context=self.data_analytics_context,
                    feature_engineering_config=self.feature_engineering_config,
                    feature_engineering_data=self.feature_engineering_data,
                    feature_importance_data=self.feature_importance_data,
                    cv_strategy=self.cv_strategy,
                    search_method=self.search_method,
                    n_iter=self.n_iter,
                    cv_folds=self.cv_folds
                )
                
                # Log context details for debugging
                logger.info(f"Context string length: {len(context_str)} characters")
                logger.debug(f"Context contains R² mentions: {'R²' in context_str}")
                logger.debug(f"Context contains RMSE mentions: {'RMSE' in context_str}")
                logger.debug(f"Context contains metric values: {('0.8770' in context_str or '0.877' in context_str)}")
                logger.debug(f"CV Strategy: {self.cv_strategy}, Search Method: {self.search_method}")
                logger.debug(f"Raw data shape: {self.raw_data.shape if self.raw_data is not None else 'None'}")
                logger.debug(f"Results df shape: {results_df.shape}")
                
                # Write context to file for debugging
                try:
                    context_file = './logs/training_context_debug.txt'
                    with open(context_file, 'w') as f:
                        f.write(context_str)
                    logger.info(f"Context written to {context_file}")
                except Exception as ce:
                    logger.warning(f"Could not write context to file: {ce}")
                
                # Use answer_user_query for better AI response
                insights = self.explainer.answer_user_query(
                    "Analyze these model training results and provide: 1) Interpretation of the R², RMSE, MAE, and RPD metrics shown, 2) Assessment of model quality based on the actual metric values, 3) Key performance insights considering the CV strategy and hyperparameter search method used, 4) Recommendations for deployment or improvement. Use the specific numbers provided in the results.",
                    context_str
                )
                
                if insights:
                    logger.info(f"AI insights generated successfully ({len(insights)} characters)")
                else:
                    logger.warning("AI returned empty insights")
                
                return insights if insights else "Could not generate AI insights."
            except Exception as e:
                logger.error(f"Context builder approach failed: {e}", exc_info=True)
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                # Fallback to the old method
                results_dict = self._generate_summary(results_df)
                insights = self.explainer.explain_training_results(results_dict)
                return insights
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return "Could not generate AI insights."
    
    
    def _generate_recommendations(self, standard: pd.DataFrame,
                                 tuned: pd.DataFrame) -> str:
        """Generate AI-powered recommendations."""
        if not self.explainer:
            return "Recommendations not available."
        
        try:
            comparison = self._calculate_improvements(standard, tuned)
            
            prompt = f"""
Based on these comparison results between Standard and Tuned models:

Standard Best R²: {comparison['standard_best_r2']:.4f}
Tuned Best R²: {comparison['tuned_best_r2']:.4f}
Improvement: {comparison['improvement_percent']:.2f}%

Provide recommendations on:
1. Whether tuning was beneficial
2. Which paradigm to use for production
3. How to further improve model performance
4. Any concerns or limitations to note

Keep recommendations practical and actionable (200-300 words).
            """
            
            # Use ChatGPT/Gemini to generate recommendations
            response = self.explainer.explainer.model.generate_content(prompt) if hasattr(self.explainer.explainer, 'model') else None
            
            if response:
                return response.text if hasattr(response, 'text') else str(response)
            else:
                return "Could not generate recommendations."
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return "Could not generate recommendations."
    
    
    def format_report_as_markdown(self, report: Dict[str, Any]) -> str:
        """Format report as Markdown string."""
        md = f"""
# {report.get('title', 'Report')}

**Generated**: {report.get('timestamp', 'N/A')}

"""
        
        # Check if this is a comparison report (has standard_summary and tuned_summary)
        if 'standard_summary' in report and 'tuned_summary' in report:
            md += self._format_comparison_summary(report)
            md += self._format_improvements(report.get('improvements', {}))
            if 'standard_top_models' in report:
                md += "## Top Performing Models - Standard Paradigm\n\n"
                for model in report['standard_top_models'][:5]:
                    md += self._format_model_entry(model)
            if 'tuned_top_models' in report:
                md += "## Top Performing Models - Tuned Paradigm\n\n"
                for model in report['tuned_top_models'][:5]:
                    md += self._format_model_entry(model)
        else:
            # Original single-paradigm report format
            md += "## Summary\n\n"
            if 'summary' in report:
                summary = report['summary']
                best_r2 = summary.get('best_r2', 0.0)
                mean_r2 = summary.get('mean_r2', 0.0)
                best_r2_str = f"{best_r2:.4f}" if isinstance(best_r2, (int, float)) else str(best_r2)
                mean_r2_str = f"{mean_r2:.4f}" if isinstance(mean_r2, (int, float)) else str(mean_r2)
                md += f"""
- **Total Models**: {summary.get('total_models', 'N/A')}
- **Best R² Score**: {best_r2_str}
- **Mean R² Score**: {mean_r2_str}
- **Best Model**: {summary.get('best_model', 'N/A')}
- **Best Technique**: {summary.get('best_technique', 'N/A')}

"""
            
            if 'top_models' in report:
                md += "## Top Performing Models\n\n"
                for model in report['top_models'][:5]:
                    md += self._format_model_entry(model)
        
        if 'ai_insights' in report:
            md += f"## AI Insights\n\n{report['ai_insights']}\n\n"
        
        if 'ai_recommendations' in report:
            md += f"## Recommendations\n\n{report['ai_recommendations']}\n\n"
        
        return md
    
    
    def _format_comparison_summary(self, report: Dict[str, Any]) -> str:
        """Format comparison summary section."""
        md = "## Summary Comparison\n\n"
        
        standard = report.get('standard_summary', {})
        tuned = report.get('tuned_summary', {})
        
        md += "### Standard Paradigm\n\n"
        md += f"- **Total Models**: {standard.get('total_models', 'N/A')}\n"
        md += f"- **Best R² Score**: {standard.get('best_r2', 0.0):.4f}\n"
        md += f"- **Mean R² Score**: {standard.get('mean_r2', 0.0):.4f}\n"
        md += f"- **Median R² Score**: {standard.get('median_r2', 0.0):.4f}\n"
        md += f"- **Std Dev**: {standard.get('std_r2', 0.0):.4f}\n"
        md += f"- **Best Model**: {standard.get('best_model', 'N/A')} ({standard.get('best_technique', 'N/A')})\n\n"
        
        md += "### Tuned Paradigm\n\n"
        md += f"- **Total Models**: {tuned.get('total_models', 'N/A')}\n"
        md += f"- **Best R² Score**: {tuned.get('best_r2', 0.0):.4f}\n"
        md += f"- **Mean R² Score**: {tuned.get('mean_r2', 0.0):.4f}\n"
        md += f"- **Median R² Score**: {tuned.get('median_r2', 0.0):.4f}\n"
        md += f"- **Std Dev**: {tuned.get('std_r2', 0.0):.4f}\n"
        md += f"- **Best Model**: {tuned.get('best_model', 'N/A')} ({tuned.get('best_technique', 'N/A')})\n\n"
        
        return md
    
    
    def _format_improvements(self, improvements: Dict[str, Any]) -> str:
        """Format improvements section."""
        md = "## Performance Improvements (Tuned vs Standard)\n\n"
        
        if not improvements:
            return md + "No significant improvements data available.\n\n"
        
        md += f"- **Best R² Improvement**: {improvements.get('best_r2_improvement', 0.0):.4f}\n"
        md += f"- **Mean R² Improvement**: {improvements.get('mean_r2_improvement', 0.0):.4f}\n"
        md += f"- **RMSE Improvement**: {improvements.get('rmse_improvement', 0.0):.4f}\n"
        md += f"- **MAE Improvement**: {improvements.get('mae_improvement', 0.0):.4f}\n\n"
        
        return md
    
    
    def _format_model_entry(self, model: Dict[str, Any]) -> str:
        """Format a single model entry."""
        r2_str = f"{model['r2']:.4f}" if isinstance(model['r2'], (int, float)) else str(model['r2'])
        rmse_str = f"{model['rmse']:.4f}" if isinstance(model['rmse'], (int, float)) and model['rmse'] > 0 else "N/A"
        mae_str = f"{model['mae']:.4f}" if isinstance(model['mae'], (int, float)) and model['mae'] > 0 else "N/A"
        rpd_str = f"{model['rpd']:.4f}" if isinstance(model['rpd'], (int, float)) and model['rpd'] > 0 else "N/A"
        
        md = f"### {model['rank']}. {model['model']} ({model['technique']})\n"
        md += f"- R² Score: {r2_str}\n"
        md += f"- RMSE: {rmse_str}\n"
        md += f"- MAE: {mae_str}\n"
        md += f"- RPD: {rpd_str}\n\n"
        
        return md
    
    
    def format_report_as_html(self, report: Dict[str, Any]) -> str:
        """Format report as HTML string."""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }}
                h1 {{ color: #2E86AB; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 1.5em; font-weight: bold; color: #2E86AB; }}
                .model-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .insights {{ background-color: #f0f2f6; padding: 15px; border-radius: 5px; border-left: 4px solid #2E86AB; }}
            </style>
        </head>
        <body>
        <h1>{report.get('title', 'Report')}</h1>
        <p><strong>Generated</strong>: {report.get('timestamp', 'N/A')}</p>
        
        <h2>Summary</h2>
        """
        
        if 'summary' in report:
            summary = report['summary']
            best_r2 = summary.get('best_r2', 0.0)
            mean_r2 = summary.get('mean_r2', 0.0)
            best_r2_str = f"{best_r2:.4f}" if isinstance(best_r2, (int, float)) else str(best_r2)
            mean_r2_str = f"{mean_r2:.4f}" if isinstance(mean_r2, (int, float)) else str(mean_r2)
            html += f"""
            <div class="metric">
                <div>Total Models</div>
                <div class="metric-value">{summary.get('total_models', 'N/A')}</div>
            </div>
            <div class="metric">
                <div>Best R²</div>
                <div class="metric-value">{best_r2_str}</div>
            </div>
            <div class="metric">
                <div>Mean R²</div>
                <div class="metric-value">{mean_r2_str}</div>
            </div>
            """
        
        if 'top_models' in report:
            html += "<h2>Top Performing Models</h2>"
            for model in report['top_models'][:5]:
                rmse_val = f"{model['rmse']:.4f}" if isinstance(model['rmse'], (int, float)) and model['rmse'] > 0 else "N/A"
                mae_val = f"{model['mae']:.4f}" if isinstance(model['mae'], (int, float)) and model['mae'] > 0 else "N/A"
                rpd_val = f"{model['rpd']:.4f}" if isinstance(model['rpd'], (int, float)) and model['rpd'] > 0 else "N/A"
                html += f"""
                <div class="model-card">
                    <strong>#{model['rank']}: {model['model']} ({model['technique']})</strong>
                    <br>R² Score: {model['r2']:.4f}
                    <br>RMSE: {rmse_val}
                    <br>MAE: {mae_val}
                    <br>RPD: {rpd_val}
                </div>
                """
        
        if 'ai_insights' in report:
            html += f"""
            <h2>AI Insights</h2>
            <div class="insights">
                {report['ai_insights']}
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
