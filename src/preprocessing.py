"""
Preprocessing Module
====================
This module handles data cleaning, normalization, spectral transformations,
and feature engineering for spectral soil data.

Classes:
    - SpectralPreprocessor: Main preprocessing class

Functions:
    - apply_reflectance(): Direct spectral reflectance
    - apply_absorbance(): Log-based absorbance transformation
    - apply_continuum_removal(): Continuum removal technique
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name__)


class SpectralPreprocessor:
    """
    A comprehensive preprocessing class for spectral soil data.
    
    Supports:
    - Data normalization (StandardScaler, MinMaxScaler, RobustScaler)
    - Spectral transformations (Reflectance, Absorbance, Continuum Removal)
    - Smoothing and denoising
    - Feature engineering (derivatives, moving averages)
    - Missing value handling
    
    Attributes:
        technique (str): Preprocessing technique used
        normalizer: Fitted scaler object
        smoothing_applied (bool): Whether smoothing was applied
    """
    
    def __init__(self):
        """Initialize SpectralPreprocessor."""
        self.technique = None
        self.normalizer = None
        self.is_fitted = False
        self.scaler_type = None
        self.feature_names = None
        logger.info("SpectralPreprocessor initialized")
    
    
    def fit(self, X: np.ndarray, technique: str = 'reflectance', 
            scaler: str = 'standard', **kwargs) -> 'SpectralPreprocessor':
        """
        Fit preprocessor on training data.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Training data (n_samples, n_features)
        technique : str, default='reflectance'
            Spectral transformation technique:
            - 'reflectance': Use as-is
            - 'absorbance': Log(1/R) transformation
            - 'continuum_removal': Continuum-removed reflectance
        scaler : str, default='standard'
            Normalization method:
            - 'standard': StandardScaler (mean=0, std=1)
            - 'minmax': MinMaxScaler ([0, 1])
            - 'robust': RobustScaler (median & IQR)
        **kwargs : dict
            Additional arguments for specific techniques
            
        Returns
        -------
        self
            Fitted preprocessor
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if X is None or len(X) == 0:
            raise ValueError("X cannot be None or empty")
        
        self.technique = technique.lower()
        self.scaler_type = scaler.lower()
        
        # Initialize scaler
        if self.scaler_type == 'standard':
            self.normalizer = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.normalizer = MinMaxScaler()
        elif self.scaler_type == 'robust':
            self.normalizer = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler: {self.scaler_type}")
        
        logger.info(f"Fitting preprocessor with technique={self.technique}, scaler={self.scaler_type}")
        
        # Apply spectral transformation
        X_transformed = self._apply_spectral_technique(X)
        
        # Fit normalizer on transformed data
        self.normalizer.fit(X_transformed)
        
        self.is_fitted = True
        logger.info("Preprocessor fitting complete")
        
        return self
    
    
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Data to transform (n_samples, n_features)
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Preprocessed data
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        if X is None or len(X) == 0:
            raise ValueError("X cannot be None or empty")
        
        # Apply spectral transformation
        X_transformed = self._apply_spectral_technique(X)
        
        # Apply normalization
        X_normalized = self.normalizer.transform(X_transformed)
        
        logger.info(f"Transformed data shape: {X_normalized.shape}")
        
        return X_normalized
    
    
    def fit_transform(self, X: np.ndarray, technique: str = 'reflectance',
                      scaler: str = 'standard', **kwargs) -> np.ndarray:
        """
        Fit and transform data in one step.
        
        Parameters
        ----------
        X : np.ndarray
            Data to fit and transform
        technique : str, default='reflectance'
            Spectral transformation technique
        scaler : str, default='standard'
            Normalization method
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        np.ndarray
            Preprocessed data
        """
        return self.fit(X, technique, scaler, **kwargs).transform(X)
    
    
    def _apply_spectral_technique(self, X: np.ndarray) -> np.ndarray:
        """
        Apply spectral transformation technique.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data
            
        Returns
        -------
        np.ndarray
            Transformed data
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.technique == 'reflectance':
            return self._reflectance(X)
        elif self.technique == 'absorbance':
            return self._absorbance(X)
        elif self.technique == 'continuum_removal':
            return self._continuum_removal(X)
        else:
            raise ValueError(f"Unknown technique: {self.technique}")
    
    
    @staticmethod
    def _reflectance(X: np.ndarray) -> np.ndarray:
        """
        Reflectance transformation (identity - no transformation).
        
        Direct use of spectral reflectance values.
        
        Parameters
        ----------
        X : np.ndarray
            Input reflectance data
            
        Returns
        -------
        np.ndarray
            Reflectance (same as input)
        """
        return X.copy()
    
    
    @staticmethod
    def _absorbance(X: np.ndarray) -> np.ndarray:
        """
        Absorbance transformation: log(1/R).
        
        Converts reflectance to absorbance using Beer-Lambert law.
        A = log10(1/R) = -log10(R)
        
        Parameters
        ----------
        X : np.ndarray
            Input reflectance data (0-1 range recommended)
            
        Returns
        -------
        np.ndarray
            Absorbance transformed data
        """
        # Ensure values are in valid range to avoid log of 0
        X_safe = np.clip(X, 1e-8, 1.0)
        # Calculate absorbance
        A = -np.log10(X_safe)
        return A
    
    
    @staticmethod
    def _continuum_removal(X: np.ndarray) -> np.ndarray:
        """
        Continuum removal transformation.
        
        Removes the background continuum to enhance absorption features.
        CR = R / Continuum
        
        Parameters
        ----------
        X : np.ndarray
            Input reflectance data (n_samples, n_wavelengths)
            
        Returns
        -------
        np.ndarray
            Continuum-removed reflectance
        """
        X_cr = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            spectrum = X[i, :]
            
            # Create convex hull to estimate continuum
            # Use simple method: interpolate between max/min points
            n_points = len(spectrum)
            
            # Find turning points (approximate convex hull)
            hull_indices = np.array([0, n_points - 1])  # Start and end points
            
            # Add extreme points
            for j in range(1, n_points - 1):
                if spectrum[j] > spectrum[j-1] and spectrum[j] > spectrum[j+1]:
                    hull_indices = np.append(hull_indices, j)
                elif spectrum[j] < spectrum[j-1] and spectrum[j] < spectrum[j+1]:
                    hull_indices = np.append(hull_indices, j)
            
            hull_indices = np.sort(np.unique(hull_indices))
            
            # If not enough points, use all points
            if len(hull_indices) < 3:
                hull_indices = np.arange(n_points)
            
            # Interpolate continuum
            f_hull = interp1d(hull_indices, spectrum[hull_indices], 
                            kind='linear', fill_value='extrapolate')
            continuum = f_hull(np.arange(n_points))
            
            # Ensure continuum is above spectrum
            continuum = np.maximum(continuum, spectrum)
            
            # Calculate continuum-removed spectrum
            X_cr[i, :] = spectrum / continuum
        
        return X_cr
    
    
    def apply_smoothing(self, X: np.ndarray, window_length: int = 5,
                       polyorder: int = 2) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing to reduce noise.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data (n_samples, n_features)
        window_length : int, default=5
            Length of smoothing window (must be odd)
        polyorder : int, default=2
            Order of polynomial fit
            
        Returns
        -------
        np.ndarray
            Smoothed data
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if window_length % 2 == 0:
            window_length += 1  # Make odd
        
        X_smooth = np.zeros_like(X)
        for i in range(X.shape[0]):
            if X.shape[1] >= window_length:
                X_smooth[i, :] = savgol_filter(X[i, :], window_length, polyorder)
            else:
                X_smooth[i, :] = X[i, :]
        
        logger.info("Smoothing applied")
        return X_smooth
    
    
    @staticmethod
    def compute_derivatives(X: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Compute spectral derivatives.
        
        First derivative: dR/dλ (spectral gradient)
        Second derivative: d²R/dλ² (curvature)
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data (n_samples, n_wavelengths)
        order : int, default=1
            Order of derivative (1 or 2)
            
        Returns
        -------
        np.ndarray
            Derivatives (n_samples, n_wavelengths-order)
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        if order == 1:
            # First derivative
            derivatives = np.diff(X, axis=1)
        elif order == 2:
            # Second derivative
            derivatives = np.diff(np.diff(X, axis=1), axis=1)
        else:
            raise ValueError(f"Order must be 1 or 2, got {order}")
        
        logger.info(f"Computed {order} order derivatives. Shape: {derivatives.shape}")
        return derivatives
    
    
    @staticmethod
    def compute_moving_average(X: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        Compute moving average features.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data (n_samples, n_wavelengths)
        window_size : int, default=3
            Size of moving window
            
        Returns
        -------
        np.ndarray
            Moving averages (n_samples, n_wavelengths-window_size+1)
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_ma = np.zeros((X.shape[0], X.shape[1] - window_size + 1))
        
        for i in range(X_ma.shape[1]):
            X_ma[:, i] = np.mean(X[:, i:i+window_size], axis=1)
        
        logger.info(f"Computed moving averages (window={window_size}). Shape: {X_ma.shape}")
        return X_ma
    
    
    @staticmethod
    def compute_statistical_features(X: np.ndarray, window_size: int = 10) -> np.ndarray:
        """
        Compute statistical features from spectral data.
        
        Computes mean, std, variance, skewness, and kurtosis for sliding windows.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data (n_samples, n_wavelengths)
        window_size : int, default=10
            Size of sliding window for statistics
            
        Returns
        -------
        np.ndarray
            Statistical features (n_samples, n_wavelengths-window_size+1)*5
            Contains: mean, std, var, skewness, kurtosis
        """
        from scipy.stats import skew, kurtosis as scipy_kurtosis
        
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        n_samples = X.shape[0]
        n_windows = X.shape[1] - window_size + 1
        n_stats = 5  # mean, std, var, skew, kurtosis
        
        features = np.zeros((n_samples, n_windows * n_stats))
        
        for i in range(n_windows):
            window_data = X[:, i:i+window_size]
            
            # Compute statistics for this window
            features[:, i*n_stats + 0] = np.mean(window_data, axis=1)      # mean
            features[:, i*n_stats + 1] = np.std(window_data, axis=1)       # std
            features[:, i*n_stats + 2] = np.var(window_data, axis=1)       # variance
            features[:, i*n_stats + 3] = skew(window_data, axis=1)         # skewness
            features[:, i*n_stats + 4] = scipy_kurtosis(window_data, axis=1)  # kurtosis
        
        logger.info(f"Computed statistical features. Shape: {features.shape}")
        return features
    
    
    @staticmethod
    def compute_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Compute polynomial interaction features.
        
        Creates polynomial features (e.g., degree 2: x², xy, y²).
        This generates interaction terms between spectral bands.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data (n_samples, n_wavelengths)
        degree : int, default=2
            Polynomial degree
            
        Returns
        -------
        np.ndarray
            Polynomial features (n_samples, n_features*degree)
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        logger.info(f"Computed polynomial features (degree={degree}). Shape: {X_poly.shape}")
        return X_poly
    
    
    @staticmethod
    def compute_spectral_indices(X: np.ndarray) -> np.ndarray:
        """
        Compute spectral indices from reflectance data.
        
        Computes various spectral indices useful for soil property estimation:
        - Mean reflectance
        - Standard deviation
        - Reflectance slope
        - Spectral curvature
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input reflectance data (n_samples, n_wavelengths)
            
        Returns
        -------
        np.ndarray
            Spectral indices (n_samples, 4)
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        indices = np.zeros((X.shape[0], 4))
        
        # Mean reflectance
        indices[:, 0] = np.mean(X, axis=1)
        
        # Standard deviation (spectral variability)
        indices[:, 1] = np.std(X, axis=1)
        
        # Reflectance slope (mean of derivatives)
        derivatives = np.diff(X, axis=1)
        indices[:, 2] = np.mean(derivatives, axis=1)
        
        # Spectral curvature (mean of second derivatives)
        second_derivatives = np.diff(derivatives, axis=1)
        indices[:, 3] = np.mean(second_derivatives, axis=1)
        
        logger.info(f"Computed spectral indices. Shape: {indices.shape}")
        return indices
    
    
    @staticmethod
    def compute_pca_features(X: np.ndarray, n_components: int = 5) -> Tuple[np.ndarray, any]:
        """
        Compute PCA features (dimensionality reduction).
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data (n_samples, n_wavelengths)
        n_components : int, default=5
            Number of PCA components
            
        Returns
        -------
        tuple
            (X_pca, pca_model) - Transformed data and fitted PCA model
        """
        from sklearn.decomposition import PCA
        
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        n_comp = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        logger.info(f"Computed PCA features ({n_comp} components). "
                   f"Explained variance: {explained_var:.4f}. Shape: {X_pca.shape}")
        
        return X_pca, pca
    
    
    @staticmethod
    def compute_wavelet_features(X: np.ndarray, wavelet: str = 'db4') -> np.ndarray:
        """
        Compute wavelet transform features (multi-scale analysis).
        
        Uses discrete wavelet transform for multi-resolution analysis.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data (n_samples, n_wavelengths)
        wavelet : str, default='db4'
            Wavelet type ('db4', 'db8', 'sym5', etc.)
            
        Returns
        -------
        np.ndarray
            Wavelet features (n_samples, n_wavelengths)
        """
        try:
            import pywt  # type: ignore
        except ImportError:
            logger.warning("pywt (PyWavelets) not installed. Returning original data.")
            if isinstance(X, pd.DataFrame):
                return X.values
            return X
        
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_wavelet = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            try:
                # Compute discrete wavelet transform
                coeffs = pywt.dwt(X[i, :], wavelet)
                cA, cD = coeffs
                
                # Reconstruct to original length
                if len(cA) + len(cD) >= X.shape[1]:
                    X_wavelet[i, :len(cA)] = cA[:X.shape[1]]
                else:
                    X_wavelet[i, :len(cA)] = cA
                    X_wavelet[i, len(cA):len(cA)+len(cD)] = cD[:X.shape[1]-len(cA)]
            except Exception as e:
                logger.warning(f"Wavelet transform failed for sample {i}: {e}. Using original data.")
                X_wavelet[i, :] = X[i, :]
        
        logger.info(f"Computed wavelet features. Shape: {X_wavelet.shape}")
        return X_wavelet
    
    
    def apply_feature_engineering(self, X: np.ndarray, 
                                 derivatives: bool = False,
                                 statistical: bool = False,
                                 polynomial: bool = False,
                                 spectral_indices: bool = False,
                                 pca: bool = False,
                                 wavelet: bool = False,
                                 **kwargs) -> np.ndarray:
        """
        Apply selected feature engineering techniques to data.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data (n_samples, n_wavelengths)
        derivatives : bool, default=False
            Compute spectral derivatives
        statistical : bool, default=False
            Compute statistical features
        polynomial : bool, default=False
            Compute polynomial features
        spectral_indices : bool, default=False
            Compute spectral indices
        pca : bool, default=False
            Compute PCA features (dimensionality reduction)
        wavelet : bool, default=False
            Compute wavelet features (multi-scale analysis)
        **kwargs : dict
            Additional parameters for individual methods
            
        Returns
        -------
        np.ndarray
            Data with engineered features concatenated
        """
        # Convert pandas DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_engineered_parts = [X.copy()]
        
        if derivatives:
            deriv_order = kwargs.get('derivative_order', 1)
            deriv = self.compute_derivatives(X, order=deriv_order)
            X_engineered_parts.append(deriv)
            logger.info(f"Added {deriv_order}-order derivatives")
        
        if statistical:
            window_size = kwargs.get('stat_window', 10)
            stat_feat = self.compute_statistical_features(X, window_size=window_size)
            X_engineered_parts.append(stat_feat)
            logger.info("Added statistical features")
        
        if polynomial:
            poly_degree = kwargs.get('poly_degree', 2)
            poly_feat = self.compute_polynomial_features(X, degree=poly_degree)
            X_engineered_parts.append(poly_feat)
            logger.info(f"Added polynomial features (degree={poly_degree})")
        
        if spectral_indices:
            spec_idx = self.compute_spectral_indices(X)
            X_engineered_parts.append(spec_idx)
            logger.info("Added spectral indices")
        
        if pca:
            try:
                pca_feat, _ = self.compute_pca_features(X, n_components=5)
                X_engineered_parts.append(pca_feat)
                logger.info("Added PCA features (5 components)")
            except Exception as e:
                logger.warning(f"PCA computation failed: {e}. Skipping PCA features.")
        
        if wavelet:
            try:
                wavelet_type = kwargs.get('wavelet', 'db4')
                wav_feat = self.compute_wavelet_features(X, wavelet=wavelet_type)
                X_engineered_parts.append(wav_feat)
                logger.info(f"Added wavelet features ({wavelet_type})")
            except Exception as e:
                logger.warning(f"Wavelet computation failed: {e}. Skipping wavelet features.")
        
        # Concatenate all features
        if len(X_engineered_parts) > 1:
            X_final = np.hstack(X_engineered_parts)
            logger.info(f"Feature engineering complete. Final shape: {X_final.shape}")
            return X_final
        else:
            return X
    
    
    @staticmethod
    def remove_outliers(X: np.ndarray, y: np.ndarray = None,
                       method: str = 'iqr', threshold: float = 1.5) -> Tuple:
        """
        Remove outlier samples.
        
        Parameters
        ----------
        X : np.ndarray
            Feature data (n_samples, n_features)
        y : np.ndarray, optional
            Target data
        method : str, default='iqr'
            Outlier detection method:
            - 'iqr': Interquartile range method
            - 'zscore': Z-score method
        threshold : float, default=1.5
            Threshold for IQR method (1.5 for outliers, 3.0 for extreme)
            
        Returns
        -------
        tuple
            (X_clean, y_clean, mask) - Clean data and boolean mask
        """
        if method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
            
        elif method == 'zscore':
            z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
            mask = np.all(z_scores < threshold, axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_clean = X[mask]
        y_clean = y[mask] if y is not None else None
        
        removed_count = len(X) - len(X_clean)
        logger.info(f"Removed {removed_count} outliers using {method} method")
        
        return X_clean, y_clean, mask
    
    
    @staticmethod
    def handle_missing_values(X: np.ndarray, y: np.ndarray = None,
                             strategy: str = 'drop') -> Tuple:
        """
        Handle missing values.
        
        Parameters
        ----------
        X : np.ndarray
            Feature data
        y : np.ndarray, optional
            Target data
        strategy : str, default='drop'
            Strategy: 'drop', 'mean', 'interpolate'
            
        Returns
        -------
        tuple
            (X_clean, y_clean)
        """
        X_clean = X.copy()
        y_clean = y.copy() if y is not None else None
        
        if strategy == 'drop':
            mask = ~np.isnan(X_clean).any(axis=1)
            X_clean = X_clean[mask]
            if y is not None:
                y_clean = y_clean[mask]
            logger.info(f"Dropped {len(X) - len(X_clean)} rows with NaN")
            
        elif strategy == 'mean':
            col_means = np.nanmean(X_clean, axis=0)
            for i in range(X_clean.shape[1]):
                mask = np.isnan(X_clean[:, i])
                X_clean[mask, i] = col_means[i]
            logger.info("Filled NaN with column means")
            
        elif strategy == 'interpolate':
            for i in range(X_clean.shape[1]):
                mask = ~np.isnan(X_clean[:, i])
                if np.sum(mask) > 1:
                    X_clean[:, i] = np.interp(
                        np.arange(len(X_clean)),
                        np.where(mask)[0],
                        X_clean[mask, i]
                    )
            logger.info("Interpolated NaN values")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return X_clean, y_clean
    
    
    def get_config(self) -> Dict:
        """
        Get configuration of fitted preprocessor.
        
        Returns
        -------
        dict
            Configuration including technique and scaler type
        """
        config = {
            'technique': self.technique,
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted,
            'normalizer_type': type(self.normalizer).__name__
        }
        return config
    
    
    def __repr__(self) -> str:
        """String representation."""
        if self.is_fitted:
            return (f"SpectralPreprocessor(technique='{self.technique}', "
                   f"scaler='{self.scaler_type}', is_fitted=True)")
        else:
            return "SpectralPreprocessor(is_fitted=False)"


# Convenience functions
def preprocess_spectral_data(X_train: np.ndarray, X_test: np.ndarray,
                            technique: str = 'reflectance',
                            scaler: str = 'standard') -> Tuple:
    """
    Convenience function to preprocess train and test data.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Testing features
    technique : str, default='reflectance'
        Spectral technique
    scaler : str, default='standard'
        Normalization method
        
    Returns
    -------
    tuple
        (X_train_prep, X_test_prep, preprocessor)
    """
    preprocessor = SpectralPreprocessor()
    preprocessor.fit(X_train, technique=technique, scaler=scaler)
    
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    return X_train_prep, X_test_prep, preprocessor


def apply_all_techniques(X_train: np.ndarray, X_test: np.ndarray,
                         scaler: str = 'standard') -> Dict:
    """
    Apply all three spectral techniques and return results.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features
    X_test : np.ndarray
        Testing features
    scaler : str, default='standard'
        Normalization method
        
    Returns
    -------
    dict
        Dictionary with keys: 'reflectance', 'absorbance', 'continuum_removal'
        Each containing: {'X_train': ..., 'X_test': ..., 'preprocessor': ...}
    """
    results = {}
    techniques = ['reflectance', 'absorbance', 'continuum_removal']
    
    for technique in techniques:
        X_train_prep, X_test_prep, prep = preprocess_spectral_data(
            X_train, X_test, technique=technique, scaler=scaler
        )
        results[technique] = {
            'X_train': X_train_prep,
            'X_test': X_test_prep,
            'preprocessor': prep
        }
        logger.info(f"Applied {technique} technique")
    
    return results
