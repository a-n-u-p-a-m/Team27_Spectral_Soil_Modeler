"""
Data Loader Module
==================
This module handles loading, validating, and preprocessing spectral soil data.

Functions:
    - load_data(): Load data from XLS/CSV files
    - validate_data(): Check data integrity
    - get_data_stats(): Compute summary statistics
    - split_train_test(): Split data into training and testing sets
    - get_feature_names(): Extract feature/column names
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class to load, validate, and manipulate spectral soil data.
    
    Attributes:
        data (pd.DataFrame): Loaded dataset
        target_col (str): Target column name
        features (list): List of feature column names
        n_samples (int): Number of samples
        n_features (int): Number of features
    """
    
    def __init__(self):
        """Initialize DataLoader."""
        self.data = None
        self.target_col = None
        self.features = None
        self.n_samples = 0
        self.n_features = 0
        self.original_data = None
        
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load spectral data from XLS, XLSX, or CSV ASCII files.
        
        Supports multiple file formats and delimiters:
        - Excel files (.xls, .xlsx)
        - CSV files (.csv)
        - ASCII CSV files with .xls extension (auto-detects delimiter)
        
        Auto-detects delimiters in order:
        1. Comma (,)
        2. Whitespace/Tab (\\s+)
        3. Semicolon (;)
        
        Parameters
        ----------
        filepath : str
            Path to the data file
            
        Returns
        -------
        pd.DataFrame
            Loaded dataframe
            
        Raises
        ------
        FileNotFoundError
            If file does not exist
        ValueError
            If file format is not supported or cannot be parsed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            if filepath.suffix.lower() in ['.xls', '.xlsx']:
                # Try to load as Excel first
                try:
                    self.data = pd.read_excel(filepath)
                    logger.info(f"Loaded as Excel file: {filepath}")
                except Exception as excel_error:
                    # If Excel loading fails, try as CSV ASCII (some .xls files are actually CSV)
                    logger.warning(f"Excel loading failed, trying as CSV: {str(excel_error)}")
                    self.data = self._load_csv_with_auto_delimiter(filepath)
            elif filepath.suffix.lower() == '.csv':
                self.data = self._load_csv_with_auto_delimiter(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            # Store original data for reference
            self.original_data = self.data.copy()
            
            # Update dimensions
            self.n_samples, self.n_features = self.data.shape
            logger.info(f"Data shape: {self.n_samples} samples × {self.n_features} features")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    
    def _load_csv_with_auto_delimiter(self, filepath: Path) -> pd.DataFrame:
        r"""
        Load CSV file with automatic delimiter detection.
        
        Tries delimiters in this order:
        1. Comma (,) - standard CSV
        2. Whitespace (\s+) - space/tab separated
        3. Semicolon (;) - European CSV
        
        Parameters
        ----------
        filepath : Path
            Path to CSV file
            
        Returns
        -------
        pd.DataFrame
            Loaded dataframe
            
        Raises
        ------
        ValueError
            If no delimiter works
        """
        delimiters = [
            (',', 'comma'),
            (r'\s+', 'whitespace/tab'),
            (';', 'semicolon')
        ]
        
        last_error = None
        
        for delimiter, delimiter_name in delimiters:
            try:
                df = pd.read_csv(filepath, delimiter=delimiter, engine='python')
                
                # Validate that we got reasonable data
                if df.shape[0] > 0 and df.shape[1] > 1:
                    logger.info(f"Loaded CSV with {delimiter_name} delimiter: {filepath}")
                    logger.info(f"  Delimiter: '{delimiter}' | Shape: {df.shape}")
                    return df
            except Exception as e:
                last_error = e
                logger.debug(f"Failed to load with {delimiter_name} delimiter: {str(e)}")
                continue
        
        # If all delimiters failed, raise error
        raise ValueError(
            f"Could not parse CSV file {filepath} with any standard delimiter. "
            f"Last error: {str(last_error)}"
        )
    
    
    def set_target_column(self, target_col: str) -> None:
        """
        Set the target column for prediction.
        
        Parameters
        ----------
        target_col : str
            Name of the target column
            
        Raises
        ------
        ValueError
            If target column not in data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")
        
        self.target_col = target_col
        
        # Features are all columns except target
        self.features = [col for col in self.data.columns if col != target_col]
        
        logger.info(f"Target column set to: {target_col}")
        logger.info(f"Number of features: {len(self.features)}")
    
    
    def validate_data(self) -> Dict:
        """
        Validate data integrity and return report.
        
        Returns
        -------
        dict
            Validation report with keys:
            - 'is_valid' (bool): True if data passes validation
            - 'shape' (tuple): Data shape
            - 'dtypes' (dict): Data types of columns
            - 'missing_values' (dict): Count of missing values per column
            - 'missing_percentage' (dict): Percentage of missing values
            - 'duplicates' (int): Number of duplicate rows
            - 'issues' (list): List of validation issues found
        """
        report = {
            'is_valid': True,
            'shape': self.data.shape,
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'issues': []
        }
        
        # Check for missing values
        missing_cols = [col for col, count in report['missing_values'].items() if count > 0]
        if missing_cols:
            report['is_valid'] = False
            report['issues'].append(f"Missing values in columns: {missing_cols}")
            logger.warning(f"Missing values detected: {missing_cols}")
        
        # Check for duplicates
        if report['duplicates'] > 0:
            report['is_valid'] = False
            report['issues'].append(f"Found {report['duplicates']} duplicate rows")
            logger.warning(f"Duplicate rows detected: {report['duplicates']}")
        
        # Check if target column is numeric
        if self.target_col and not pd.api.types.is_numeric_dtype(self.data[self.target_col]):
            report['is_valid'] = False
            report['issues'].append(f"Target column '{self.target_col}' is not numeric")
            logger.warning(f"Target column is not numeric")
        
        # Check if features are numeric
        non_numeric_features = [col for col in self.features 
                               if not pd.api.types.is_numeric_dtype(self.data[col])]
        if non_numeric_features:
            report['is_valid'] = False
            report['issues'].append(f"Non-numeric features: {non_numeric_features}")
            logger.warning(f"Non-numeric features detected: {non_numeric_features}")
        
        if report['is_valid']:
            logger.info("Data validation passed!")
        else:
            logger.warning("Data validation failed!")
        
        return report
    
    
    def get_data_stats(self) -> pd.DataFrame:
        """
        Get summary statistics of the data.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics (mean, std, min, 25%, 50%, 75%, max)
        """
        stats = self.data.describe()
        logger.info("Computed data statistics")
        return stats
    
    
    def get_feature_stats(self) -> pd.DataFrame:
        """
        Get summary statistics of features only (excluding target).
        
        Returns
        -------
        pd.DataFrame
            Summary statistics for features
        """
        if self.features is None:
            raise ValueError("Target column not set. Call set_target_column() first.")
        
        stats = self.data[self.features].describe()
        logger.info("Computed feature statistics")
        return stats
    
    
    def get_target_stats(self) -> Dict:
        """
        Get summary statistics of the target variable.
        
        Returns
        -------
        dict
            Statistics including mean, std, min, max, median, quartiles
        """
        if self.target_col is None:
            raise ValueError("Target column not set. Call set_target_column() first.")
        
        target_data = self.data[self.target_col]
        stats = {
            'mean': target_data.mean(),
            'std': target_data.std(),
            'min': target_data.min(),
            'max': target_data.max(),
            'median': target_data.median(),
            'q25': target_data.quantile(0.25),
            'q75': target_data.quantile(0.75),
            'count': len(target_data)
        }
        logger.info(f"Computed target statistics: {self.target_col}")
        return stats
    
    
    def split_train_test(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Parameters
        ----------
        test_size : float, default=0.2
            Proportion of data to use for testing (0.0 to 1.0)
        random_state : int, default=42
            Random seed for reproducibility
            
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
            Where X contains features and y contains target
        """
        if self.features is None:
            raise ValueError("Target column not set. Call set_target_column() first.")
        
        from sklearn.model_selection import train_test_split
        
        X = self.data[self.features]
        y = self.data[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Data split - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        logger.info(f"Train/Test ratio: {len(X_train)/(len(X_train)+len(X_test)):.1%} / {test_size:.1%}")
        
        return X_train, X_test, y_train, y_test
    
    
    def get_column_names(self) -> list:
        """
        Get all column names in the dataset.
        
        Returns
        -------
        list
            List of column names
        """
        return self.data.columns.tolist()
    
    
    def get_feature_names(self) -> list:
        """
        Get feature column names (excluding target).
        
        Returns
        -------
        list
            List of feature names
        """
        if self.features is None:
            raise ValueError("Target column not set. Call set_target_column() first.")
        return self.features
    
    
    def get_numeric_columns(self) -> list:
        """
        Get names of all numeric columns.
        
        Returns
        -------
        list
            List of numeric column names
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_cols
    
    
    def get_data_info(self) -> Dict:
        """
        Get comprehensive information about the dataset.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'shape': (n_samples, n_features)
            - 'columns': list of column names
            - 'dtypes': data types
            - 'target': target column name
            - 'n_features': number of features
            - 'n_samples': number of samples
        """
        info = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'target': self.target_col,
            'n_features': len(self.features) if self.features else 0,
            'n_samples': self.n_samples,
            'features': self.features if self.features else []
        }
        return info
    
    
    def remove_missing_values(self, strategy: str = 'drop') -> None:
        """
        Handle missing values in the dataset.
        
        Parameters
        ----------
        strategy : str, default='drop'
            Strategy to handle missing values:
            - 'drop': Remove rows with missing values
            - 'mean': Fill with mean (numeric columns only)
            - 'median': Fill with median (numeric columns only)
            
        Raises
        ------
        ValueError
            If strategy is not recognized
        """
        if strategy == 'drop':
            initial_rows = len(self.data)
            self.data = self.data.dropna()
            removed_rows = initial_rows - len(self.data)
            logger.info(f"Removed {removed_rows} rows with missing values")
            
        elif strategy == 'mean':
            self.data = self.data.fillna(self.data.mean(numeric_only=True))
            logger.info("Filled missing values with mean")
            
        elif strategy == 'median':
            self.data = self.data.fillna(self.data.median(numeric_only=True))
            logger.info("Filled missing values with median")
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.n_samples, self.n_features = self.data.shape
    
    
    def remove_duplicates(self) -> None:
        """Remove duplicate rows from the dataset."""
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        removed_rows = initial_rows - len(self.data)
        logger.info(f"Removed {removed_rows} duplicate rows")
        self.n_samples, self.n_features = self.data.shape
    
    
    def get_X_y(self) -> Tuple:
        """
        Get feature matrix X and target vector y.
        
        Returns
        -------
        tuple
            (X, y) where X is features and y is target
        """
        if self.features is None:
            raise ValueError("Target column not set. Call set_target_column() first.")
        
        X = self.data[self.features].values
        y = self.data[self.target_col].values
        
        return X, y


# Standalone functions for convenience
def load_data(filepath: str) -> pd.DataFrame:
    """
    Convenience function to load data.
    
    Parameters
    ----------
    filepath : str
        Path to data file
        
    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    loader = DataLoader()
    return loader.load_data(filepath)


def validate_data(data: pd.DataFrame, target_col: str) -> Dict:
    """
    Convenience function to validate data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to validate
    target_col : str
        Target column name
        
    Returns
    -------
    dict
        Validation report
    """
    loader = DataLoader()
    loader.data = data
    loader.set_target_column(target_col)
    return loader.validate_data()
