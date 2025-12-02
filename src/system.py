"""
System Module - Consolidated Backend Utilities
===============================================
Handles all backend utilities: logging, persistence, and exporting.

Consolidates:
- logger.py (SystemLogger, PerformanceTracker)
- persistence.py (ModelPersistence)
- export_manager.py (ResultsExporter, StreamlitExporter)
"""

import logging
import logging.handlers
import joblib
import json
import pickle
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import time
import numpy as np
import pandas as pd

# ============================================================================
# LOGGER MODULE
# ============================================================================

class SystemLogger:
    """
    Comprehensive system logging.
    
    Handles:
    - Component logging (data_loader, preprocessing, etc.)
    - Performance tracking
    - Error logging
    - Event timestamping
    - Log file management
    
    Attributes:
        log_dir: Directory for log files
        logger: Main logger instance
        event_log: Structured event tracking
    """
    
    def __init__(self, log_dir: str = "./logs", name: str = "SoilModeler"):
        """
        Initialize SystemLogger.
        
        Parameters
        ----------
        log_dir : str
            Directory for log files
        name : str
            Logger name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.name = name
        self.logger = self._setup_logger()
        self.event_log = []
        
        self.log_info(f"SystemLogger initialized: {log_dir}")
    
    
    def _setup_logger(self) -> logging.Logger:
        """Setup main logger with file and console handlers."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler (rotating)
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    
    def log_debug(self, message: str, component: str = None) -> None:
        """Log debug message."""
        msg = f"[{component}] {message}" if component else message
        self.logger.debug(msg)
    
    
    def log_info(self, message: str, component: str = None) -> None:
        """Log info message."""
        msg = f"[{component}] {message}" if component else message
        self.logger.info(msg)
    
    
    def log_warning(self, message: str, component: str = None) -> None:
        """Log warning message."""
        msg = f"[{component}] {message}" if component else message
        self.logger.warning(msg)
    
    
    def log_error(self, message: str, component: str = None, exception: Exception = None) -> None:
        """Log error message."""
        msg = f"[{component}] {message}" if component else message
        if exception:
            self.logger.error(msg, exc_info=exception)
        else:
            self.logger.error(msg)
    
    
    def log_critical(self, message: str, component: str = None) -> None:
        """Log critical message."""
        msg = f"[{component}] {message}" if component else message
        self.logger.critical(msg)
    
    
    # Proxy methods for standard logging interface
    def debug(self, message: str) -> None:
        """Log debug message (standard interface)."""
        self.logger.debug(message)
    
    
    def info(self, message: str) -> None:
        """Log info message (standard interface)."""
        self.logger.info(message)
    
    
    def warning(self, message: str) -> None:
        """Log warning message (standard interface)."""
        self.logger.warning(message)
    
    
    def error(self, message: str, exc_info: Exception = None) -> None:
        """Log error message (standard interface)."""
        if exc_info:
            self.logger.error(message, exc_info=exc_info)
        else:
            self.logger.error(message)
    
    
    def critical(self, message: str) -> None:
        """Log critical message (standard interface)."""
        self.logger.critical(message)
    
    
    def log_performance(self, operation: str, duration: float,
                       component: str = None, status: str = "success") -> None:
        """Log performance metric."""
        msg = f"{component}/{operation}" if component else operation
        self.logger.info(f"{msg} completed in {duration:.2f}s [{status}]")
    
    
    def log_event(self, event_type: str, data: Dict, component: str = None) -> None:
        """Log structured event."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'component': component or 'unknown',
            'data': data
        }
        self.event_log.append(event)
        self.logger.info(f"EVENT: {event_type} - {data}")
    
    
    def log_data_loading(self, filename: str, n_samples: int, n_features: int,
                        status: str = "success") -> None:
        """Log data loading event."""
        self.log_event(
            'data_loading',
            {
                'filename': filename,
                'n_samples': n_samples,
                'n_features': n_features,
                'status': status
            },
            component='data_loader'
        )
    
    
    def log_preprocessing(self, technique: str, n_samples: int,
                         scalers: list = None, status: str = "success") -> None:
        """Log preprocessing event."""
        self.log_event(
            'preprocessing',
            {
                'technique': technique,
                'n_samples': n_samples,
                'scalers': scalers or [],
                'status': status
            },
            component='preprocessing'
        )
    
    
    def log_model_training(self, model_name: str, technique: str,
                          n_samples: int, n_features: int,
                          status: str = "success") -> None:
        """Log model training event."""
        self.log_event(
            'model_training',
            {
                'model': model_name,
                'technique': technique,
                'n_samples': n_samples,
                'n_features': n_features,
                'status': status
            },
            component='model_trainer'
        )
    
    
    def log_model_evaluation(self, model_name: str, technique: str,
                            metrics: Dict, status: str = "success") -> None:
        """Log model evaluation event."""
        self.log_event(
            'model_evaluation',
            {
                'model': model_name,
                'technique': technique,
                'metrics': metrics,
                'status': status
            },
            component='evaluator'
        )
    
    
    def log_persistence(self, operation: str, filepath: str,
                       status: str = "success") -> None:
        """Log persistence event."""
        self.log_event(
            'persistence',
            {
                'operation': operation,
                'filepath': filepath,
                'status': status
            },
            component='persistence'
        )
    
    
    def get_statistics(self) -> Dict:
        """Get logging statistics."""
        stats = {
            'total_events': len(self.event_log),
            'event_types': {},
            'components': {}
        }
        
        for event in self.event_log:
            event_type = event['type']
            component = event['component']
            
            if event_type not in stats['event_types']:
                stats['event_types'][event_type] = 0
            stats['event_types'][event_type] += 1
            
            if component not in stats['components']:
                stats['components'][component] = 0
            stats['components'][component] += 1
        
        return stats
    
    
    def save_event_log(self, filename: str = "event_log.json") -> str:
        """Save event log to JSON file."""
        filepath = self.log_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.event_log, f, indent=4)
            
            self.log_info(f"Event log saved: {filepath}")
            return str(filepath)
        except Exception as e:
            self.log_error(f"Error saving event log: {str(e)}", exception=e)
            raise
    
    
    def load_event_log(self, filename: str = "event_log.json") -> list:
        """Load event log from JSON file."""
        filepath = self.log_dir / filename
        
        try:
            with open(filepath, 'r') as f:
                events = json.load(f)
            
            self.log_info(f"Event log loaded: {filepath}")
            return events
        except Exception as e:
            self.log_error(f"Error loading event log: {str(e)}", exception=e)
            raise
    
    
    def print_summary(self) -> None:
        """Print summary of all logged events."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("SYSTEM LOG SUMMARY")
        print("="*60)
        print(f"\nTotal Events: {stats['total_events']}")
        
        print("\nEvents by Type:")
        for event_type, count in sorted(stats['event_types'].items()):
            print(f"  - {event_type}: {count}")
        
        print("\nEvents by Component:")
        for component, count in sorted(stats['components'].items()):
            print(f"  - {component}: {count}")
        
        print("\n" + "="*60)


class PerformanceTracker:
    """Track and log performance metrics."""
    
    def __init__(self, logger: SystemLogger):
        """Initialize PerformanceTracker."""
        self.logger = logger
        self.timers = {}
        self.metrics = {}
    
    
    def start_timer(self, operation: str) -> None:
        """Start a performance timer."""
        self.timers[operation] = time.time()
        self.logger.log_debug(f"Timer started: {operation}")
    
    
    def end_timer(self, operation: str, component: str = None) -> float:
        """End a performance timer and return elapsed time."""
        if operation not in self.timers:
            self.logger.log_warning(f"No timer for operation: {operation}")
            return None
        
        elapsed = time.time() - self.timers.pop(operation)
        
        self.logger.log_performance(operation, elapsed, component)
        
        # Store metric
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(elapsed)
        
        return elapsed
    
    
    def get_metrics(self, operation: str = None) -> Dict:
        """Get performance metrics."""
        if operation:
            times = self.metrics.get(operation, [])
            if not times:
                return None
            
            return {
                'operation': operation,
                'count': len(times),
                'min': min(times),
                'max': max(times),
                'mean': sum(times) / len(times),
                'total': sum(times)
            }
        else:
            # Return all metrics
            result = {}
            for op in self.metrics:
                result[op] = self.get_metrics(op)
            return result
    
    
    def print_performance_report(self) -> None:
        """Print performance report."""
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        for operation, metric in self.get_metrics().items():
            if metric:
                print(f"\n{operation}:")
                print(f"  Count: {metric['count']}")
                print(f"  Total: {metric['total']:.2f}s")
                print(f"  Mean: {metric['mean']:.2f}s")
                print(f"  Min: {metric['min']:.2f}s")
                print(f"  Max: {metric['max']:.2f}s")
        
        print("\n" + "="*60)


# ============================================================================
# PERSISTENCE MODULE
# ============================================================================

class ModelPersistence:
    """
    Handle model serialization and persistence.
    
    Saves:
    - Trained model objects (joblib)
    - Metadata (hyperparameters, metrics, timestamp)
    - Configuration (preprocessing settings)
    - Results (predictions, metrics)
    
    Attributes:
        model_dir: Directory for storing models
        metadata_dir: Directory for metadata
    """
    
    def __init__(self, model_dir: str = "./models", metadata_dir: str = "./metadata"):
        """Initialize ModelPersistence."""
        self.model_dir = Path(model_dir)
        self.metadata_dir = Path(metadata_dir)
        
        # Create directories if they don't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    
    def save_model(self, model: Any, model_name: str, model_type: str,
                  technique: str, metadata: Dict = None) -> str:
        """Save trained model to disk."""
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{technique}_{timestamp}.joblib"
        filepath = self.model_dir / filename
        
        try:
            # Save model
            joblib.dump(model, filepath)
            
            # Save metadata
            metadata_dict = {
                'model_name': model_name,
                'model_type': model_type,
                'technique': technique,
                'timestamp': timestamp,
                'filepath': str(filepath)
            }
            
            if metadata:
                metadata_dict.update(metadata)
            
            self._save_metadata(metadata_dict, filename)
            
            return str(filepath)
            
        except Exception as e:
            raise
    
    
    def load_model(self, filepath: str) -> Any:
        """Load trained model from disk."""
        try:
            model = joblib.load(filepath)
            return model
        except Exception as e:
            raise
    
    
    def save_best_model(self, trainer, model_trainer_obj, y_test: np.ndarray,
                       target_column: str = None) -> Dict:
        """Save best model from training results."""
        best_result, best_model = trainer.get_best_model()
        
        # Prepare metadata
        metadata = {
            'algorithm': best_result['Model'],
            'technique': best_result['Technique'],
            'train_r2': float(best_result['Train_R²']),
            'test_r2': float(best_result['Test_R²']),
            'test_rmse': float(best_result['Test_RMSE']),
            'test_mae': float(best_result['Test_MAE']),
            'rpd': float(best_result['RPD']),
            'target_column': target_column or 'unknown',
            'n_test_samples': len(y_test),
            'y_test_mean': float(np.mean(y_test)),
            'y_test_std': float(np.std(y_test))
        }
        
        # Save model
        filepath = self.save_model(
            best_model,
            model_name=best_result['Model'],
            model_type='regression',
            technique=best_result['Technique'],
            metadata=metadata
        )
        
        return {
            'filepath': filepath,
            'metadata': metadata
        }
    
    
    def _save_metadata(self, metadata: Dict, model_filename: str) -> None:
        """Save metadata JSON file."""
        metadata_filename = model_filename.replace('.joblib', '_metadata.json')
        metadata_path = self.metadata_dir / metadata_filename
        
        try:
            metadata_serializable = self._make_serializable(metadata)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_serializable, f, indent=4)
        except Exception as e:
            raise
    
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy and other types to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    
    def load_metadata(self, metadata_filename: str) -> Dict:
        """Load metadata JSON file."""
        metadata_path = self.metadata_dir / metadata_filename
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
        except Exception as e:
            raise
    
    
    def save_results(self, results_df: pd.DataFrame, filename: str) -> str:
        """Save results DataFrame to CSV."""
        filepath = self.metadata_dir / filename
        
        try:
            results_df.to_csv(filepath, index=False)
            return str(filepath)
        except Exception as e:
            raise
    
    
    def save_predictions(self, predictions: np.ndarray, filename: str,
                        y_true: np.ndarray = None) -> str:
        """Save predictions to CSV."""
        filepath = self.metadata_dir / filename
        
        try:
            data = {'predictions': predictions}
            if y_true is not None:
                data['actual'] = y_true
                data['residual'] = y_true - predictions
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            return str(filepath)
        except Exception as e:
            raise
    
    
    def save_configuration(self, config: Dict, filename: str = "config.json") -> str:
        """Save experiment configuration."""
        filepath = self.metadata_dir / filename
        
        try:
            config_serializable = self._make_serializable(config)
            
            with open(filepath, 'w') as f:
                json.dump(config_serializable, f, indent=4)
            
            return str(filepath)
        except Exception as e:
            raise
    
    
    def list_saved_models(self) -> list:
        """List all saved models."""
        model_files = list(self.model_dir.glob("*.joblib"))
        return sorted([str(f) for f in model_files])
    
    
    def list_saved_metadata(self) -> list:
        """List all saved metadata files."""
        metadata_files = list(self.metadata_dir.glob("*_metadata.json"))
        return sorted([str(f) for f in metadata_files])
    
    
    def delete_model(self, filepath: str) -> None:
        """Delete a saved model."""
        try:
            model_path = Path(filepath)
            model_path.unlink()
            
            # Also delete metadata
            metadata_path = self.metadata_dir / model_path.name.replace('.joblib', '_metadata.json')
            if metadata_path.exists():
                metadata_path.unlink()
        except Exception as e:
            raise
    
    
    def get_model_info(self, model_filename: str) -> Dict:
        """Get information about a saved model."""
        metadata_filename = model_filename.replace('.joblib', '_metadata.json')
        
        try:
            metadata = self.load_metadata(metadata_filename)
            
            model_path = self.model_dir / model_filename
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            
            info = {
                'filename': model_filename,
                'file_size_mb': round(file_size, 2),
                **metadata
            }
            
            return info
        except Exception as e:
            return None
    
    
    def export_best_model_summary(self, trainer, y_test: np.ndarray,
                                 output_file: str = "best_model_summary.json") -> str:
        """Export comprehensive summary of best model."""
        best_result, best_model = trainer.get_best_model()
        
        summary = {
            'best_model': best_result['Model'],
            'best_technique': best_result['Technique'],
            'performance': {
                'train_r2': float(best_result['Train_R²']),
                'test_r2': float(best_result['Test_R²']),
                'train_rmse': float(best_result['Train_RMSE']),
                'test_rmse': float(best_result['Test_RMSE']),
                'train_mae': float(best_result['Train_MAE']),
                'test_mae': float(best_result['Test_MAE']),
                'rpd': float(best_result['RPD'])
            },
            'test_set_statistics': {
                'n_samples': int(len(y_test)),
                'mean': float(np.mean(y_test)),
                'std': float(np.std(y_test)),
                'min': float(np.min(y_test)),
                'max': float(np.max(y_test))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = self.metadata_dir / output_file
        
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=4)
            
            return str(filepath)
        except Exception as e:
            raise


# ============================================================================
# EXPORT MODULE
# ============================================================================

class ResultsExporter:
    """Export results in multiple formats."""
    
    def __init__(self, export_dir: str = "./exports"):
        """Initialize exporter."""
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
    def export_to_csv(self, data: pd.DataFrame, filename: str = None) -> str:
        """Export data to CSV."""
        if filename is None:
            filename = f"results_{self.timestamp}.csv"
        
        filepath = self.export_dir / filename
        data.to_csv(filepath, index=False)
        
        return str(filepath)
    
    
    def export_to_excel(self, data_dict: Dict[str, pd.DataFrame],
                       filename: str = None) -> str:
        """Export multiple dataframes to Excel with separate sheets."""
        if filename is None:
            filename = f"results_{self.timestamp}.xlsx"
        
        filepath = self.export_dir / filename
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                for sheet_name, df in data_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Format the worksheet
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(cell.value)
                            except:
                                pass
                        
                        adjusted_width = (max_length + 2)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            return str(filepath)
        
        except ImportError:
            raise
    
    
    def export_to_json(self, data: Dict[str, Any], filename: str = None) -> str:
        """Export data to JSON with metadata."""
        if filename is None:
            filename = f"results_{self.timestamp}.json"
        
        filepath = self.export_dir / filename
        
        # Add metadata
        export_data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'data': data
        }
        
        # Convert DataFrames to dictionaries if present
        for key, value in export_data['data'].items():
            if isinstance(value, pd.DataFrame):
                export_data['data'][key] = value.to_dict(orient='records')
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return str(filepath)
    
    
    def create_summary_report(self, results_df: pd.DataFrame,
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a summary report from results."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(results_df),
            'summary_statistics': {
                'mean_r2': float(results_df['Test_R²'].mean()),
                'std_r2': float(results_df['Test_R²'].std()),
                'min_r2': float(results_df['Test_R²'].min()),
                'max_r2': float(results_df['Test_R²'].max()),
            },
            'best_model': {
                'model': results_df.loc[results_df['Test_R²'].idxmax(), 'Model'],
                'technique': results_df.loc[results_df['Test_R²'].idxmax(), 'Technique'],
                'r2_score': float(results_df['Test_R²'].max()),
            }
        }
        
        if metadata:
            report['metadata'] = metadata
        
        return report
    
    
    def export_comprehensive_excel(self, results_df: pd.DataFrame,
                                  paradigm: str = "Standard",
                                  filename: str = None) -> str:
        """
        Export comprehensive results with feature importance and predictions to Excel.
        
        Includes:
        - Model Results: Basic metrics (R², RMSE, MAE, RPD, hyperparameters)
        - Feature Importance: Top features by importance method
        - Predictions: Test predictions and residuals
        - Feature Engineering: Config and generated features
        - Summary: Overall statistics
        """
        import numpy as np
        
        if filename is None:
            filename = f"results_comprehensive_{paradigm}_{self.timestamp}.xlsx"
        
        filepath = self.export_dir / filename
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Sheet 1: Main Results
                results_export = results_df[[
                    'Model', 'Technique', 'Train_R²', 'Test_R²', 'Train_RMSE', 
                    'Test_RMSE', 'Train_MAE', 'Test_MAE', 'RPD'
                ]].copy()
                
                # Add hyperparameters as string
                results_export['Hyperparameters'] = results_df['Hyperparameters'].astype(str)
                results_export.to_excel(writer, sheet_name='Results', index=False)
                
                # Sheet 2: Feature Importance (All Methods)
                importance_list = []
                for idx, row in results_df.iterrows():
                    importance_data = row.get('Feature_Importance')
                    if importance_data and isinstance(importance_data, dict):
                        model_name = row['Model']
                        technique = row['Technique']
                        
                        # Extract all methods
                        all_methods = importance_data.get('all_methods', {})
                        top_features = importance_data.get('top_features', [])
                        
                        for feature_info in top_features:
                            feature_row = {
                                'Model': model_name,
                                'Technique': technique,
                                'Feature_Name': feature_info.get('name', 'Unknown'),
                                'Feature_Index': feature_info.get('index', -1),
                                'Importance_Value': feature_info.get('importance', 0)
                            }
                            
                            # Add values from each method
                            for method_name, method_importances in all_methods.items():
                                if isinstance(method_importances, np.ndarray) and feature_info['index'] < len(method_importances):
                                    feature_row[f'Importance_{method_name}'] = float(method_importances[feature_info['index']])
                            
                            importance_list.append(feature_row)
                
                if importance_list:
                    importance_df = pd.DataFrame(importance_list)
                    importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
                
                # Sheet 3: Predictions
                predictions_list = []
                for idx, row in results_df.iterrows():
                    predictions = row.get('Predictions')
                    y_test = row.get('y_Test')
                    
                    if predictions is not None and y_test is not None:
                        pred_array = predictions if isinstance(predictions, np.ndarray) else np.array(predictions)
                        y_test_array = y_test if isinstance(y_test, np.ndarray) else np.array(y_test)
                        
                        residuals = y_test_array - pred_array
                        
                        pred_df = pd.DataFrame({
                            'Model': row['Model'],
                            'Technique': row['Technique'],
                            'Actual': y_test_array,
                            'Predicted': pred_array,
                            'Residual': residuals,
                            'Abs_Error': np.abs(residuals)
                        })
                        predictions_list.append(pred_df)
                
                if predictions_list:
                    all_predictions = pd.concat(predictions_list, ignore_index=True)
                    all_predictions.to_excel(writer, sheet_name='Predictions', index=False)
                
                # Sheet 4: Feature Engineering Configuration
                fe_config_list = []
                for idx, row in results_df.iterrows():
                    fe_config = row.get('FE_Config', {})
                    X_test = row.get('X_Test')
                    
                    fe_row = {
                        'Model': row['Model'],
                        'Technique': row['Technique'],
                        'Use_Derivatives': fe_config.get('derivatives', False),
                        'Use_Statistical': fe_config.get('statistical', False),
                        'Use_Polynomial': fe_config.get('polynomial', False),
                        'Use_Spectral_Indices': fe_config.get('spectral_indices', False),
                        'Use_PCA': fe_config.get('pca', False),
                        'Use_Wavelet': fe_config.get('wavelet', False),
                        'Total_Features': X_test.shape[1] if X_test is not None else 0,
                    }
                    
                    # Add polynomial degree if applicable
                    if fe_config.get('polynomial'):
                        fe_row['Polynomial_Degree'] = fe_config.get('poly_degree', 2)
                    
                    fe_config_list.append(fe_row)
                
                if fe_config_list:
                    fe_df = pd.DataFrame(fe_config_list)
                    fe_df.to_excel(writer, sheet_name='Feature_Engineering', index=False)
                
                # Sheet 5: Summary Statistics
                summary_data = {
                    'Paradigm': [paradigm],
                    'Total_Models': [len(results_df)],
                    'Best_R²': [results_df['Test_R²'].max()],
                    'Mean_R²': [results_df['Test_R²'].mean()],
                    'Std_R²': [results_df['Test_R²'].std()],
                    'Models_Above_0.7': [len(results_df[results_df['Test_R²'] > 0.7])],
                    'Mean_RMSE': [results_df['Test_RMSE'].mean()],
                    'Mean_MAE': [results_df['Test_MAE'].mean()],
                    'Best_Model': [results_df.loc[results_df['Test_R²'].idxmax(), 'Model']],
                    'Best_Technique': [results_df.loc[results_df['Test_R²'].idxmax(), 'Technique']],
                    'Unique_Models': [results_df['Model'].nunique()],
                    'Unique_Techniques': [results_df['Technique'].nunique()],
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format all sheets
                for worksheet in writer.sheets.values():
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
            
            return str(filepath)
        
        except Exception as e:
            raise Exception(f"Failed to export comprehensive Excel: {str(e)}")
    
    
    def export_comprehensive_json(self, results_df: pd.DataFrame,
                                 paradigm: str = "Standard",
                                 filename: str = None) -> str:
        """
        Export comprehensive results to JSON including feature importance and predictions.
        """
        import numpy as np
        
        if filename is None:
            filename = f"results_comprehensive_{paradigm}_{self.timestamp}.json"
        
        filepath = self.export_dir / filename
        
        export_json = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'paradigm': paradigm,
                'total_models': len(results_df),
                'version': '2.0'
            },
            'summary': {
                'best_r2': float(results_df['Test_R²'].max()),
                'mean_r2': float(results_df['Test_R²'].mean()),
                'std_r2': float(results_df['Test_R²'].std()),
                'best_model': results_df.loc[results_df['Test_R²'].idxmax(), 'Model'],
                'best_technique': results_df.loc[results_df['Test_R²'].idxmax(), 'Technique'],
                'unique_models': int(results_df['Model'].nunique()),
                'unique_techniques': int(results_df['Technique'].nunique()),
            },
            'results': []
        }
        
        for idx, row in results_df.iterrows():
            result_entry = {
                'model': row['Model'],
                'technique': row['Technique'],
                'metrics': {
                    'train_r2': float(row['Train_R²']),
                    'test_r2': float(row['Test_R²']),
                    'train_rmse': float(row['Train_RMSE']),
                    'test_rmse': float(row['Test_RMSE']),
                    'train_mae': float(row['Train_MAE']),
                    'test_mae': float(row['Test_MAE']),
                    'rpd': float(row['RPD']),
                },
                'hyperparameters': dict(row['Hyperparameters']) if isinstance(row['Hyperparameters'], dict) else str(row['Hyperparameters']),
            }
            
            # Add feature importance
            importance_data = row.get('Feature_Importance')
            if importance_data and isinstance(importance_data, dict):
                result_entry['feature_importance'] = {
                    'primary_type': importance_data.get('importance_type'),
                    'top_features': importance_data.get('top_features', []),
                    'methods_available': list(importance_data.get('all_methods', {}).keys()),
                }
            
            # Add feature engineering config
            fe_config = row.get('FE_Config', {})
            result_entry['feature_engineering'] = {
                'derivatives': bool(fe_config.get('derivatives', False)),
                'statistical': bool(fe_config.get('statistical', False)),
                'polynomial': bool(fe_config.get('polynomial', False)),
                'poly_degree': fe_config.get('poly_degree', 2),
                'spectral_indices': bool(fe_config.get('spectral_indices', False)),
                'pca': bool(fe_config.get('pca', False)),
                'wavelet': bool(fe_config.get('wavelet', False)),
            }
            
            # Add prediction stats
            predictions = row.get('Predictions')
            y_test = row.get('y_Test')
            if predictions is not None and y_test is not None:
                pred_array = predictions if isinstance(predictions, np.ndarray) else np.array(predictions)
                y_test_array = y_test if isinstance(y_test, np.ndarray) else np.array(y_test)
                residuals = y_test_array - pred_array
                
                result_entry['prediction_stats'] = {
                    'n_predictions': int(len(pred_array)),
                    'mean_residual': float(np.mean(residuals)),
                    'std_residual': float(np.std(residuals)),
                    'max_residual': float(np.max(np.abs(residuals))),
                    'min_residual': float(np.min(np.abs(residuals))),
                }
            
            export_json['results'].append(result_entry)
        
        with open(filepath, 'w') as f:
            json.dump(export_json, f, indent=2, default=str)
        
        return str(filepath)


class StreamlitExporter:
    """Streamlit-specific export utilities."""
    
    @staticmethod
    def get_csv_download_button(df: pd.DataFrame, filename: str = "data.csv",
                               label: str = "📥 Download CSV") -> bytes:
        """Get CSV download button for Streamlit."""
        import streamlit as st
        
        csv = df.to_csv(index=False)
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        
        return csv.encode()
    
    
    @staticmethod
    def get_excel_download_button(data_dict: Dict[str, pd.DataFrame],
                                 filename: str = "data.xlsx",
                                 label: str = "📥 Download Excel") -> bytes:
        """Get Excel download button for Streamlit."""
        import streamlit as st
        
        output = io.BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                for sheet_name, df in data_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            excel_bytes = output.getvalue()
            
            st.download_button(
                label=label,
                data=excel_bytes,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            return excel_bytes
        
        except ImportError:
            st.error("openpyxl not installed. Install with: pip install openpyxl")
            return b""
    
    
    @staticmethod
    def get_json_download_button(data: Dict[str, Any], filename: str = "data.json",
                                label: str = "📥 Download JSON") -> bytes:
        """Get JSON download button for Streamlit."""
        import streamlit as st
        
        # Convert DataFrames to dictionaries if present
        export_data = {}
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                export_data[key] = value.to_dict(orient='records')
            else:
                export_data[key] = value
        
        json_str = json.dumps(export_data, indent=2, default=str)
        
        st.download_button(
            label=label,
            data=json_str,
            file_name=filename,
            mime="application/json"
        )
        
        return json_str.encode()
    
    
    @staticmethod
    def create_multi_export_section(results_df: pd.DataFrame,
                                   metadata: Dict[str, Any] = None):
        """Create a complete export section in Streamlit."""
        import streamlit as st
        
        st.markdown("### 📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            StreamlitExporter.get_csv_download_button(
                results_df,
                f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "📊 CSV"
            )
        
        with col2:
            export_data = {
                'Results': results_df,
                'Summary': pd.DataFrame([metadata] if metadata else [])
            }
            StreamlitExporter.get_excel_download_button(
                export_data,
                f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "📈 Excel"
            )
        
        with col3:
            export_json = {
                'results': results_df.to_dict(orient='records'),
                'metadata': metadata or {}
            }
            StreamlitExporter.get_json_download_button(
                export_json,
                f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "📄 JSON"
            )
    
    
    @staticmethod
    def create_comprehensive_export_section(results_df: pd.DataFrame, paradigm: str = "Standard"):
        """
        Create comprehensive export section with feature importance, predictions, and engineering config.
        
        Includes:
        - Excel with multiple sheets (Results, Feature Importance, Predictions, Feature Engineering, Summary)
        - JSON with complete metadata and all computed values
        - CSV with main results
        """
        import streamlit as st
        from system import ResultsExporter
        
        st.markdown("### 📥 Comprehensive Export Results")
        st.markdown("""
        Download your complete analysis including:
        - **Feature Importance**: Top features from all computation methods
        - **Predictions**: Test set predictions with residuals
        - **Feature Engineering**: Configuration and feature counts
        - **Summary Statistics**: Overall model performance
        """)
        
        col1, col2, col3 = st.columns(3)
        
        # Create exporter
        exporter = ResultsExporter()
        
        with col1:
            st.markdown("#### 📊 CSV (Main Results)")
            # Prepare results for CSV
            csv_df = results_df[[
                'Model', 'Technique', 'Train_R²', 'Test_R²', 'Train_RMSE', 
                'Test_RMSE', 'Train_MAE', 'Test_MAE', 'RPD'
            ]].copy()
            csv_df['Hyperparameters'] = results_df['Hyperparameters'].astype(str)
            
            csv_bytes = csv_df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download CSV",
                data=csv_bytes,
                file_name=f"results_comprehensive_{paradigm.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.markdown("#### 📈 Excel (All Data)")
            st.markdown("*Includes: Results, Feature Importance, Predictions, Feature Engineering, Summary*")
            try:
                excel_path = exporter.export_comprehensive_excel(results_df, paradigm=paradigm)
                with open(excel_path, 'rb') as f:
                    excel_bytes = f.read()
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_bytes,
                    file_name=f"results_comprehensive_{paradigm.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error preparing Excel export: {str(e)}")
        
        with col3:
            st.markdown("#### 📄 JSON (Complete)")
            st.markdown("*Includes: All metrics, feature importance, predictions stats, engineering config*")
            try:
                json_path = exporter.export_comprehensive_json(results_df, paradigm=paradigm)
                with open(json_path, 'r') as f:
                    json_bytes = f.read().encode()
                st.download_button(
                    label="📥 Download JSON",
                    data=json_bytes,
                    file_name=f"results_comprehensive_{paradigm.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error preparing JSON export: {str(e)}")


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

_system_logger = None
_performance_tracker = None


def initialize_logging(log_dir: str = "./logs") -> SystemLogger:
    """Initialize system logging."""
    global _system_logger
    _system_logger = SystemLogger(log_dir=log_dir)
    return _system_logger


def get_logger() -> SystemLogger:
    """Get system logger instance."""
    global _system_logger
    if _system_logger is None:
        _system_logger = SystemLogger()
    return _system_logger


def get_performance_tracker() -> PerformanceTracker:
    """Get performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker(get_logger())
    return _performance_tracker


# ============================================================================
# ENVIRONMENT LOADING
# ============================================================================

def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Simple .env loader without external dependencies.
    Reads key=value pairs from file and returns as dictionary.
    
    Parameters
    ----------
    env_path : str
        Path to .env file
        
    Returns
    -------
    Dict[str, str]
        Environment variables as dictionary
    """
    import os
    
    env_vars = {}
    env_file = Path(env_path)
    
    if not env_file.exists():
        return env_vars
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    env_vars[key] = value
                    
                    # Also set in os.environ
                    os.environ[key] = value
        
        return env_vars
    except Exception as e:
        get_logger().log_error(f"Error loading .env file: {str(e)}")
        return env_vars


def get_api_key(key_name: str, env_file: str = ".env") -> Optional[str]:
    """
    Get API key from environment or .env file.
    
    Parameters
    ----------
    key_name : str
        Name of the API key (e.g., 'GEMINI_API_KEY', 'OPENAI_API_KEY')
    env_file : str
        Path to .env file
        
    Returns
    -------
    Optional[str]
        API key value or None if not found
    """
    import os
    
    # First check environment variables
    if key_name in os.environ:
        return os.environ[key_name]
    
    # Then load from .env file
    env_vars = load_env_file(env_file)
    
    return env_vars.get(key_name)
