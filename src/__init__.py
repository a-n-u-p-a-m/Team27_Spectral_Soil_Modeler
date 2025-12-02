"""
Spectral Soil Modeler - Source Package
======================================

This package contains all backend modules for the automated ML pipeline.

Modules:
    - data_loader: Load and validate spectral data
    - preprocessing: Data cleaning and spectral transformations
    - models: ML model implementations and training
    - evaluation: Model evaluation and metrics
    - persistence: Save and load models
    - logger: System logging
    - app: Streamlit web application
"""

__version__ = "0.1.0"
__author__ = "Team 27 - SERC, IIIT Hyderabad"

from . import data_loader

__all__ = ['data_loader']
