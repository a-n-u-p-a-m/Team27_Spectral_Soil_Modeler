"""
Models Package
==============
Collection of ML models for spectral soil property prediction.

Models:
    - PLSR: Partial Least Squares Regression
    - GBRT: Gradient Boosting Regression Trees
    - KRR: Kernel Ridge Regression
    - SVR: Support Vector Regression
    - Cubist: Rule-Based Regression
"""

from .plsr import PLSRModel
from .gbrt import GBRTModel
from .krr import KRRModel
from .svr import SVRModel
from .cubist import CubistModel

__all__ = ['PLSRModel', 'GBRTModel', 'KRRModel', 'SVRModel', 'CubistModel']
