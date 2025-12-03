"""
Mean-Variance Optimal Delta-Hedging Package
===========================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

Main package for short strangle delta-hedging analysis.
"""

__version__ = "1.0.0"
__author__ = "HKUST Financial Engineering Students"

from . import data
from . import returns
from . import covariance
from . import optimize
from . import backtest
from . import plots
from . import robo_advisor

__all__ = [
    'data',
    'returns',
    'covariance',
    'optimize',
    'backtest',
    'plots',
    'robo_advisor'
]

