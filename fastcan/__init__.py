"""
The implementation of fast canonical correlation analysis based feature selection
algorithm.
"""

from . import narx, utils
from ._fastcan import FastCan
from ._minibatch import minibatch
from ._refine import refine

__all__ = [
    "FastCan",
    "refine",
    "minibatch",
    "narx",
    "utils",
]
