"""
The implementation of fast canonical correlation analysis based feature selection
algorithm.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from . import narx, utils
from ._fastcan import FastCan
from ._minibatch import minibatch
from ._refine import refine

__all__ = [
    "FastCan",
    "minibatch",
    "narx",
    "refine",
    "utils",
]
