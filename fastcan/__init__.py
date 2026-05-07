"""
Fast canonical correlation analysis based search algorithm.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from . import narx, utils
from ._fastcan import FastCan
from ._lazyfastcan import LazyFastCan
from ._minibatch import minibatch
from ._refine import refine

__all__ = [
    "FastCan",
    "LazyFastCan",
    "minibatch",
    "narx",
    "refine",
    "utils",
]
