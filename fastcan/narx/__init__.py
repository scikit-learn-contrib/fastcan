"""
Nonlinear autoregressive exogenous (NARX) model for system identification.
"""

# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

from ._base import NARX
from ._feature import (
    fd2tp,
    make_poly_features,
    make_poly_ids,
    make_time_shift_features,
    make_time_shift_ids,
    tp2fd,
)
from ._utils import (
    make_narx,
    print_narx,
)

__all__ = [
    "NARX",
    "fd2tp",
    "make_narx",
    "make_poly_features",
    "make_poly_ids",
    "make_time_shift_features",
    "make_time_shift_ids",
    "print_narx",
    "tp2fd",
]
