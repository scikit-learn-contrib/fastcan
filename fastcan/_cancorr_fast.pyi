# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.typing as npt

def _greedy_search(
    X: npt.NDArray[np.floating],
    V: npt.NDArray[np.floating],
    t: int,
    tol: float,
    num_threads: int,
    verbose: int,
    mask: npt.NDArray[np.uint8],
    indices: npt.NDArray[np.intc],
    scores: npt.NDArray[np.floating],
) -> int: ...
