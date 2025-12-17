# Authors: The fastcan developers
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.typing as npt

def _predict(
    X: npt.NDArray[np.float64],
    y_ref: npt.NDArray[np.float64],
    coef: npt.NDArray[np.float64],
    intercept: npt.NDArray[np.float64],
    feat_ids: npt.NDArray[np.intc],
    delay_ids: npt.NDArray[np.intc],
    output_ids: npt.NDArray[np.intc],
    session_sizes_cumsum: npt.NDArray[np.intc],
    y_hat: npt.NDArray[np.float64],
) -> None: ...
def _update_dydx(
    X: npt.NDArray[np.float64],
    y_hat: npt.NDArray[np.float64],
    coef: npt.NDArray[np.float64],
    feat_ids: npt.NDArray[np.intc],
    delay_ids: npt.NDArray[np.intc],
    y_ids: npt.NDArray[np.intc],
    grad_yyd_ids: npt.NDArray[np.intc],
    grad_delay_ids: npt.NDArray[np.intc],
    grad_coef_ids: npt.NDArray[np.intc],
    grad_feat_ids: npt.NDArray[np.intc],
    session_sizes_cumsum: npt.NDArray[np.intc],
    dydx: npt.NDArray[np.float64],
    dcf: npt.NDArray[np.float64],
) -> None: ...
