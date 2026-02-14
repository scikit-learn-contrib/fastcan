import itertools
import pickle

from fastcan.narx import make_narx

from .common import Benchmark, get_estimator_path
from .datasets import _co2_dataset


class HessBenchmark(Benchmark):
    """
    Benchmarks for Hessian matrix computation in NARX.
    """

    param_names = ["opt_alg"]
    params = (["lsq", "min-hess", "min-hessp"],)

    n_terms_to_select = 10
    max_delay = 10
    poly_degree = 2

    def _parse_args(self, *args):
        (opt_alg,) = args
        if opt_alg == "lsq":
            solver = "least_squares"
            method = "trf"
        elif opt_alg == "min-hess":
            solver = "minimize"
            method = "trust-exact"
        else:
            solver = "minimize"
            method = "trust-ncg"
        return solver, method

    def setup_cache(self):
        """Pickle a fitted estimator for all combinations of parameters"""
        # This is run once per benchmark class.

        param_grid = list(itertools.product(*self.params))

        for params in param_grid:
            X, _, y, _ = self.make_data(params)
            estimator = make_narx(
                X,
                y,
                n_terms_to_select=self.n_terms_to_select,
                max_delay=self.max_delay,
                poly_degree=self.poly_degree,
            )

            estimator = estimator.fit(X, y)

            est_path = get_estimator_path(self, params)
            with est_path.open(mode="wb") as f:
                pickle.dump(estimator, f)

    def make_data(self, params):
        return _co2_dataset()

    def time_fit(self, *args):
        solver, method = self._parse_args(*args)
        self.estimator.fit(
            self.X, self.y, coef_init="one_step_ahead", solver=solver, method=method
        )

    def peakmem_fit(self, *args):
        solver, method = self._parse_args(*args)
        self.estimator.fit(
            self.X, self.y, coef_init="one_step_ahead", solver=solver, method=method
        )
