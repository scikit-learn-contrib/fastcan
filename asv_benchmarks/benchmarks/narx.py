import itertools
import pickle

from sklearn.metrics import r2_score

from fastcan.narx import make_narx

from .common import Benchmark, get_estimator_path
from .datasets import _co2_dataset


class NARXBenchmark(Benchmark):
    """
    Benchmarks for NARX.
    """

    param_names = ["opt_alg"]
    params = (["osa", "msa"],)

    n_terms_to_select = 10
    max_delay = 10
    poly_degree = 2

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

            estimator = estimator.fit(X, y, coef_init="one_step_ahead")

            est_path = get_estimator_path(self, params)
            with est_path.open(mode="wb") as f:
                pickle.dump(estimator, f)

    def make_data(self, params):
        return _co2_dataset()

    def time_fit(self, *args):
        (opt_alg,) = args
        if opt_alg == "osa":
            coef_init = None
        else:
            coef_init = [0] * (self.n_terms_to_select + 1)
        self.estimator.fit(self.X, self.y, coef_init=coef_init)

    def peakmem_fit(self, *args):
        self.estimator.fit(self.X, self.y)

    def track_train_score(self, *args):
        y_pred = self.estimator.predict(self.X, self.y[: self.max_delay])
        return float(r2_score(self.y, y_pred))

    def track_test_score(self, *args):
        y_val_pred = self.estimator.predict(
            self.X_val,
            self.y_val[: self.max_delay],
        )
        return float(r2_score(self.y_val, y_val_pred))

    def time_predict(self, *args):
        self.estimator.predict(self.X, self.y[: self.max_delay])

    def peakmem_predict(self, *args):
        self.estimator.predict(self.X, self.y[: self.max_delay])
