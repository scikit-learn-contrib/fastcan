import itertools
import pickle
from functools import partial

from fastcan import FastCan, LazyFastCan
from fastcan.narx import gen_poly_features, make_poly_features, make_poly_ids

from .common import Benchmark, get_estimator_path
from .datasets import _synth_regression_dataset


class LazyFastCanBenchmark(Benchmark):
    """
    Benchmarks comparing LazyFastCan with FastCan.
    """

    param_names = ["estimator"]
    params = (["lazy", "normal"],)

    n_samples = 10000
    n_features_to_select = 10
    n_features = 10
    poly_degree = 3

    def setup_cache(self):
        """Pickle a fitted estimator for all combinations of parameters"""
        # This is run once per benchmark class.

        param_grid = list(itertools.product(*self.params))

        for params in param_grid:
            (est,) = params
            X, _, y, _ = self.make_data(params)

            if est == "normal":
                estimator = FastCan(
                    n_features_to_select=self.n_features_to_select, verbose=0
                )
            else:
                ids = make_poly_ids(self.n_features, degree=self.poly_degree)
                gen = partial(gen_poly_features, ids=ids)
                estimator = LazyFastCan(
                    n_features_to_select=self.n_features_to_select,
                    feature_generator=gen,
                )

            estimator.fit(X, y)

            est_path = get_estimator_path(self, params)
            with est_path.open(mode="wb") as f:
                pickle.dump(estimator, f)

    def make_data(self, params):
        (est,) = params
        X, X_val, y, y_val = _synth_regression_dataset(
            n_samples=self.n_samples, n_features=self.n_features
        )

        if est == "normal":
            ids = make_poly_ids(self.n_features, degree=self.poly_degree)
            X = make_poly_features(X, ids)
            X_val = make_poly_features(X_val, ids)

        return X, X_val, y, y_val

    def time_fit(self, *args):
        self.estimator.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        self.estimator.fit(self.X, self.y)
