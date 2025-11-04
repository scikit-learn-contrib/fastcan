import itertools
import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression

from fastcan import FastCan

from .common import Benchmark, get_estimator_path
from .datasets import _digits_dataset, _synth_regression_dataset


class FastCanBenchmark(Benchmark):
    """
    Benchmarks for FastCan.
    """

    param_names = ["task", "alg"]
    params = (["classif", "reg"], ["h", "eta", "beam"])

    def setup_cache(self):
        """Pickle a fitted estimator for all combinations of parameters"""
        # This is run once per benchmark class.

        param_grid = list(itertools.product(*self.params))

        for params in param_grid:
            _, alg = params
            X, _, y, _ = self.make_data(params)

            if alg == "h":
                eta = False
                beam_width = 1
            elif alg == "eta":
                eta = True
                beam_width = 1
            else:
                eta = False
                beam_width = 10
            estimator = FastCan(
                n_features_to_select=20,
                eta=eta,
                beam_width=beam_width
            )
            estimator.fit(X, y)

            est_path = get_estimator_path(self, params)
            with est_path.open(mode="wb") as f:
                pickle.dump(estimator, f)

    def make_data(self, params):
        task, _ = params
        if task == "classif":
            return _digits_dataset()
        return _synth_regression_dataset()

    def time_fit(self, *args):
        self.estimator.fit(self.X, self.y)

    def peakmem_fit(self, *args):
        self.estimator.fit(self.X, self.y)

    def track_train_score(self, *args):
        task, _ = args
        X_t = self.estimator.transform(self.X)
        if task == "classif":
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_t, self.y)
            return float(clf.score(X_t, self.y))
        else:
            reg = LinearRegression()
            reg.fit(X_t, self.y)
            return float(reg.score(X_t, self.y))

    def track_test_score(self, *args):
        task, _ = args
        X_t = self.estimator.transform(self.X_val)
        if task == "classif":
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_t, self.y_val)
            return float(clf.score(X_t, self.y_val))
        else:
            reg = LinearRegression()
            reg.fit(X_t, self.y_val)
            return float(reg.score(X_t, self.y_val))

    def time_transform(self, *args):
        self.estimator.transform(self.X)

    def peakmem_transform(self, *args):
        self.estimator.transform(self.X)
