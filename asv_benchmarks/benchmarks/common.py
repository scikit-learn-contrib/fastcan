import pickle
import timeit
from abc import ABC, abstractmethod
from pathlib import Path


def get_estimator_path(benchmark, params):
    """Get path of pickled fitted estimator"""
    path = Path(__file__).resolve().parent / "cache" / "estimators"

    filename = (
        benchmark.__class__.__name__
        + "_estimator_"
        + "_".join(list(map(str, params)))
        + ".pkl"
    )

    return path / filename


class Benchmark(ABC):
    """Abstract base class for all the benchmarks"""

    timer = timeit.default_timer  # wall time
    timeout = 500

    # save estimators
    current_path = Path(__file__).resolve().parent
    cache_path = current_path / "cache"
    cache_path.mkdir(exist_ok=True)
    (cache_path / "estimators").mkdir(exist_ok=True)

    def setup(self, *params):
        """Generate dataset and load the fitted estimator"""
        # This is run once per combination of parameters and per repeat so we
        # need to avoid doing expensive operations there.

        self.X, self.X_val, self.y, self.y_val = self.make_data(params)

        est_path = get_estimator_path(self, params)
        with est_path.open(mode="rb") as f:
            self.estimator = pickle.load(f)

    @abstractmethod
    def make_data(self, params):
        """Return the dataset for a combination of parameters"""
        # The datasets are cached using joblib.Memory so it's fast and can be
        # called for each repeat

    @property
    @abstractmethod
    def params(self):
        pass
