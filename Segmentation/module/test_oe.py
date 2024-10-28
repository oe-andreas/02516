import numpy as np

from .plot import plot_all_metrics

observed_eval_metrics_array = np.random.rand(3, 2, 2, 8)

plot_all_metrics(observed_eval_metrics_array, "delete")
