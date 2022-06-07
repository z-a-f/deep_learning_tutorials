from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset(data_set,
                      # Estimator
                      estimator: Optional[Callable]=None,
                      *,
                      # Visualization parameters
                      ax: Optional[mpl.axes.Axes]=None,
                      contour_kwargs: Optional[Dict[str, Any]]=None,
                      scatter_kwargs: Optional[Dict[str, Any]]=None):
    r"""Visualizes the dataset using a scatter plot.

    If 'estimator' is provided, this utility also shows the estimator prediction over the whole space.

    Args:
        data_set: Classification dataset to be plotted.
                  It should have two entries, one for the predictors and another
                  for the labels.
        estimator: A callable that takes predictors to create labels
        ax (optional): Instance of the pyplot axes
        contour_kwargs (optional): Dictionary with contour arguments used to visualize the estimator.
        scatter_kwargs (optional): Dictionary with scatter arguments used to visualize the datapoints.


    Note: This only draws the data classes over the first two columns in the dataset.
    Note: Currently this is only for two-class datasets.

    """
    if ax is None:
        fig, ax = plt.subplots(1)

    # Red/Blue coloring for two labels.
    cm = plt.cm.RdBu
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])

    if contour_kwargs is None and estimator is not None:
        contour_kwargs = {'cmap': cm, 'alpha': 0.8, 'levels': 100}
    if scatter_kwargs is None:
        scatter_kwargs = {'cmap': cm_bright, 'edgecolors': 'k'}
    
    X_data, y_data = data_set
    x_min, x_max = X_data[:, 0].min() - .5, X_data[:, 0].max() + .5
    y_min, y_max = X_data[:, 1].min() - .5, X_data[:, 1].max() + .5
    
    # Plot the dcision boundaries if possible
    if estimator is not None:
        assert callable(estimator)
        h = 0.02  # Step size for mesh grid
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
        ravel_grid = np.c_[xx.ravel(), yy.ravel()]
        if hasattr(estimator, 'decision_function'):
            Z = estimator.decision_function(ravel_grid)
        elif hasattr(estimator, 'predict_proba'):
            Z = estimator.predict_proba(ravel_grid)[:, 1]
        else:
            # Assume the estimator is callable
            Z = estimator(ravel_grid)
            Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, **contour_kwargs)

    ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, **scatter_kwargs)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_xticks(())
    # ax.set_yticks(())

    return ax
