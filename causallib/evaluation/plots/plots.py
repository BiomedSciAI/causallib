"""
(C) Copyright 2019 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on Aug 22, 2018

"""
from itertools import cycle
from typing import Callable
import warnings

import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from sklearn import metrics

# TODO: propensity distribution using CDF (and not reflecting if so)
# TODO: consider making plots to not rely on pandas input (and can work more generally with numpy)?
# TODO: consider refactoring each type (family?) of plots to its own module (unify through __init__?)
# TODO: consider making plot module be class-based instead, taking its argument during init
#       and having a `plot()` interface

CONTINUOUS_ACCURACY_PLOT = "continuous_accuracy"
RESIDUALS_PLOT = "residuals"
COMMON_SUPPORT_PLOT = "common_support"
ROC_CURVE_PLOT = "roc_curve"
PR_CURVE_PLOT = "pr_curve"
CALIBRATION_PLOT = "calibration"
WEIGHT_DISTRIBUTION_PLOT = "weight_distribution"
COVARIATE_BALANCE_LOVE_PLOT = "covariate_balance_love"
COVARIATE_BALANCE_SLOPE_PLOT = "covariate_balance_slope"
COVARIATE_BALANCE_GENERIC_PLOT = "covariate_balance"

WeightPlotNames = frozenset({
    WEIGHT_DISTRIBUTION_PLOT,
    COVARIATE_BALANCE_LOVE_PLOT,
    COVARIATE_BALANCE_SLOPE_PLOT,
})

ContinuousOutputPlotNames = frozenset({
    CONTINUOUS_ACCURACY_PLOT,
    RESIDUALS_PLOT,
    COMMON_SUPPORT_PLOT,
})

BinaryOutputPlotNames = frozenset({CALIBRATION_PLOT, ROC_CURVE_PLOT, PR_CURVE_PLOT})


PropensityPlotNames = frozenset(BinaryOutputPlotNames | WeightPlotNames)



def lookup_name(name: str) -> Callable:
    """Lookup function for plot name.

    Canonical plot names are defined in this file as globals.
    Incorrect names will raise KeyError.

    Args:
        name (str): plot name to lookup

    Returns:
        Callable: plot function
    """
 
    return {
        CONTINUOUS_ACCURACY_PLOT: plot_continuous_prediction_accuracy_folds,
        RESIDUALS_PLOT: plot_residual_folds,
        COMMON_SUPPORT_PLOT: plot_counterfactual_common_support_folds,
        ROC_CURVE_PLOT: plot_roc_curve_folds,
        PR_CURVE_PLOT: plot_precision_recall_curve_folds,
        CALIBRATION_PLOT: plot_calibration_folds,
        WEIGHT_DISTRIBUTION_PLOT: plot_propensity_score_distribution_folds,
        COVARIATE_BALANCE_LOVE_PLOT: plot_mean_features_imbalance_love_folds,
        COVARIATE_BALANCE_SLOPE_PLOT: plot_mean_features_imbalance_slope_folds,
    }[name]


def _calculate_mutual_bins(x, y, bins="auto"):
    """
    A common support for two vectors.

    Args:
        x (pd.Series):
        y (pd.Series):
        bins: compatible with numpy's bins parameter.

    Returns:
        np.array: bins cutoffs.
    """
    data = np.append(x, y)
    bins = np.histogram(data, bins=bins)[1]
    return bins


def plot_counterfactual_common_support(prediction, a, ax=None):
    cv = [np.arange(a.shape[0])]
    ax = plot_counterfactual_common_support_folds([prediction], hue_by=a, cv=cv, ax=ax)
    return ax


def plot_counterfactual_common_support_folds(
    predictions, hue_by, cv, alpha_by_density=True, ax=None
):
    """Plot the scatter plot of y0 vs. y1 for multiple scoring results, colored by the treatment

    Args:
        predictions (list[pd.Series]): List, the size of number of folds, of outcome prediction values.
        hue_by (pd.Series): Group assignment (as in treatment assignment) of the entire dataset.
                            (indices from `cv` will be used to slice this vector)
        cv (list[np.array]): List, the size of number of folds, of row indices (as in iloc locations) - the indices
                             of samples participating the fold.
        alpha_by_density (bool): Whether to calculate points alpha value (transparent-opaque) with density estimation.
                                 This can take some time to compute for large number of points.
                                 If False, alpha calculation will be a simple fast heuristic.
        ax (plt.Axes): The axes on which the plot will be displayed. Optional.

    """
    effect_folds = [
        (prediction.iloc[:, 1] - prediction.iloc[:, 0]).mean()
        for prediction in predictions
    ]
    predictions = pd.concat(predictions)  # type: pd.DataFrame
    treatment = pd.concat([hue_by.iloc[fold_idx] for fold_idx in cv])  # type: pd.Series

    ax = _scatter_hue(
        predictions.iloc[:, 0],
        predictions.iloc[:, 1],
        treatment,
        alpha_by_density,
        ax=ax,
    )

    effect_label = rf"mean effect={np.mean(effect_folds):.2g}"
    effect_label += rf"$\pm${np.std(effect_folds):.2g}" if len(effect_folds) > 1 else ""
    ax.plot(
        [], [], color=ax.get_facecolor(), label=effect_label  # Use background color
    )
    _add_diagonal(ax)
    ax.legend(loc="best")
    ax.set_xlabel(r"Predicted $Y^0$")
    ax.set_ylabel(r"Predicted $Y^1$")
    ax.set_title("Predicted Common Support")
    return ax


def plot_continuous_prediction_accuracy(
    predictions, y, a, alpha_by_density=True, ax=None
):
    cv = [np.arange(a.shape[0])]
    ax = plot_continuous_prediction_accuracy_folds(
        [predictions], y, a, cv, alpha_by_density, ax=ax, plot_residuals=False
    )
    return ax


def plot_continuous_prediction_accuracy_folds(
    predictions, y, a, cv, alpha_by_density=True, plot_residuals=False, ax=None
):
    # Concatenate data across folds:
    treatments = []
    outcomes = []
    predictions_on_actual = []
    r2_scores = []
    for fold_prediction, fold_idx in zip(predictions, cv):
        fold_a = a.iloc[fold_idx]
        fold_y = y.iloc[fold_idx]
        if plot_residuals:
            fold_y = fold_y - fold_prediction

        r2_scores.append(metrics.r2_score(fold_y, fold_prediction))
        treatments.append(fold_a)
        outcomes.append(fold_y)
        predictions_on_actual.append(fold_prediction)

    treatments = pd.concat(treatments)  # type: pd.Series
    outcomes = pd.concat(outcomes)  # type: pd.Series
    predictions_on_actual = pd.concat(predictions_on_actual)  # type: pd.Series

    ax = _scatter_hue(predictions_on_actual, outcomes, treatments, alpha_by_density, ax)

    # R-squared label:
    if not plot_residuals:
        r2_label = rf"$R^2={np.mean(r2_scores):.2f}"
        r2_label += rf"\pm{np.std(r2_scores):.2f}$" if len(r2_scores) > 1 else "$"
        ax.plot(
            [], [], color=ax.get_facecolor(), label=r2_label
        )  # invisible color so as to not show line in legend
        _add_diagonal(ax)

    ax.legend(loc="best")
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Prediction residuals" if plot_residuals else "True values")
    ax.set_title("Residual Plot" if plot_residuals else "Continuous Accuracy Plot")
    return ax


def plot_residual_folds(predictions, y, a, cv, alpha_by_density=True, ax=None):
    ax = plot_continuous_prediction_accuracy_folds(
        predictions, y, a, cv, alpha_by_density, plot_residuals=True, ax=ax
    )
    ax.axhline(0.0, linestyle="--", color="grey", zorder=0, alpha=0.75)
    return ax


def plot_residual(predictions, y, a, alpha_by_density=True, ax=None):
    cv = [np.arange(a.shape[0])]
    ax = plot_residual_folds([predictions], y, a, cv, alpha_by_density, ax)
    return ax


def _scatter_hue(x, y, hue, alpha_by_density=True, ax=None):
    ax = ax or plt.gca()

    points_rgba = (
        _get_alpha_per_point_with_density(X=[x, y], hue=hue)
        if alpha_by_density
        else None
    )

    for i, treatment_val in enumerate(np.sort(np.unique(hue))):
        idx_mask = hue == treatment_val  # type: pd.Series
        cur_color = (
            points_rgba.loc[idx_mask].values if points_rgba is not None else None
        )
        cur_alpha = np.clip(10 / np.sqrt(idx_mask.sum()), 0.01, 1)

        ax.scatter(
            x=x.loc[idx_mask],
            y=y.loc[idx_mask],
            alpha=cur_alpha if points_rgba is None else None,
            facecolor=cur_color,
            edgecolors="none",
            label=f"treatment={treatment_val}",
        )
    return ax


def _get_alpha_per_point_with_density(X, hue, min_alpha_bound=0.3, max_alpha_bound=1.0):
    """
    Matplotlib does not support pointwise alpha values (rather, constant value for an entire plt.plot()).
    This function will utilize a supported pointwise color-scheme, using rgba, and passing the individual alpha values
    as the 4th dimension ('a') of the rgba.

    Args:
        X: in a form compatible with statsmodels' KDEMultivariate (list of pd.Series, or pd.DataFrame)
        hue (pd.Series): A vector with group assignment for each point in x.
        min_alpha_bound (float | None): Value between 0 and 1, used to linearly rescale the alpha values.
                                        If None, rescale is avoided.
                                        Default of 0.3, since lower values are usually too unobservable.
        max_alpha_bound (float | None): Value between 0 and 1, used to linearly rescale the alpha values.
                                        If None, rescale is avoided.

    Returns:

    """
    points_rgba = pd.DataFrame(index=hue.index, columns=list("rgba"), dtype=np.float64)

    # Calculate alpha for each point based on its density:
    kde = sm.nonparametric.KDEMultivariate(data=X, var_type="cc", bw="normal_reference")
    # kde.bw = kde.bw * 0.5         # Rescale bandwidth to be narrower
    points_density = kde.pdf(X)
    # Invert values - the denser the point -> the lower its alpha (more transparent)
    points_alpha = 1 / points_density
    if (min_alpha_bound is not None) and (max_alpha_bound is not None):
        #   Rescale alphas (linearly) to the range of 0.3 to 1:
        points_alpha = min_alpha_bound + (max_alpha_bound - min_alpha_bound) * (
            (points_alpha - points_alpha.min())
            / (points_alpha.max() - points_alpha.min())
        )
    points_rgba["a"] = points_alpha  # Assign the alpha values

    for i, hue_val in enumerate(np.sort(np.unique(hue))):
        idx_mask = hue == hue_val
        cur_color = f"C{i}"  # Cycle through the colors
        cur_color = matplotlib.colors.to_rgb(
            cur_color
        )  # Get RGB value of the current color
        points_rgba.loc[
            idx_mask, ["r", "g", "b"]
        ] = cur_color  # Assign that constant RGB val for all current points

    return points_rgba


def plot_calibration_folds(
    predictions,
    targets,
    cv,
    n_bins=10,
    plot_se=True,
    plot_rug=False,
    plot_histogram=False,
    quantile=False,
    ax=None,
):
    """Plot calibration curves for multiple models (presumably in folds)

    Args:
        predictions (list[pd.Series]): list (each entry of a fold) of arrays - probability ("scores") predictions.
        targets (pd.Series): true labels to calibrate against on the overall data (not divided to folds).
        cv (list[np.array]):
        n_bins (int): number of bins to evaluate in the plot
        plot_se (bool): Whether to plot standard errors around the mean bin-probability estimation.
        plot_rug:
        plot_histogram:
        quantile (bool): If true, the binning of the calibration curve is by quantiles. Default is false
        ax (plt.Axes): Optional

    Note:
        One of plot_propensity or plot_model must be True.

    Returns:

    """
    for i, idx_fold in enumerate(cv):
        predictions_fold = predictions[i]
        target_fold = targets.iloc[idx_fold]

        ax = _plot_calibration_single(
            y_true=target_fold,
            y_prob=predictions_fold,
            n_bins=n_bins,
            plot_diagonal=False,
            plot_se=plot_se,
            plot_rug=plot_rug,
            plot_histogram=plot_histogram,
            quantile=quantile,
            label=f"fold {i}",
            ax=ax,
        )
    _add_diagonal(ax)
    ax.legend(loc="best")
    # ax.set_title("{} Calibration".format("Propensity" if y is None else "Outcome"))
    ax.set_title("Calibration")
    return ax


def plot_calibration(
    predictions,
    targets,
    n_bins=10,
    plot_se=True,
    plot_rug=False,
    plot_histogram=True,
    quantile=False,
    ax=None,
):
    cv = [np.arange(predictions.shape[0])]
    return plot_calibration_folds(
        [predictions],
        targets,
        cv=cv,
        n_bins=n_bins,
        plot_se=plot_se,
        plot_rug=plot_rug,
        plot_histogram=plot_histogram,
        quantile=quantile,
        ax=ax,
    )


def _plot_calibration_single(
    y_true,
    y_prob,
    n_bins=10,
    plot_diagonal=True,
    plot_se=True,
    plot_rug=False,
    plot_histogram=False,
    quantile=False,
    label=None,
    ax=None,
):
    """Plot a calibration curve showing how well y_prob predicts the probability of a binary outcome y

    The standard deviation of a binomial distribution p(1-p)/sqrt(n) is used to calculate the values for which p
    would be one standard deviation away. This means we are looking for
    r +/- sqrt(r(1-r)/n) = p
    This provides a cubic equation for r whose solution is
    r = (2np+1 +/- sqrt(4np(1-p)+1)) / (2n+2)

    Args:
        y_prob (pd.Series):
        y_true (pd.Series):
        n_bins (int): the number of bins to use for the calibration plot
        plot_se (bool): Whether to plot standard errors around the
                        mean bin-probability estimation.
        plot_diagonal (bool): Whether to plot a diagonal line or not.
        plot_rug (bool): Whether to plot rug of the prediction
        plot_histogram (bool): Whether to plot histogram at the background.
        quantile (bool): If False specifies equal sized bins,
                         if True splits the probabilities into n_bins quantiles.
        ax (plt.Axes):
        label(str): The label for the plotted line

    Returns:

    """
    ax = ax or plt.gca()
    if quantile:
        bins = np.unique(
            np.percentile(y_prob, np.linspace(0, 100, n_bins + 1).astype(int))
        )
        # in case all values of y_prob are the same
        bins = bins if len(bins) > 1 else np.concatenate([bins, bins])
        bins[-1] += 1e-8
        prob_true, prob_pred, counts = calibration_curve(y_true, y_prob, bins=bins)
    else:
        prob_true, prob_pred, counts = calibration_curve(y_true, y_prob, bins=n_bins)
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

    if plot_rug:
        ax.plot(y_prob, np.full_like(y_prob, 0.01), "|", color="black", alpha=0.7)

    line_color = None
    if plot_histogram:
        hist_line = ax.plot(
            bins, (counts / counts.sum()), drawstyle="steps-post", alpha=0.8
        )
        hist_line = hist_line[0]
        # keep histogram behind any new lines that are plotted after it.
        hist_line.set_zorder(2)
        # if plotting hist, keep track of color to use in the line to be plotted
        line_color = hist_line.get_color()

    if plot_diagonal:
        _add_diagonal(ax)
    lines = ax.plot(prob_pred, prob_true, "s-", color=line_color, label=label)

    # Plot standard error:
    if plot_se:
        disc = (4 * counts * prob_true) * (1 - prob_true) + 1
        upper = (2 * counts * prob_true + 1 + np.sqrt(disc)) / (2 * counts + 2)
        lower = (2 * counts * prob_true + 1 - np.sqrt(disc)) / (2 * counts + 2)
        ax.fill_between(
            x=prob_pred, y1=lower, y2=upper, color=lines[-1].get_color(), alpha=0.5
        )

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    return ax


def calibration_curve(y_true, y_prob, bins=5):
    """
    Compute calibration curve of a classifier given its scores output and true label assignment.

    Args:
        y_true (pd.Series): True binary label assignment.
        y_prob (pd.Series): Predicted probability of each sample being the positive label.
        bins (int | list | np.ndarray | pd.Series):
            If int, it defines the number of equal-width bins in the
            given range (5, by default).
            If bins a sequence, it defines the bin edges, including the
            rightmost edge, allowing for non-uniform bin widths.

    Returns:
        (pd.Series, pd.Series, pd.Series): empirical_prob, predicted_prob, bin_counts
            empirical_prob: The fraction of positive labels in each bins
            predicted_prob: The average of predicted probability in each bin
            bin_counts: The number of samples fallen in each bin

    References:
        [1] Zadrozny, B., & Elkan, C. (2002, July).
            Transforming classifier scores into accurate multiclass probability estimates

    """
    # Get binning out of provided bins
    if type(bins) is int:
        bins = np.linspace(0.0, 1.0 + 1e-8, bins + 1)
    elif hasattr(bins, "__len__") and not isinstance(bins, str):  # Some sort of vector
        bins = np.sort(np.ravel(bins))
        if y_prob.max() > bins.max() or y_prob.min() < bins.min():
            raise ValueError("y_prob has values outside the provided bins")
    else:
        raise TypeError("bins must either be an integer or a sequence of scalars")

    bin_of_samples = pd.cut(y_prob, bins, labels=np.arange(len(bins) - 1)).astype(int)
    predicted_prob = y_prob.groupby(bin_of_samples).mean()
    empirical_prob = y_true.groupby(bin_of_samples).mean()
    bin_counts = bin_of_samples.value_counts(sort=False)
    return empirical_prob, predicted_prob, bin_counts


def plot_roc_curve_folds(
    curve_data, ax=None, plot_folds=False, label_folds=False, label_std=False, **kwargs
):
    num_of_curves = len(curve_data.keys())
    color_list = [f"C{_}" for _ in range(num_of_curves)]

    for (curve_name, curve_data), color in zip(curve_data.items(), color_list):
        fprs = curve_data["FPR"]
        tprs = curve_data["TPR"]
        aucs = curve_data["AUC"]

        ax = _plot_single_performance_curve(
            fprs,
            tprs,
            aucs,
            "AUC",
            color,
            curve_name,
            label_std,
            label_folds,
            plot_folds,
            num_of_curves != 1,
            ax,
        )
    # Plot chance curve:
    ax.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="black", label="Chance", alpha=0.8
    )

    ax.set_xlim(left=-0.05, right=1.05)
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return ax


def plot_precision_recall_curve_folds(
    curve_data, ax=None, plot_folds=False, label_folds=False, label_std=False, **kwargs
):
    # TODO: Check why it does not end at class prevalence (for recall=1.0)
    num_of_curves = len(curve_data.keys())
    color_list = [f"C{_}" for _ in range(num_of_curves)]

    pos_class_prevalence = curve_data.pop("prevalence", None)

    for (curve_name, curve_data), color in zip(curve_data.items(), color_list):
        recalls = curve_data["Recall"]
        precisions = curve_data["Precision"]
        aps = curve_data["AP"]

        ax = _plot_single_performance_curve(
            recalls,
            precisions,
            aps,
            "AP",
            color,
            curve_name,
            label_std,
            label_folds,
            plot_folds,
            num_of_curves != 1,
            ax,
        )
    # Plot chance curve:
    if pos_class_prevalence is not None:
        ax.plot(
            [0, 1],
            [pos_class_prevalence, pos_class_prevalence],
            linestyle="--",
            lw=2,
            color="black",
            label="Chance",
            alpha=0.8,
        )

    ax.set_xlim(left=-0.05, right=1.05)
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR Curve")
    ax.legend(loc="lower left")
    return ax


def _plot_single_performance_curve(
    xs,
    ys,
    areas,
    areas_type,
    color="C0",
    curve_name="",
    label_std=False,
    label_folds=False,
    plot_folds=False,
    colored_folds=False,
    ax=None,
):
    ax = ax or plt.gca()
    assert len(xs) == len(ys) == len(areas)

    n_folds = len(xs)
    x_domain = np.linspace(0, 1, 100)
    ys_interp = []
    for i in range(n_folds):
        if areas_type == "AP":  # precision/recall need to be reversed for interpolation
            ys_interp.append(np.interp(x_domain, xs[i][::-1], ys[i][::-1]))
        else:
            ys_interp.append(np.interp(x_domain, xs[i], ys[i]))
            ys_interp[-1][0] = 0.0
        area = areas[i]

        folds_label = f"Fold {i} ({areas_type} = {area:.2f})" if label_folds else None
        if plot_folds:
            # use multiple colors if plotting only one stratum
            folds_color = None if colored_folds else color
            ax.plot(xs[i], ys[i], lw=1, alpha=0.3, color=folds_color, label=folds_label)

    # Plot main (folds average) curve
    mean_ys = np.nanmean(ys_interp, axis=0)
    # if areas_type == "AUC":
    #     mean_ys[-1] = 1.0
    mean_area = np.nanmean(areas)
    std_area = np.nanstd(areas)
    ax.plot(
        x_domain,
        mean_ys,
        color=color,
        label=rf"{curve_name} ({areas_type} = {mean_area:.2f} $\pm$ {std_area:.2f})",
        lw=2,
        alpha=0.9,
    )

    # Plot uncertainty around main curve:
    ys_std = np.std(ys_interp, axis=0)
    upper_ys = np.minimum(mean_ys + ys_std, 1)
    lower_ys = np.maximum(mean_ys - ys_std, 0)
    std_label = r"$\pm$ 1 std. dev." if label_std else None
    ax.fill_between(
        x_domain, lower_ys, upper_ys, color=color, alpha=0.2, label=std_label
    )

    return ax


def plot_propensity_score_distribution(
    propensity,
    treatment,
    reflect=True,
    kde=False,
    cumulative=False,
    norm_hist=True,
    ax=None,
):
    """
    Plot the distribution of propensity score

    Args:
        propensity (pd.Series):
        treatment (pd.Series):
        reflect (bool): Whether to plot second treatment group on the opposite sides of the x-axis.
                        This can only work if there are exactly two groups.
        kde (bool): Whether to plot kernel density estimation
        cumulative (bool): Whether to plot cumulative distribution.
        norm_hist (bool): If False - use raw counts on the y-axis.
                          If kde=True, then norm_hist should be True as well.
        ax (plt.Axes | None):

    Returns:

    """
    # assert propensity.index.symmetric_difference(a.index).size == 0
    ax = ax or plt.gca()
    if kde and not norm_hist:
        warnings.warn(
            "kde=True and norm_hist=False is not supported. Forcing norm_hist from False to True."
        )
        norm_hist = True
    bins = np.histogram(propensity, bins="auto")[1]
    plot_params = dict(bins=bins, density=norm_hist, alpha=0.5, cumulative=cumulative)

    unique_treatments = np.sort(np.unique(treatment))
    for treatment_number, treatment_value in enumerate(unique_treatments):
        cur_propensity = propensity.loc[treatment == treatment_value]
        cur_color = f"C{treatment_number}"
        ax.hist(
            cur_propensity,
            label=f"treatment = {treatment_value}",
            color=[cur_color],
            **plot_params,
        )
        if kde:
            cur_kde = gaussian_kde(cur_propensity)
            min_support = max(0, cur_propensity.values.min() - cur_kde.factor)
            max_support = min(1, cur_propensity.values.max() + cur_kde.factor)
            X_plot = np.linspace(min_support, max_support, 200)
            if cumulative:
                density = np.array(
                    [cur_kde.integrate_box_1d(X_plot[0], x_i) for x_i in X_plot]
                )
                ax.plot(
                    X_plot,
                    density,
                    color=cur_color,
                )
            else:
                ax.plot(
                    X_plot,
                    cur_kde.pdf(X_plot),
                    color=cur_color,
                )
    if reflect:
        if len(unique_treatments) != 2:
            raise ValueError(
                "Reflecting density across X axis can only be done for two groups. "
                "This one has {}".format(len(unique_treatments))
            )
        # Update line:
        if kde:
            last_line = ax.get_lines()[-1]
            last_line.set_ydata(-1 * last_line.get_ydata())
        # Update histogram bars:
        idx_of_first_hist_rect = [patch.get_label() for patch in ax.patches].index(
            f"treatment = {unique_treatments[-1]}"
        )
        for patch in ax.patches[idx_of_first_hist_rect:]:
            patch.set_height(-1 * patch.get_height())

        # Re-set the view of axes:
        ax.relim()
        ax.autoscale()
        # Remove negation sign from lower y-axis:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda x, pos: str(x) if x >= 0 else str(-x)
            )
        )

    ax.legend(loc="best")
    x_type = (
        "Propensity" if propensity.between(0, 1, inclusive="both").all() else "Weights"
    )
    ax.set_xlabel(x_type)
    y_type = "Probability density" if norm_hist else "Counts"
    ax.set_ylabel(y_type)
    ax.set_title(f"{x_type} Distribution")
    return ax


def plot_propensity_score_distribution_folds(
    predictions,
    hue_by,
    cv,
    reflect=True,
    kde=False,
    cumulative=False,
    norm_hist=True,
    ax=None,
):
    """

    Args:
        predictions (list[pd.Series]):
        X (pd.DataFrame):
        hue_by (pd.Series):
        y (pd.Series):
        cv (list[np.array]):
        reflect (bool): Whether to plot second treatment group on the opposite sides of the x-axis.
                        This can only work if there are exactly two groups.
        kde (bool): Whether to plot kernel density estimation
        cumulative (bool): Whether to plot cumulative distribution.
        norm_hist (bool): If False - use raw counts on the y-axis.
                          If kde=True, then norm_hist should be True as well.
        ax (plt.Axis):

    Returns:

    """
    propensity = pd.concat(predictions)  # type: pd.Series
    # treatment = hue_by         # if train phase then there will  be no duplication of records.
    treatment = pd.concat([hue_by.iloc[fold_idx] for fold_idx in cv])  # type: pd.Series
    ax = plot_propensity_score_distribution(
        propensity,
        treatment,
        reflect=reflect,
        kde=kde,
        cumulative=cumulative,
        norm_hist=norm_hist,
        ax=ax,
    )
    return ax


def plot_mean_features_imbalance_love_folds(
    table1_folds,
    cv=None,
    aggregate_folds=True,
    thresh=None,
    plot_semi_grid=True,
    ax=None,
):
    method_pretty_name = {
        "smd": "Standard Mean Difference",
        "abs_smd": "Absolute Standard Mean Difference",
        "ks": "Kolmogorov-Smirnov",
    }
    ax = ax or plt.gca()

    # Aggregate across folds. This will be used to determine order, and extreme values.
    # Use this groupby trick: https://stackoverflow.com/a/25058102
    aggregated_table1 = pd.concat(table1_folds)  # type: pd.DataFrame
    aggregated_table1 = aggregated_table1.groupby(aggregated_table1.index)

    order = aggregated_table1.mean().sort_values(by="unweighted", ascending=True).index

    if aggregate_folds:
        # place in iterable to make compatible with input
        table1_folds = [aggregated_table1.mean()]

    # Plot:
    for table1 in table1_folds:
        color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        marker_cycle = cycle(["o", "^", "P", "s", "*"])
        for col_name, col_data in table1.items():
            col_data = col_data.loc[order]  # unified order
            ax.scatter(
                col_data,
                order,
                label=col_name,
                marker=next(marker_cycle),
                color=next(color_cycle),
                zorder=0,
            )

    # Plot line connecting the dots: (before plotting dots so they would be underneath the dots)
    if plot_semi_grid:
        if aggregate_folds:
            h_max = aggregated_table1.mean().max(axis="columns")
            h_min = aggregated_table1.mean().min(axis="columns")
        else:
            h_max = aggregated_table1.max().max(axis="columns")
            h_min = aggregated_table1.min().min(axis="columns")
        ax.hlines(
            h_min.index,
            xmin=h_min,
            xmax=h_max,  # ax.hlines(order, xmin=h_min, xmax=h_max,
            colors="grey",
            linestyles="dashed",
            zorder=1,
            label=None,
        )

    # Plot vertical threshold line
    if thresh is not None:
        ax.axvline(thresh, color="grey", linestyle="--", zorder=2)
        if aggregated_table1.min().min().min() < 0:
            # There are negative values, plot the minus of threshold and adjust x-limits to be symmetric:
            ax.axvline(-thresh, color="grey", linestyle="--", zorder=2)
            ax.set_xlim(-np.max(np.abs(ax.get_xlim())), np.max(np.abs(ax.get_xlim())))

    # # If too many features, remove their tick labels:
    fig = ax.get_figure()
    ax_pixel_height = (
        ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).height
        * fig.dpi
    )
    # 10 is hypothesized to be font size + 3 pt. margin
    if ax_pixel_height / order.size < 10 + 3:
        ax.set_yticklabels([])  # Too many y-ticks for axis size, remove them.

    ax.set_xlabel(
        method_pretty_name.get(
            table1_folds[0].columns.name, table1_folds[0].columns.name
        )
    )
    ax.set_ylabel("Covariates")
    ax.legend(loc="lower right")
    return ax


def plot_mean_features_imbalance_scatter_plot(
    table1_folds,
    aggregate_folds=True,
    thresh=None,
    label_imbalanced=True,
    ax=None,
):
    # get current axes
    ax = ax or plt.gca()

    method_pretty_name = {
        "smd": "Standard Mean Difference",
        "abs_smd": "Absolute Standard Mean Difference",
        "ks": "Kolmogorov-Smirnov",
    }
    # Aggregate across folds. This will be used to determine order, and extreme values.
    # Use this groupby trick: https://stackoverflow.com/a/25058102
    aggregated_table1 = pd.concat(table1_folds)  # type: pd.DataFrame
    aggregated_table1 = aggregated_table1.groupby(aggregated_table1.index)

    if aggregate_folds:
        table1_folds = [aggregated_table1.mean()]
    
    # Plot:

    for table1 in table1_folds:
        
        # setting different marker shapes for each fold in aggregated_foldes == False
        marker_cycle = cycle(["o", "^", "P", "s", "*"]) 

        # find index of features that are above threshold 
        violating = table1["weighted"] > thresh
        # determain color for dot on plot 
        color = violating.replace({False: "C0", True: "C1"})

        ax.scatter( 
            x=table1['unweighted'],
            y=table1['weighted'],
            marker=next(marker_cycle),
            color=color
        )
        if label_imbalanced:
            for covariate_name, covariate_diff in table1.loc[violating].iterrows():
                ax.text(
                    x=covariate_diff["unweighted"],
                    y=covariate_diff["weighted"],
                    s=covariate_name,
                    horizontalalignment="left",
                )
            
    # Plot vertical and horizontal threshold line
    if thresh is not None:
        ax.axvline(thresh, color="grey", linestyle="--", zorder=2)
        ax.axhline(thresh, color="grey", linestyle="--", zorder=2)
        # There are negative values, plot the minus of threshold   
        if aggregated_table1.min().min().min() < 0:
            # There are negative values, plot the minus of threshold and adjust x-limits to be symmetric:
            ax.axvline(-thresh, color="grey", linestyle="--", zorder=2)
            ax.axhline(-thresh, color="grey", linestyle="--", zorder=2)
            ax.set_xlim(-np.max(np.abs(ax.get_xlim())), np.max(np.abs(ax.get_xlim())))

    
    # adding labels 

    metric_name = table1_folds[0].columns.name
    metric_name = method_pretty_name.get(metric_name, metric_name)

    ax.set_xlabel(f'Unweighted [{metric_name}]')
    ax.set_ylabel(f'Weighted [{metric_name}]')

    return ax 


def plot_mean_features_imbalance_slope_folds(
    table1_folds, cv=None, thresh=None, label_imbalanced=True, ax=None
):
    method_pretty_name = {
        "smd": "Standard Mean Difference",
        "abs_smd": "Absolute Standard Mean Difference",
        "ks": "Kolmogorov-Smirnov",
    }
    # ax = ax or plt.gca()

    # Aggregate across folds. This will be used to determine order, and extreme values.
    # Use this groupby trick: https://stackoverflow.com/a/25058102
    aggregated_table1 = pd.concat(table1_folds)  # type: pd.DataFrame
    aggregated_table1 = aggregated_table1.groupby(aggregated_table1.index)
    aggregated_table1 = aggregated_table1.mean()

    # Reorder:
    aggregated_table1 = aggregated_table1.sort_values(by="unweighted", ascending=True)

    # Slope graph:
    ax = slope_graph(
        left=aggregated_table1["unweighted"],
        right=aggregated_table1["weighted"],
        thresh=thresh,
        label_imbalanced=label_imbalanced,
        ax=ax,
    )

    ax.set_ylabel(
        method_pretty_name.get(
            table1_folds[0].columns.name, table1_folds[0].columns.name
        )
    )
    # ax.legend(loc="upper right")
    return ax


def slope_graph(
    left, right, thresh=None, label_imbalanced=True, color_below="C0", color_above="C1", marker="o", ax=None
):
    ax = ax or plt.gca()
    left_xtick = left.name or "unweighted"
    right_xtick = right.name or "weighted"

    if thresh is not None:
        ax.axhline(thresh, color="grey", linestyle="--", zorder=2)
        # There are negative values, plot the minus of threshold
        if left.min() < 0 or right.min() < 0:
            ax.axhline(-thresh, color="grey", linestyle="--", zorder=2)
    else:
        thresh = np.nan  # will be now used to compare against values

    for idx in left.index:
        cur_left = left[idx]
        cur_right = right[idx]
        # make default color_below if thresh is nan
        cur_color = color_above if cur_right > thresh else color_below

        ax.plot(
            [left_xtick, right_xtick],
            [cur_left, cur_right],
            label=None,
            color=cur_color,
            marker=marker,
        )
        if label_imbalanced and cur_right > thresh:
            ax.text(x=1.01, y=cur_right, s=idx, horizontalalignment="left")

    # Place y-tick labels on both sides:
    ax.tick_params(left=True, labelleft=True, right=True, labelright=True)
    return ax


def get_subplots(n_features, max_cols=5, fig_size=(16, 16), sharex=False, sharey=False):
    """Initializes the grid of subplots and returns the axes

    Args:
        n_features (int): The total number of features to plot
        max_cols (int): The maximal number of figures in each row of figures
        fig_size (tuple[int, int]): Passed on to matplotlib
        sharex (str|bool): will be passed to subplots
        sharey (str|bool): will be passed to subplots

    Returns:
        tuple[Figure, np.ndarray]: the figure and the array of axes

    """
    # try to make the plots as square as possible
    ncols = min(int(np.round(np.sqrt(n_features))), max_cols)
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=fig_size,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
    )
    return fig, axes


def _add_diagonal(
    ax, fraction=0.04, label="x=y", color="grey", linestyle="--", zorder=1
):
    diagonal = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    # ax.plot(diagonal, diagonal, color="black", label="x=y")  # plot diagonal
    lim_range_frac = (
        np.array([np.diff(ax.get_xlim()), np.diff(ax.get_ylim())]) * fraction
    )
    while np.any(np.abs(np.diff(diagonal)[0]) < lim_range_frac):
        if np.abs(np.diff(diagonal)[0]) < lim_range_frac[0]:
            ax.set_ylim(
                *(
                    ax.get_ylim()
                    + np.diff(ax.get_ylim()) * [-fraction / 2, fraction / 2]
                )
            )
        if np.abs(np.diff(diagonal)[0]) < lim_range_frac[1]:
            ax.set_xlim(
                *(
                    ax.get_xlim()
                    + np.diff(ax.get_xlim()) * [-fraction / 2, fraction / 2]
                )
            )
        diagonal = [
            max(ax.get_xlim()[0], ax.get_ylim()[0]),
            min(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        # ax.plot(diagonal, diagonal, color="black")  # extend diagonal
        lim_range_frac = (
            np.array([np.diff(ax.get_xlim()), np.diff(ax.get_ylim())]) * fraction
        )

    # plot diagonal
    ax.plot(
        diagonal, diagonal, color=color, label=label, linestyle=linestyle, zorder=zorder
    )


WEIGHT_PLOTS = {
    "weight_distribution": plot_propensity_score_distribution_folds,
    "covariate_balance_love": plot_mean_features_imbalance_love_folds,
    "covariate_balance_slope": plot_mean_features_imbalance_slope_folds,
}

OUTCOME_PLOTS = {
    "continuous_accuracy": plot_continuous_prediction_accuracy_folds,
    "residuals": plot_residual_folds,
    "common_support": plot_counterfactual_common_support_folds,
}

SHARED_PLOTS = {
    "roc_curve": plot_roc_curve_folds,
    "pr_curve": plot_precision_recall_curve_folds,
    "calibration": plot_calibration_folds,
}
OUTCOME_PLOTS.update(SHARED_PLOTS)
WEIGHT_PLOTS.update(SHARED_PLOTS)
