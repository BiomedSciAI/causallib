"""Mixins for plotting.

To work the mixin requires the class to implement `get_data_for_plot` with the
supported plot names. See .data_extractors for examples. """

from . import plots


class WeightPlotterMixin:
    """Mixin to add members to for weight estimation plotting.

    Class must implement:
      * `get_data_for_plot(plots.COVARIATE_BALANCE_GENERIC_PLOT)`
      * `get_data_for_plot(plots.WEIGHT_DISTRIBUTION_PLOT)`
    """

    def plot_covariate_balance(
        self,
        kind="love",
        phase="train",
        ax=None,
        aggregate_folds=True,
        thresh=None,
        plot_semi_grid=True,
        label_imbalanced=True,
        **kwargs,
    ):
        """Plot covariate balance before and after weighting.

        Args:
            kind (str, optional): Plot kind, "love" ,"slope" or "scatter". Defaults to "love".
            phase (str, optional): Phase to plot: "train" or "valid". Defaults to "train".
            ax (matplotlib.axes.Axes, optional): axis to plot on, if None creates new axis.
                Defaults to None.
            aggregate_folds (bool, optional): Whether to aggregate folds. Defaults to True.
                Ignored when kind="slope".
            thresh (float, optional): Draw threshold line at value. Defaults to None.
            plot_semi_grid (bool, optional): Defaults to True. only for kind="love".
            label_imbalanced (bool): Label covariates that weren't properly balanced. Ignored when kind="love".

        Returns:
            matplotlib.axes.Axes: axis with plot
        """
        (table1_folds,) = self.get_data_for_plot(
            plots.COVARIATE_BALANCE_GENERIC_PLOT, phase=phase
        )
        if kind == "love":
            return plots.plot_mean_features_imbalance_love_folds(
                table1_folds=table1_folds,
                ax=ax,
                aggregate_folds=aggregate_folds,
                thresh=thresh,
                plot_semi_grid=plot_semi_grid,
                **kwargs,
            )
        if kind == "slope":
            return plots.plot_mean_features_imbalance_slope_folds(
                table1_folds=table1_folds,
                ax=ax,
                thresh=thresh,
                label_imbalanced=label_imbalanced,
                **kwargs,
            )

        if kind == "scatter":
            return plots.plot_mean_features_imbalance_scatter_plot(
                table1_folds=table1_folds,
                ax=ax,
                thresh=thresh,
                label_imbalanced=label_imbalanced,
                **kwargs,
            )

        raise ValueError(f"Unsupported covariate balance plot kind {kind}")

    def plot_weight_distribution(
        self,
        phase="train",
        reflect=True,
        kde=False,
        cumulative=False,
        norm_hist=True,
        ax=None,
    ):
        """
        Plot the distribution of propensity score.

        Args:
            phase (str, optional): Phase to plot: "train" or "valid". Defaults to "train".
            reflect (bool): Whether to plot treatment groups on opposite sides of the x-axis.
                This can only work if there are exactly two groups.
            kde (bool): Whether to plot kernel density estimation
            cumulative (bool): Whether to plot cumulative distribution.
            norm_hist (bool): If False - use raw counts on the y-axis.
                            If kde=True, then norm_hist should be True as well.
            ax (matplotlib.axes.Axes, optional): axis to plot on, if None creates new axis.
                Defaults to None.
        Returns:
            matplotlib.axes.Axes

        """
        weights, treatments, cv = self.get_data_for_plot(
            plots.WEIGHT_DISTRIBUTION_PLOT, phase=phase
        )

        return plots.plot_propensity_score_distribution_folds(
            predictions=weights,
            hue_by=treatments,
            cv=cv,
            reflect=reflect,
            kde=kde,
            cumulative=cumulative,
            norm_hist=norm_hist,
            ax=ax,
        )


class ClassificationPlotterMixin:
    """Mixin to add members to for classification/binary prediction estimation.

    This occurs for propensity models (treatment assignment is inherently binary)
    and for outcome models where the outcome is binary.

    Class must implement:
      * `get_data_for_plot(plots.ROC_CURVE_PLOT)`
      * `get_data_for_plot(plots.PR_CURVE_PLOT)`
      * `get_data_for_plot(plots.CALIBRATION_PLOT)`
    """

    def plot_roc_curve(
        self,
        phase="train",
        plot_folds=False,
        label_folds=False,
        label_std=False,
        ax=None,
    ):
        """Plot ROC curve.

        Args:
            phase (str, optional): Phase to plot: "train" or "valid". Defaults to "train".
            plot_folds (bool, optional): Whether to plot individual folds. Defaults to False.
            label_folds (bool, optional): Whether to label folds. Defaults to False.
            label_std (bool, optional): Whether to label std. Defaults to False.
            ax (matplotlib.axes.Axes, optional): axis to plot on, if None creates new axis.
                Defaults to None.
        Returns:
            matplotlib.axes.Axes
        """
        (roc_curve_data,) = self.get_data_for_plot(plots.ROC_CURVE_PLOT, phase=phase)
        return plots.plot_roc_curve_folds(
            roc_curve_data,
            ax=ax,
            plot_folds=plot_folds,
            label_folds=label_folds,
            label_std=label_std,
        )

    def plot_pr_curve(
        self,
        phase="train",
        plot_folds=False,
        label_folds=False,
        label_std=False,
        ax=None,
    ):
        """Plot precision-recall (PR) curve.

        Args:
            phase (str, optional): Phase to plot: "train" or "valid". Defaults to "train".
            plot_folds (bool, optional): Whether to plot individual folds. Defaults to False.
            label_folds (bool, optional): Whether to label folds. Defaults to False.
            label_std (bool, optional): Whether to label std. Defaults to False.
            ax (matplotlib.axes.Axes, optional): axis to plot on, if None creates new axis.
                Defaults to None.
        Returns:
            matplotlib.axes.Axes
        """
        (pr_curve_data,) = self.get_data_for_plot(plots.PR_CURVE_PLOT, phase=phase)
        return plots.plot_precision_recall_curve_folds(
            pr_curve_data,
            ax=ax,
            plot_folds=plot_folds,
            label_folds=label_folds,
            label_std=label_std,
        )

    def plot_calibration_curve(
        self,
        phase="train",
        n_bins=10,
        plot_se=True,
        plot_rug=False,
        plot_histogram=False,
        quantile=False,
        ax=None,
    ):
        """Plot calibration curves for multiple models (presumably in folds)

        Args:
            phase (str, optional): Phase to plot: "train" or "valid". Defaults to "train".
            n_bins (int): number of bins to evaluate in the plot
            plot_se (bool): Whether to plot standard errors around the mean
                bin-probability estimation.
            plot_rug (bool):
            plot_histogram (bool):
            quantile (bool): If true, the binning of the calibration curve is by quantiles.
                Defaults to False.
            ax (matplotlib.axes.Axes, optional): axis to plot on, if None creates new axis.
                Defaults to None.
        Note:
            One of plot_propensity or plot_model must be True.

        Returns:
            matplotlib.axes.Axes
        """

        predictions, targets, cv = self.get_data_for_plot(
            plots.CALIBRATION_PLOT, phase=phase
        )
        return plots.plot_calibration_folds(
            predictions=predictions,
            targets=targets,
            cv=cv,
            n_bins=n_bins,
            plot_se=plot_se,
            plot_rug=plot_rug,
            plot_histogram=plot_histogram,
            quantile=quantile,
            ax=ax,
        )


class ContinuousOutcomePlotterMixin:
    """Mixin to add members to for continous outcome estimation.

    Class must implement:
      * `get_data_for_plot(plots.CONTINUOUS_ACCURACY_PLOT)`
      * `get_data_for_plot(plots.RESIDUALS_PLOT)`
      * `get_data_for_plot(plots.CONTINUOUS_ACCURACY_PLOT)`
    """

    def plot_continuous_accuracy(
        self, phase="train", alpha_by_density=True, plot_residuals=False, ax=None
    ):
        """Plot continuous accuracy,

        Args:
            phase (str, optional): Phase to plot: "train" or "valid". Defaults to "train".
            alpha_by_density (bool, optional): Whether to calculate points alpha value
                (transparent-opaque) with density estimation. This can take some time
                to compute for a large number of points. If False, alpha calculation
                will be a simple fast heuristic.
            plot_residuals (bool, optional): Whether to plot residuals. Defaults to False.
            ax (matplotlib.axes.Axes, optional): axis to plot on, if None creates new axis.
                Defaults to None.
        Returns:
            matplotlib.axes.Axes
        """
        predictions, y, a, cv = self.get_data_for_plot(
            plots.CONTINUOUS_ACCURACY_PLOT, phase=phase
        )
        return plots.plot_continuous_prediction_accuracy_folds(
            predictions=predictions,
            y=y,
            a=a,
            cv=cv,
            alpha_by_density=alpha_by_density,
            plot_residuals=plot_residuals,
            ax=ax,
        )

    def plot_residuals(self, phase="train", alpha_by_density=True, ax=None):
        """Plot residuals of predicted outcome vs ground truth.

        Args:
            phase (str, optional): Phase to plot: "train" or "valid". Defaults to "train".
            alpha_by_density (bool, optional): Whether to calculate points alpha value
                (transparent-opaque) with density estimation. This can take some time
                to compute for a large number of points. If False, alpha calculation
                will be a simple fast heuristic.
            ax (matplotlib.axes.Axes, optional): axis to plot on, if None creates new axis.
                Defaults to None.
        Returns:
            matplotlib.axes.Axes
        """
        predictions, y, a, cv = self.get_data_for_plot(
            plots.RESIDUALS_PLOT, phase=phase
        )

        return plots.plot_residual_folds(
            predictions=predictions,
            y=y,
            a=a,
            cv=cv,
            alpha_by_density=alpha_by_density,
            ax=ax,
        )

    def plot_common_support(self, phase="train", alpha_by_density=True, ax=None):
        """Plot the scatter plot of y0 vs. y1 for multiple scoring results, colored by the treatment

        Args:
            alpha_by_density (bool): Whether to calculate points alpha value (transparent-opaque)
               with density estimation. This can take some time to compute for a large number
               of points. If False, alpha calculation will be a simple fast heuristic.
            ax (plt.Axes): The axes on which the plot will be displayed. Optional.
        """
        predictions, treatments, cv = self.get_data_for_plot(
            plots.COMMON_SUPPORT_PLOT, phase=phase
        )
        return plots.plot_counterfactual_common_support_folds(
            predictions=predictions,
            hue_by=treatments,
            cv=cv,
            alpha_by_density=alpha_by_density,
            ax=ax,
        )


class PlotAllMixin:
    """Mixin to make all the train and validation plots.

    Class must implement:
      * `all_plot_names`
      * `get_data_for_plot(name)` for every name in `all_plot_names`
    """

    def plot_all(self, phase=None):
        """Create plot of all available EvaluationResults.

        Will create a figure with a subplot for each plot name in `all_plot_names`.
        If `results` have train and validation data, will create separate
        "train" and "valid" figures. If a single plot is requested, only that plot is created.

        Args:
            phase (Union[str, None], optional): phase to plot "train" or "valid". If not supplied,
                defaults to both if available.

        Returns:
            Dict[str, matplotlib.axis.Axis]]: the Axis objects of the plots in a nested dictionary:
              * First key is the phase ("train" or "valid")
              * Second key is the plot name.
        """
        phases_to_plot = self.predictions.keys() if phase is None else [phase]
        multipanel_plot = {
            plotted_phase: self._make_multipanel_evaluation_plot(
                plot_names=self.all_plot_names, phase=plotted_phase
            )
            for plotted_phase in phases_to_plot
        }
        return multipanel_plot

    def _make_multipanel_evaluation_plot(self, plot_names, phase):
        phase_fig, phase_axes = plots.get_subplots(len(plot_names))
        named_axes = {
            name: self._make_single_panel_evaluation_plot(name, phase, ax)
            for name, ax in zip(plot_names, phase_axes.ravel())
        }

        phase_fig.suptitle(f"Evaluation on {phase} phase")
        return named_axes

    def _make_single_panel_evaluation_plot(self, plot_name, phase, ax=None, **kwargs):
        """Create a single evaluation plot.

        For a single phase and a single plot name.

        Args:
            results (EvaluationResults): evaluation results to plot
            plot_name (str): plot name (from results.all_plot_names)
            phase (str): "train" or "valid"
            ax (matplotlib.axis.Axis, optional): axis to plot on. Defaults to None.
            **kwargs: passed to underlying plotting function

        Returns:
            Union[matplotlib.axis.Axis, None]: axis with plot if successful, else None
        """

        plot_func = plots.lookup_name(plot_name)
        plot_data = self.get_data_for_plot(plot_name, phase=phase)
        return plot_func(*plot_data, ax=ax, **kwargs)
