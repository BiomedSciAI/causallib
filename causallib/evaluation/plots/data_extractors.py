"""Plot data extractors.

The responsibility of these classes is to extract the data from the
EvaluationResults objects to match the requested plot.
"""

import abc

from sklearn import metrics

from . import curve_data_makers, plots
from ...metrics.weight_metrics import calculate_covariate_balance


class BaseEvaluationPlotDataExtractor(abc.ABC):
    """Extractor to get plot data from EvaluationResults.

    Subclasses also have a `plot_names` property.
    """

    def __init__(
        self, evaluation_results: "causallib.evaluation.results.EvaluationResults"
    ):
        self.predictions = evaluation_results.predictions
        self.X = evaluation_results.X
        self.a = evaluation_results.a
        self.y = evaluation_results.y
        self.cv = evaluation_results.cv

    def cv_by_phase(self, phase="train"):
        """Get the cross-validation indices of all folds for a given phase.

        Args:
            phase (str, optional): Requested phase: "train" or "valid. Defaults to "train".

        Returns:
            List: _description_
        """
        fold_idx = 0 if phase == "train" else 1
        return [fold[fold_idx] for fold in self.cv]

    @abc.abstractmethod
    def get_data_for_plot(self, plot_name, phase="train"):
        """Get data for plot with name `plot_name`."""
        raise NotImplementedError


class WeightPlotDataExtractor(BaseEvaluationPlotDataExtractor):
    """Extractor to get plot data from WeightEvaluatorPredictions."""

    plot_names = plots.WeightPlotNames

    def get_data_for_plot(self, plot_name, phase="train"):
        """Retrieve the data needed for each provided plot.

        Plot functions are in plots module.

        Args:
            plot_name (str): Plot name.

        Returns:
            tuple: Plot data
        """

        folds_predictions = self.predictions[phase]
        if plot_name in {plots.WEIGHT_DISTRIBUTION_PLOT}:
            return (
                [p.weight_for_being_treated for p in folds_predictions],
                self.a,
                self.cv_by_phase(phase),
            )
        if plot_name in {
            plots.COVARIATE_BALANCE_LOVE_PLOT,
            plots.COVARIATE_BALANCE_SLOPE_PLOT,
            plots.COVARIATE_BALANCE_GENERIC_PLOT,
        }:
            distribution_distances = []
            for fold_prediction in folds_predictions:
                fold_w = fold_prediction.weight_by_treatment_assignment
                fold_X = self.X.loc[fold_w.index]
                fold_a = self.a.loc[fold_w.index]
                dist_dist = calculate_covariate_balance(fold_X, fold_a, fold_w)
                distribution_distances.append(dist_dist)
            return (distribution_distances,)

        raise ValueError(f"Received unsupported plot name {plot_name}!")


class PropensityPlotDataExtractor(WeightPlotDataExtractor):
    """Extractor to get plot data from PropensityEvaluatorPredictions."""

    plot_names = plots.PropensityPlotNames

    def get_data_for_plot(self, plot_name, phase="train"):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots.py module.

        Args:
            plot_name (str): Plot name.
            fold_predictions (list[PropensityEvaluatorPredictions]): Predictions for each fold.
            cv list[np.ndarray]: Indices (in iloc positions) of each fold.

        Returns:
            tuple: Plot data
        """
        fold_predictions = self.predictions[phase]

        if plot_name in {plots.WEIGHT_DISTRIBUTION_PLOT, plots.CALIBRATION_PLOT}:
            return (
                [p.propensity for p in fold_predictions],
                self.a,
                self.cv_by_phase(phase),
            )
        if plot_name in {plots.ROC_CURVE_PLOT}:
            curve_data = curve_data_makers.calculate_curve_data_propensity(
                fold_predictions, self.a, metrics.roc_curve, metrics.roc_auc_score
            )
            roc_curve = curve_data_makers.calculate_roc_curve(curve_data)
            return (roc_curve,)
        if plot_name in {plots.PR_CURVE_PLOT}:
            curve_data = curve_data_makers.calculate_curve_data_propensity(
                fold_predictions,
                self.a,
                metrics.precision_recall_curve,
                metrics.average_precision_score,
            )
            pr_curve = curve_data_makers.calculate_pr_curve(curve_data, self.a)
            return (pr_curve,)

        # Common plots are implemented at top-most level possible.
        # Plot might be implemented by WeightEvaluator:
        return super().get_data_for_plot(plot_name, phase=phase)


class ContinuousOutcomePlotDataExtractor(BaseEvaluationPlotDataExtractor):
    """Extractor to get plot data from OutcomeEvaluatorPredictions.

    Note that the available plots are different if the outcome predictions
    are binary/classification or continuous/regression.
    """

    plot_names = plots.ContinuousOutputPlotNames

    def get_data_for_plot(self, plot_name, phase="train"):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots module.

        Args:
            plot_name (str): Plot name.

        Returns:
            tuple: Plot data
        """
        fold_predictions = self.predictions[phase]
        if plot_name in {plots.CONTINUOUS_ACCURACY_PLOT, plots.RESIDUALS_PLOT}:
            return (
                [x.get_prediction_by_treatment(self.a) for x in fold_predictions],
                self.y,
                self.a,
                self.cv_by_phase(phase),
            )
        if plot_name in {plots.COMMON_SUPPORT_PLOT}:
            return (
                [p.prediction for p in fold_predictions],
                self.a,
                self.cv_by_phase(phase),
            )

        raise ValueError(f"Received unsupported plot name {plot_name}!")


class BinaryOutcomePlotDataExtractor(BaseEvaluationPlotDataExtractor):
    """Extractor to get plot data from OutcomeEvaluatorPredictions.

    Note that the available plots are different if the outcome predictions
    are binary/classification or continuous/regression.
    """

    plot_names = plots.BinaryOutputPlotNames

    def get_data_for_plot(self, plot_name, phase="train"):
        """Retrieve the data needed for each provided plot.
        Plot interfaces are at the plots module.

        Args:
            plot_name (str): Plot name.

        Returns:
            tuple: Plot data
        """
        fold_predictions = self.predictions[phase]

        if plot_name in {plots.CALIBRATION_PLOT}:
            return (
                [x.get_proba_by_treatment(self.a) for x in fold_predictions],
                self.y,
                self.cv_by_phase(phase),
            )
        if plot_name in {plots.ROC_CURVE_PLOT}:
            proba_list = [x.get_proba_by_treatment(self.a) for x in fold_predictions]
            curve_data = curve_data_makers.calculate_curve_data_binary_outcome(
                proba_list,
                self.y,
                metrics.roc_curve,
                metrics.roc_auc_score,
                stratify_by=self.a,
            )
            roc_curve_data = curve_data_makers.calculate_roc_curve(curve_data)
            return (roc_curve_data,)

        if plot_name in {plots.PR_CURVE_PLOT}:
            proba_list = [x.get_proba_by_treatment(self.a) for x in fold_predictions]
            curve_data = curve_data_makers.calculate_curve_data_binary_outcome(
                proba_list,
                self.y,
                metrics.precision_recall_curve,
                metrics.average_precision_score,
                stratify_by=self.a,
            )
            pr_curve_data = curve_data_makers.calculate_pr_curve(curve_data, self.y)
            return (pr_curve_data,)

        raise ValueError(f"Received unsupported plot name {plot_name}!")
