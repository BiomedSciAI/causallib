import numpy as np
import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from causallib.datasets.data_loader import load_nhefs_survival
from causallib.survival.survival_utils import get_person_time_df, safe_join
from causallib.estimation.ipw import IPW
from causallib.survival.standardized_survival import StandardizedSurvival
from causallib.survival.weighted_standardized_survival import WeightedStandardizedSurvival
from causallib.survival.weighted_survival import WeightedSurvival
from causallib.survival.marginal_survival import MarginalSurvival
try:
    import lifelines
    LIFELINES_FOUND = True
except ImportError:
    LIFELINES_FOUND = False

"""
Simulated Test Data For Drug Effects
-------------------------------------
Drug A is simulated with a beneficial effect, drug B with zero effect.
x_0 is a dummy variable that has no effect on treatment nor on the outcome.
Drug C involves informative censoring, so don't expect accurate results
with censoring-naive models.

Data generation and fitting yielded the following:
  IPW-LR S diff IPW-LR effect S-LR S diff S-LR effect T-LR S diff T-LR effect  oracle effect
A          0.24          0.45        0.31        0.43        0.34        0.49           0.41
B         -0.39        -0.077       -0.38       -0.01       -0.39     -0.0069          -0.06
C          0.34           0.5        0.27        0.26         0.3        0.35           0.56

Where -
* S diff = difference in observed survival curves at end of follow-up
* effect = difference in adjusted survival curves at the end of follow-up
* IPW-LR = inverse propensity weighting with logistic regression
* S-LR = standardization with logistic regression (non-stratified)

"""

RANDOM_SEED = 1
TEST_DATA_TTE_DRUG_EFFECTS = dict()
TEST_DATA_TTE_DRUG_EFFECTS['A'] = pd.DataFrame(
    columns=['x_0', 'x_1', 'a', 'y', 't'],
    data=[
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 3],
        [0, 0, 0, 1, 2],
        [0, 0, 1, 0, 2],
        [0, 0, 0, 1, 2],
        [0, 1, 1, 0, 2],
        [0, 0, 1, 0, 3],
        [0, 1, 1, 1, 2],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 1, 2],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 1],
        [1, 0, 0, 0, 2],
        [0, 0, 0, 0, 3],
        [1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 1, 2],
        [0, 1, 1, 1, 3],
        [1, 1, 1, 1, 2],
        [1, 0, 0, 1, 1],
        [0, 0, 0, 0, 2],
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 0, 2],
        [0, 1, 1, 0, 1],
        [1, 1, 1, 0, 2],
        [0, 1, 1, 0, 2],
        [0, 1, 0, 1, 1],
        [1, 1, 1, 1, 2],
        [1, 0, 0, 1, 2],
        [1, 0, 1, 0, 3],
        [0, 1, 1, 1, 2],
        [1, 0, 1, 0, 2],
        [1, 1, 1, 1, 2],
        [0, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 0, 0, 2],
        [1, 1, 1, 0, 3],
        [0, 0, 0, 1, 2],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 0, 3],
        [0, 0, 0, 0, 2],
        [0, 1, 1, 1, 1]
    ]
)
TEST_DATA_TTE_DRUG_EFFECTS['B'] = pd.DataFrame(
    columns=['x_0', 'x_1', 'a', 'y', 't'],
    data=[
        [0, 0, 0, 0, 3],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 3],
        [0, 0, 1, 1, 2],
        [0, 0, 1, 1, 2],
        [0, 0, 0, 0, 3],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 3],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 3],
        [1, 0, 0, 0, 3],
        [0, 1, 0, 1, 1],
        [1, 0, 0, 1, 2],
        [0, 0, 0, 1, 3],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 1, 3],
        [0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 1, 2],
        [1, 0, 1, 0, 3],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 1, 3],
        [0, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 1, 3],
        [1, 0, 0, 1, 2],
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 3],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 1, 1, 3],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 3],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 3],
        [0, 0, 0, 1, 3],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 1, 2],
        [0, 1, 1, 1, 1]
    ]
)

TEST_DATA_DRUG_EFFECTS_A_OBSERVED_DIFF = 0.24
TEST_DATA_DRUG_EFFECTS_A_ORACLE_DIFF = 0.41
TEST_DATA_DRUG_EFFECTS_B_OBSERVED_DIFF = -0.40
TEST_DATA_DRUG_EFFECTS_B_ORACLE_DIFF = -0.06
TEST_DATA_DRUG_EFFECTS_DELTA = 0.15
TEST_DATA_DRUG_EFFECTS_FOLLOWUP_LAST_TIMESTEP = 2
TEST_DATA_PERSON_TIME_LAST_STEP_HAZARD_0 = 0.38
TEST_DATA_PERSON_TIME_LAST_STEP_HAZARD_1 = 0.28
TEST_DATA_PERSON_TIME_LAST_STEP_SURVIVAL_0 = 0.33
TEST_DATA_PERSON_TIME_LAST_STEP_SURVIVAL_1 = 0.58

"""
Test Data for Person-Time Format Conversion
-------------------------------------------
"""
TEST_DATA_PERSON_TIME_INPUT = pd.DataFrame(
    columns=['id', 'age', 'height', 'a', 'y', 't'],
    data=[
        [1, 22, 170, 0, 1, 2],
        [2, 40, 180, 1, 0, 1],
        [3, 30, 165, 1, 0, 2]
    ],
).set_index('id')

TEST_DATA_PERSON_TIME_OUTPUT = pd.DataFrame(
    columns=['id', 'age', 'height', 'a', 'y', 't'],
    data=[
        [1, 22, 170, 0, 0, 0],
        [1, 22, 170, 0, 0, 1],
        [1, 22, 170, 0, 1, 2],
        [2, 40, 180, 1, 0, 0],
        [2, 40, 180, 1, 0, 1],
        [3, 30, 165, 1, 0, 0],
        [3, 30, 165, 1, 0, 1],
        [3, 30, 165, 1, 0, 2]
    ],
).set_index('id')


def fit_synthetic_data(model_cls, params, test_data):
    """
    Test util that runs the following steps:
    1. Initializes a causal survival model.
    2. Fits simulated test data.
    3. Estimates adjusted diff in survival curves between two treatment groups.
    Args:
        model_cls:  causal survival class reference
        params: constructor params
        test_data: DataFrame with covariates 'x_0', 'x_1', treatment assignment 'a', outcome 'y' and time to event 't'

    Returns:
        Adjusted diff in survival curves at last time step
    """
    model = model_cls(**params)
    X = test_data[['x_0', 'x_1']]
    a = test_data['a']
    y = test_data['y']
    t = test_data['t']
    model.fit(X=X, a=a, t=t, y=y)

    # Generate survival curves
    adjusted_curves = model.estimate_population_outcome(X=X, a=a, t=t, y=y)
    adjusted_diff = adjusted_curves[1][TEST_DATA_DRUG_EFFECTS_FOLLOWUP_LAST_TIMESTEP] - adjusted_curves[0][
        TEST_DATA_DRUG_EFFECTS_FOLLOWUP_LAST_TIMESTEP]

    return adjusted_diff


class TestUtils(unittest.TestCase):
    """"
    Test utility methods for causal survival analysis (data handling, etc.)
    """
    def setUp(self) -> None:
        np.random.seed(RANDOM_SEED)

    def test_person_time_expansion(self):
        res = get_person_time_df(a=TEST_DATA_PERSON_TIME_INPUT['a'],
                                 t=TEST_DATA_PERSON_TIME_INPUT['t'],
                                 y=TEST_DATA_PERSON_TIME_INPUT['y'],
                                 X=TEST_DATA_PERSON_TIME_INPUT[['age', 'height']])
        pd.testing.assert_frame_equal(res, TEST_DATA_PERSON_TIME_OUTPUT, check_dtype=False)

    def test_safe_join(self):
        X = TEST_DATA_PERSON_TIME_INPUT.copy()
        uniquely_named_series = [pd.Series(data=0, name='new_col1', index=X.index),
                                 pd.Series(data=0, name='new_col2', index=X.index)]
        duplicate_named_series = [pd.Series(data=0, name='new_col', index=X.index),
                                  pd.Series(data=0, name='new_col', index=X.index)]
        duplicate_in_df_named_series = [pd.Series(data=0, name='age', index=X.index),
                                        pd.Series(data=0, name='height', index=X.index)]

        # Unique Series name - should not rename
        # =======================================
        # Series only
        res, new_names = safe_join(df=None, list_of_series=uniquely_named_series, return_series_names=True)
        self.assertEqual(list(res.columns), [s.name for s in uniquely_named_series])  # res is concat of Series
        self.assertEqual(new_names, [s.name for s in uniquely_named_series])  # Series names UNCHANGED
        self.assertEqual(res.shape, (X.shape[0], len(uniquely_named_series)))

        # DataFrame + Series
        res, new_names = safe_join(df=X, list_of_series=uniquely_named_series, return_series_names=True)
        self.assertEqual(list(res.columns)[:X.shape[1]], list(X.columns))  # DataFrame col names UNCHANGED
        self.assertEqual(new_names, [s.name for s in uniquely_named_series])  # Series names UNCHANGED
        self.assertEqual(res.shape, (X.shape[0], X.shape[1] + len(uniquely_named_series)))

        # Duplicate Series name - should add random suffix
        # ================================================
        # Series only
        res, new_names = safe_join(df=None, list_of_series=duplicate_named_series, return_series_names=True)
        self.assertNotEqual(new_names, [s.name for s in duplicate_named_series])  # Series names CHANGED
        self.assertEqual(res.shape, (X.shape[0], len(duplicate_named_series)))

        # DataFrame + Series
        res, new_names = safe_join(df=X, list_of_series=duplicate_named_series, return_series_names=True)
        self.assertEqual(list(res.columns)[:X.shape[1]], list(X.columns))  # DataFrame col names UNCHANGED
        self.assertNotEqual(new_names, [s.name for s in duplicate_named_series])  # Series names CHANGED
        self.assertEqual(res.shape, (X.shape[0], X.shape[1] + len(duplicate_named_series)))

        # DataFrame + Series with names already in DataFrame
        res, new_names = safe_join(df=X, list_of_series=duplicate_in_df_named_series, return_series_names=True)
        self.assertEqual(list(res.columns)[:X.shape[1]], list(X.columns))  # DataFrame col names UNCHANGED
        self.assertNotEqual(new_names, [s.name for s in duplicate_in_df_named_series])
        self.assertEqual(res.shape, (X.shape[0], X.shape[1] + len(duplicate_in_df_named_series)))


class TestNHEFS(unittest.TestCase):
    """
    Test multiple models on the NHEFS dataset.
    Details: https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
    """
    def setUp(self) -> None:
        np.random.seed(RANDOM_SEED)
        data = load_nhefs_survival()
        self.X, self.a, self.t, self.y = data.X, data.a, data.t, data.y

        # Init various multiple models
        self.estimators = {
            'observed_non_parametric': MarginalSurvival(),
            'observed_parametric': MarginalSurvival(survival_model=LogisticRegression(max_iter=2000)),
            'ipw_non_parametric': WeightedSurvival(
                weight_model=IPW(LogisticRegression(max_iter=4000), use_stabilized=True),
                survival_model=None),
            'ipw_parametric': WeightedSurvival(weight_model=IPW(LogisticRegression(max_iter=4000), use_stabilized=True),
                                               survival_model=LogisticRegression(max_iter=4000)),
            'ipw_parametric_pipeline': WeightedSurvival(
                weight_model=IPW(LogisticRegression(max_iter=4000), use_stabilized=True),
                survival_model=Pipeline(
                    [('transform', PolynomialFeatures(degree=2)), ('LR', LogisticRegression(max_iter=1000, C=2))])),
            'standardization_non_stratified': StandardizedSurvival(survival_model=LogisticRegression(max_iter=4000),
                                                                   stratify=False),
            'standardization_stratified': StandardizedSurvival(survival_model=LogisticRegression(max_iter=4000),
                                                               stratify=True),
        }

    def test_nhefs(self, plot=False):
        EXPECTED_SURVIVAL_DIFFS = {'observed_non_parametric': 6,
                                   'observed_parametric': 6,
                                   'ipw_non_parametric': 1,
                                   'ipw_parametric': 1,
                                   'ipw_parametric_pipeline': 1,
                                   'standardization_non_stratified': 1,
                                   'standardization_stratified': 1,
                                   }
        TEST_DELTA = 2

        res = {}
        for estimator_name, estimator in self.estimators.items():
            estimator.fit(X=self.X, a=self.a, t=self.t, y=self.y)
            survival_curves = estimator.estimate_population_outcome(X=self.X, a=self.a, t=self.t, y=self.y)
            res[estimator_name] = survival_curves

        for estimator_name, survival_curve in res.items():
            surv_non_qsmk = 100.0 * float(survival_curve[0].iloc[-1])
            surv_qsmk = 100.0 * float(survival_curve[1].iloc[-1])
            self.assertAlmostEqual(surv_non_qsmk - surv_qsmk, EXPECTED_SURVIVAL_DIFFS[estimator_name], delta=TEST_DELTA)

        # Plots - for debugging purposes
        if plot:
            import matplotlib.pyplot as plt
            import itertools

            grid_dims = (int(np.ceil(np.sqrt(len(self.estimators)))), int(np.round(np.sqrt(len(self.estimators)))))
            grid_indices = itertools.product(range(grid_dims[0]), range(grid_dims[1]))
            fig, ax = plt.subplots(*grid_dims)

            estimator_names = list(self.estimators.keys())
            for estimator_name, plot_idx in zip(estimator_names, grid_indices):
                curve = res[estimator_name]
                ax[plot_idx].plot(curve[0])
                ax[plot_idx].plot(curve[1])
                ax[plot_idx].set_title(estimator_name)
                ax[plot_idx].set_ylim(0.5, 1.02)
                ax[plot_idx].grid()

            plt.show()


class TestMarginalOutcome(unittest.TestCase):
    """
    Test marginal (observed) survival curves.
    """
    def setUp(self) -> None:
        np.random.seed(RANDOM_SEED)

    def test_marginal_outcome_non_parametric_beneficial_effect(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['A']
        model = MarginalSurvival
        params = {}
        observed_diff = fit_synthetic_data(model_cls=model, params=params, test_data=test_data)

        self.assertAlmostEqual(observed_diff, TEST_DATA_DRUG_EFFECTS_A_OBSERVED_DIFF,
                               delta=TEST_DATA_DRUG_EFFECTS_DELTA)

    def test_marginal_outcome_parametric_null_effect(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['B']
        model = MarginalSurvival
        params = {'survival_model': LogisticRegression(max_iter=4000, C=10)}
        observed_diff = fit_synthetic_data(model_cls=model, params=params, test_data=test_data)

        self.assertAlmostEqual(observed_diff, TEST_DATA_DRUG_EFFECTS_B_OBSERVED_DIFF,
                               delta=TEST_DATA_DRUG_EFFECTS_DELTA)


class TestIPW(unittest.TestCase):
    """
    Test inverse propensity weighted survival curves.
    """
    def setUp(self) -> None:
        np.random.seed(RANDOM_SEED)

    def test_ipw_beneficial_effect(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['A']
        model = WeightedSurvival
        params = {'weight_model': IPW(LogisticRegression(max_iter=10000), use_stabilized=True)}
        adjusted_diff = fit_synthetic_data(model_cls=model, params=params, test_data=test_data)

        self.assertAlmostEqual(adjusted_diff, TEST_DATA_DRUG_EFFECTS_A_ORACLE_DIFF,
                               delta=TEST_DATA_DRUG_EFFECTS_DELTA)

    def test_ipw_null_effect(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['B']
        model = WeightedSurvival
        params = {'weight_model': IPW(LogisticRegression(max_iter=10000), use_stabilized=True)}
        adjusted_diff = fit_synthetic_data(model_cls=model, params=params, test_data=test_data)

        self.assertAlmostEqual(adjusted_diff, TEST_DATA_DRUG_EFFECTS_B_ORACLE_DIFF,
                               delta=TEST_DATA_DRUG_EFFECTS_DELTA)

    def test_unnamed_input(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['A']
        X = test_data[['x_0', 'x_1']]
        a = test_data['a']
        y = test_data['y']
        t = test_data['t']

        X.index.name = None
        a.name = None
        y.name = None
        t.name = None
        ipw = WeightedSurvival(weight_model=IPW(LogisticRegression()), survival_model=LogisticRegression())
        ipw.fit(X, a, t, y)
        outcomes = ipw.estimate_population_outcome(X=X, a=a, t=t, y=y)

        self.assertEqual(outcomes.index.name, "t")  # Default time name when canonizing names
        self.assertEqual(outcomes.columns.name, "a")  # Default time name when canonizing names


class TestStandardization(unittest.TestCase):
    """
    Test standardized survival curves.
    """
    def setUp(self) -> None:
        np.random.seed(RANDOM_SEED)

    def test_standardization_non_stratified_beneficial_effect(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['A']
        model = StandardizedSurvival
        params = {'survival_model': LogisticRegression(max_iter=2000, C=10), 'stratify': False}
        adjusted_diff = fit_synthetic_data(model_cls=model, params=params, test_data=test_data)

        self.assertAlmostEqual(adjusted_diff, TEST_DATA_DRUG_EFFECTS_A_ORACLE_DIFF,
                               delta=TEST_DATA_DRUG_EFFECTS_DELTA)

    def test_standardization_stratified_beneficial_effect(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['A']
        model = StandardizedSurvival
        params = {'survival_model': LogisticRegression(max_iter=2000, C=10), 'stratify': True}
        adjusted_diff = fit_synthetic_data(model_cls=model, params=params, test_data=test_data)

        self.assertAlmostEqual(adjusted_diff, TEST_DATA_DRUG_EFFECTS_A_ORACLE_DIFF,
                               delta=TEST_DATA_DRUG_EFFECTS_DELTA)

    def test_standardization_non_stratified_null_effect(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['B']
        model = StandardizedSurvival
        params = {'survival_model': LogisticRegression(max_iter=2000), 'stratify': False}
        adjusted_diff = fit_synthetic_data(model_cls=model, params=params, test_data=test_data)

        self.assertAlmostEqual(adjusted_diff, TEST_DATA_DRUG_EFFECTS_B_ORACLE_DIFF,
                               delta=TEST_DATA_DRUG_EFFECTS_DELTA)

    def test_standardization_stratified_null_effect(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['B']
        model = StandardizedSurvival
        params = {'survival_model': LogisticRegression(max_iter=2000), 'stratify': True}
        adjusted_diff = fit_synthetic_data(model_cls=model, params=params, test_data=test_data)

        self.assertAlmostEqual(adjusted_diff, TEST_DATA_DRUG_EFFECTS_B_ORACLE_DIFF,
                               delta=TEST_DATA_DRUG_EFFECTS_DELTA)

    def test_weighted_standardization_stratified(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['A']
        model = WeightedStandardizedSurvival
        params = {'weight_model': IPW(LogisticRegression(max_iter=1000), use_stabilized=True),
                  'survival_model': LogisticRegression(max_iter=1000),
                  'stratify': True}
        adjusted_diff = fit_synthetic_data(model_cls=model, params=params, test_data=test_data)

        self.assertAlmostEqual(adjusted_diff, TEST_DATA_DRUG_EFFECTS_A_ORACLE_DIFF,
                               delta=TEST_DATA_DRUG_EFFECTS_DELTA)

    def test_weighted_standardization_non_stratified(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['A']
        model = WeightedStandardizedSurvival
        params = {'weight_model': IPW(LogisticRegression(max_iter=2000), use_stabilized=True),
                  'survival_model': LogisticRegression(max_iter=4000, C=5),
                  'stratify': False}
        adjusted_diff = fit_synthetic_data(model_cls=model, params=params, test_data=test_data)

        self.assertAlmostEqual(adjusted_diff, TEST_DATA_DRUG_EFFECTS_A_ORACLE_DIFF,
                               delta=TEST_DATA_DRUG_EFFECTS_DELTA)

    def test_unnamed_input(self):
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['A']
        X = test_data[['x_0', 'x_1']]
        a = test_data['a']
        y = test_data['y']
        t = test_data['t']

        X.index.name = None
        a.name = None
        y.name = None
        t.name = None
        std = StandardizedSurvival(survival_model=LogisticRegression())
        std.fit(X, a, t, y)
        outcomes = std.estimate_population_outcome(X=X, a=a, t=t, y=y)

        self.assertEqual(outcomes.index.name, "t")  # Default time name when canonizing names
        self.assertEqual(outcomes.columns.name, "a")  # Default time name when canonizing names


@unittest.skipUnless(LIFELINES_FOUND, 'lifelines not found')
class TestLifelines(unittest.TestCase):
    """
    Test integration with 'lifelines' Python package.
    """
    def setUp(self) -> None:
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['A']
        self.X = test_data[['x_0', 'x_1']]
        self.a = test_data['a']
        self.y = test_data['y']
        self.t = test_data['t']

    def test_weighted_kaplan_meier_curves(self):
        weighted_survival = WeightedSurvival(
            weight_model=IPW(LogisticRegression(max_iter=10000, C=10), use_stabilized=True),
            survival_model=None
        )
        weighted_survival.fit(self.X, self.a)
        curves_causallib = weighted_survival.estimate_population_outcome(self.X, self.a, self.t, self.y)

        weighted_survival_lifelines_km = WeightedSurvival(
            weight_model=IPW(LogisticRegression(max_iter=10000, C=10), use_stabilized=True),
            survival_model=lifelines.KaplanMeierFitter()
        )
        weighted_survival_lifelines_km.fit(self.X, self.a)
        curves_causallib_lifelines = weighted_survival_lifelines_km.estimate_population_outcome(self.X, self.a, self.t, self.y)

        np.testing.assert_array_almost_equal(curves_causallib, curves_causallib_lifelines, decimal=8)

    def test_marginal_kaplan_meier_curves(self):
        marginal_survival = MarginalSurvival(survival_model=None)
        marginal_survival.fit(self.X, self.a)
        marginal_curves_causallib = marginal_survival.estimate_population_outcome(self.X, self.a, self.t, self.y)

        marginal_survival_lifelines = MarginalSurvival(survival_model=lifelines.KaplanMeierFitter())
        marginal_survival_lifelines.fit(self.X, self.a)
        marginal_curves_causallib_lifelines = marginal_survival_lifelines.estimate_population_outcome(self.X, self.a, self.t, self.y)

        lifelines_km_a0 = lifelines.KaplanMeierFitter()
        lifelines_km_a0.fit(durations=self.t[self.a == 0], event_observed=self.y[self.a == 0])
        lifelines_km_a1 = lifelines.KaplanMeierFitter()
        lifelines_km_a1.fit(durations=self.t[self.a == 1], event_observed=self.y[self.a == 1])
        marginal_curves_lifelines = pd.DataFrame({0: lifelines_km_a0.predict(sorted(self.t.unique())),
                                                  1: lifelines_km_a1.predict(sorted(self.t.unique()))})
        marginal_curves_lifelines.columns.name = 'a'
        marginal_curves_lifelines.index.name = 't'

        pd.testing.assert_frame_equal(marginal_curves_causallib, marginal_curves_causallib_lifelines)
        pd.testing.assert_frame_equal(marginal_curves_causallib, marginal_curves_lifelines)

    def test_cox(self):
        standardized_survival_cox = StandardizedSurvival(survival_model=lifelines.CoxPHFitter())
        standardized_survival_cox.fit(self.X, self.a, self.t, self.y)
        _ = standardized_survival_cox.estimate_population_outcome(self.X, self.a, self.t, self.y)
        # not validating results - only testing for pass/fail

    def test_fit_kwargs(self):
        ipw = IPW(learner=LogisticRegression(max_iter=1000))
        weighted_standardized_survival = WeightedStandardizedSurvival(survival_model=lifelines.CoxPHFitter(),
                                                                      weight_model=ipw)

        # Without fit_kwargs - should raise StatisticalWarning with a suggestion to pass robust=True in fit
        with self.assertWarns(lifelines.exceptions.StatisticalWarning):
            weighted_standardized_survival.fit(self.X, self.a, self.t, self.y)

        # With fit_kwargs - should not raise StatisticalWarning (might raise other warnings, though)
        with self.assertRaises(AssertionError):  # negation workaround since there's no assertNotWarns
            with self.assertWarns(lifelines.exceptions.StatisticalWarning):
                weighted_standardized_survival.fit(self.X, self.a, self.t, self.y, fit_kwargs={'robust': True})


class TestFeatureTransform(unittest.TestCase):
    """
    Test scikit-learn Pipeline/transform as a base learner for standardized survival curves.
    """
    def setUp(self) -> None:
        test_data = TEST_DATA_TTE_DRUG_EFFECTS['A']
        self.X = test_data[['x_0', 'x_1']]
        self.a = test_data['a']
        self.y = test_data['y']
        self.t = test_data['t']
        self.eps = 0.2

        feature_transform = PolynomialFeatures(degree=2)
        learner = LogisticRegression(max_iter=4000, C=0.8)
        self.pipeline = Pipeline([('transform', feature_transform), ('LR', learner)])

    def test_regression_with_transform(self):
        standardized_survival = StandardizedSurvival(survival_model=LogisticRegression(max_iter=1000), stratify=False)
        standardized_survival.fit(self.X, self.a, self.t, self.y)
        curves = standardized_survival.estimate_population_outcome(self.X, self.a, self.t, self.y)

        standardized_survival_poly = StandardizedSurvival(survival_model=self.pipeline, stratify=False)
        standardized_survival_poly.fit(self.X, self.a, self.t, self.y)
        curves_poly = standardized_survival_poly.estimate_population_outcome(self.X, self.a, self.t, self.y)

        diff = np.abs(curves - curves_poly)
        condition = (diff < self.eps).all(axis=None)
        self.assertTrue(condition)
