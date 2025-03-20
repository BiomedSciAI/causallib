import unittest
from causallib.positivity.datasets.positivity_data_simulator import make_1d_overlap_data
from causallib.positivity.datasets.pizza_data_simulator import pizza


class TestPositivityDataSim(unittest.TestCase):

    def test_get_1d_data(self):
        treatment_bounds = (0, 75)
        control_bounds = (25, 100)
        X, a = make_1d_overlap_data(
            treatment_bounds=treatment_bounds, control_bounds=control_bounds)
        self.assertEqual(X[a == 0].values.min(), control_bounds[0])
        self.assertEqual(X[a == 0].values.max(), control_bounds[1]-1)
        self.assertEqual(X[a == 1].values.min(), treatment_bounds[0])
        self.assertEqual(X[a == 1].values.max(), treatment_bounds[1]-1)


class TestPositivityPizzaData(unittest.TestCase):
    def test_pizza_data(self):
        X, a = pizza(seed=0, n_samples=10000)
        self.assertAlmostEqual(
            X.loc[a == 0, 0].values.min(), X.loc[a == 1, 0].values.min(),
            delta=0.1,  # depends on the density of the points
        )
        self.assertAlmostEqual(
            X.loc[a == 0, 1].values.min(), X.loc[a == 1, 1].values.min(),
            delta=0.1,  # depends on the density of the points
        )
        self.assertAlmostEqual(
            X.loc[a == 0, 0].values.max(), X.loc[a == 1, 0].values.max(),
            delta=0.1,  # depends on the density of the points
        )
        self.assertAlmostEqual(
            X.loc[a == 0, 1].values.max(), X.loc[a == 1, 1].values.max(),
            delta=0.1,  # depends on the density of the points
        )

    def test_dimensions(self):
        X, a = pizza(n_dim=3, n_samples=10)
        self.assertEqual(3, X.shape[1])
        self.assertEqual(10, X.shape[0])
        self.assertEqual(10, a.shape[0])
