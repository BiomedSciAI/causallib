import unittest
from causallib.positivity import base_positivity
from causallib.positivity.tests.testdata import make_1d_overlap_data


class TestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_get_1d_data(self):
        treatment_bounds = (0, 75)
        control_bounds = (25, 100)
        X, a = make_1d_overlap_data(
            treatment_bounds=treatment_bounds, control_bounds=control_bounds)
        self.assertEqual(X[a == 0].values.min(), control_bounds[0])
        self.assertEqual(X[a == 0].values.max(), control_bounds[1]-1)
        self.assertEqual(X[a == 1].values.min(), treatment_bounds[0])
        self.assertEqual(X[a == 1].values.max(), treatment_bounds[1]-1)
