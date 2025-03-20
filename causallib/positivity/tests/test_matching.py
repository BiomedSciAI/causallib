import unittest
from causallib.positivity import Matching
from causallib.positivity.datasets.test_data_simulator import make_1d_overlap_data, make_random_y_like


class TestMatching(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matching = Matching()

    def test_1d_euclidean_caliper_1_match(self):
        X, a= make_1d_overlap_data(treatment_bounds=(0, 75),
                                      control_bounds=(25, 100))
        self.matching.matching_transformer.set_params(metric="euclidean", caliper=1.0)
        predict = self.matching.fit_predict(X, a)
        Xp, ap= self.matching.transform(X, a,)
        self.assertEqual(sum(predict), 100)
        self.assertEqual(sum(predict), Xp.shape[0])
        self.assertEqual(Xp.values.min(), 25)
        self.assertEqual(Xp.values.max(), 74)

    def test_transform_multiple_outputs(self):
        X, a= make_1d_overlap_data(treatment_bounds=(0, 75),
                                      control_bounds=(25, 100))
        y = make_random_y_like(a)
        t = make_random_y_like(a)

        self.matching.matching_transformer.set_params(metric="euclidean", caliper=1.01)
        self.matching.fit(X, a)

        Xm, am = self.matching.transform(X, a)
        unique_sample_counts = {i.shape[0] for i in (Xm, am)}
        self.assertEqual(len(unique_sample_counts), 1)

        Xm, am, ym= self.matching.transform(X, a, y)
        unique_sample_counts = {i.shape[0] for i in (Xm, am, ym)}
        self.assertEqual(len(unique_sample_counts), 1)

        Xm, am, ym, tm = self.matching.transform(X, a, y, t)
        unique_sample_counts = {i.shape[0] for i in (Xm, am, ym, tm)}
        self.assertEqual(len(unique_sample_counts), 1)