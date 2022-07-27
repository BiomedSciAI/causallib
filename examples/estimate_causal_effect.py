from causallib.estimation.matching import OneDimensionalNearestNeighbors
from causallib.estimation import (
    DoublyRobustIpFeature,
    DoublyRobustJoffe,
    DoublyRobustVanilla,
    IPW,
    Standardization,
    StratifiedStandardization,
    MarginalOutcomeEstimator,
    Matching,
    PropensityMatching,
    MatchingTransformer,
    MatchingIndividualOutcomeEstimator,
)
from causallib.datasets import load_nhefs, load_card_krueger, load_lalonde, load_acic16
from causallib.preprocessing.transformers import PropensityTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
import pandas as pd


def makeipw(): return IPW(learner=LogisticRegression(solver="liblinear"))
def makestd(): return StratifiedStandardization(learner=LinearRegression())


all_estimators = [
    MarginalOutcomeEstimator(learner=LinearRegression()),
    Matching(),
    PropensityMatching(learner=LogisticRegression(
        solver="liblinear"), caliper=0.02),
    MatchingIndividualOutcomeEstimator(),
    IPW(learner=LogisticRegression(solver="liblinear")),
    Standardization(learner=LinearRegression()),
    StratifiedStandardization(learner=LinearRegression()),
    DoublyRobustIpFeature(makestd(), makeipw()),
    DoublyRobustJoffe(makestd(), makeipw()),
    DoublyRobustVanilla(makestd(), makeipw()),
]


def get_all_estimates(x, a, y):
    estimates = {}
    m0 = Matching(
        propensity_transform=PropensityTransformer(
            include_covariates=True,
            learner=LogisticRegression(
                solver="liblinear", class_weight="balanced", max_iter=5000)
        ),
        caliper=0.0001,)
    m0.knn_backend = lambda: OneDimensionalNearestNeighbors()
    m1 = Matching(
        propensity_transform=PropensityTransformer(
            include_covariates=True,
            learner=LogisticRegression(
                solver="liblinear", class_weight="balanced", max_iter=5000)
        ),
        caliper=0.0001)
    for idx, estimator in enumerate([
        m0,
        #Matching(caliper=0.0001),

    ]):
        try:
            estimator.fit(x, a, y)
        except:
            estimator.fit(x, a)
        estimator_name = estimator.__class__.__name__
        estimates[f"{estimator_name}_{idx}"] = estimator.estimate_population_outcome(
            x, a, y)
    estimates_df = pd.concat(estimates, axis=1).T
    estimates_df = estimates_df.assign(ATE=estimates_df[1] - estimates_df[0])
    return estimates_df


def load_lalonde_local(limit_to=None):
    from sklearn.utils import Bunch
    lalonde = pd.read_csv("~/lalonde_data.csv")
    if limit_to:
        lalonde = lalonde.sample(limit_to)
        lalonde = lalonde.loc[:, (lalonde != 0).any(axis=0)]
    lalonde = lalonde[[i for i in lalonde.columns if "education" not in i]]
    lalonde = lalonde.set_index("sample_id")
    a = lalonde.pop("training")
    y = lalonde.pop("re78")
    X = lalonde
    return Bunch(X=X, a=a, y=y)


#data_nhefs = load_nhefs()
#data_cardkrueger = load_card_krueger()
#data_acic16 = load_acic16()
data_lalonde = load_lalonde_local(limit_to=None)
#all_data = [data_nhefs, data_cardkrueger, data_acic16, data_lalonde]
for dataset in [data_lalonde]:
    print(dataset.y.name)
    estimates_df = get_all_estimates(dataset.X, dataset.a, dataset.y)
    print(estimates_df)
