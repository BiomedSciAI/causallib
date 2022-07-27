from causallib.estimation import IPW,Matching
from causallib.evaluation.weight_predictor import calculate_covariate_balance
from causallib.preprocessing.transformers import PropensityTransformer
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

learner = lambda : LogisticRegression(solver="liblinear",max_iter=5000,class_weight="balanced")
X = pd.read_csv("~/lalonde_data.csv").set_index("sample_id")
a = X.pop("training")       
y = X.pop("re78")

caliper_vec = np.logspace(-6,-1,20)
covbal=[]

def match_then_ipw_weight(caliper):
    propensity_transform = PropensityTransformer(include_covariates=True,learner=learner())
    matcher = Matching(propensity_transform=propensity_transform,caliper=caliper)
    matcher.fit(X,a,y)
    Xm,am,ym=matcher.transform(X,a,y)
    ipw = IPW(learner=learner())
    ipw.fit(Xm,am,)
    ipw_weights = ipw.compute_weights(Xm,am)
    ipw_outcome = ipw.estimate_population_outcome(Xm,am,ym)
    matched_treated = sum(am==1)
    matched_control = sum(am==0)
    covbalance = calculate_covariate_balance(Xm,am,ipw_weights)
    return {"caliper":caliper,"n_treated":matched_treated,"n_control":matched_control,
            "ipw_weights":ipw_weights,"ipw_outcome":ipw_outcome,
            "covariate_balance":covbalance.drop(columns="unweighted")}

results = [match_then_ipw_weight(c) for c in caliper_vec]
covbal_df = pd.concat([i["covariate_balance"] for i in results],axis=1)

