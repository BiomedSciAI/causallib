<!---
Authors: Ehud Karavani, Yishai Shimoni
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

--->

# Module `causallib.datasets`

This module contains an example dataset, 
and a simulator to create a dataset.

## Datasets
Currently one dataset is included.
This is the National Health and Nutrition Examination Survey (NNHEFS) dataset.
The dataset was adapted from the data available at
<https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/>.

It can be loaded using 
```Python
from causallib.datasets.data_loader import load_nhefs
data = load_nhefs()
covariates = data.X, 
treatment_assignment = data.a, 
observed_outcome = data.y 
``` 

This loads an object in which `data.X`, `data.a`, and `data.y`
respectively hold the features for each individual,
whether they stopped-smoking, 
and their observed difference in weight between 1971 and 1983.

## Simulator 
This module implements a simulator and some related functions 
(e.g. creating random graph topologies)

CausalSimulator is based on an explicit graphical model connecting 
the feature data with several special nodes for treatment assignmenr, outcome,
and censoring.  
CausalSimulator can generate the feature data randomly, 
or it can use a given dataset. 
The approach without input data is exhibited below, 
and the approach based on existing data is exemplified in the 
notebook [`CasualSimulator_example.ipynb`](CasualSimulator_example.ipynb) 

### With no given data

To initialize the simulator you need to state all the arguments
regarding the graph's structure and variable related information

```Python
import numpy as np
from causallib.datasets import CausalSimulator
topology = np.zeros((4, 4), dtype=np.bool)  # topology[i,j] iff node j is a parent of node i
topology[1, 0] = topology[2, 0] = topology[2, 1] = topology[3, 1] = topology[3, 2] = True
var_types = ["hidden", "covariate", "treatment", "outcome"]
link_types = ['linear', 'linear', 'linear', 'linear']
prob_categories = [[0.25, 0.25, 0.5], None, [0.5, 0.5], None]
treatment_methods = "gaussian"
snr = 0.9
treatment_importance = 0.8
effect_sizes = None
outcome_types = "binary"

sim = CausalSimulator(topology=topology, prob_categories=prob_categories,
                      link_types=link_types, snr=snr, var_types=var_types,
                      treatment_importances=treatment_importance, 
                      outcome_types=outcome_types, 
                      treatment_methods=treatment_methods, 
                      effect_sizes=effect_sizes)
X, prop, (y0, y1) = sim.generate_data(num_samples=100)
```

```plantuml
digraph CausalGraph {
hidden -> covariate
hidden -> treatment
covariate -> treatment
covariate -> outcome
treatment -> outcome
}
```

* This creates a graph `topology` of 4 variables, as depicted in the graph above: 
  1 hidden var (i.e. latent
  covariate), 1 regular covariate, 1 treatment variable and 1 outcome.
* `link_types` determines that all variables will have linear 
  dependencies on their predecessors.
* `var_types`, together with `prov_categories` define: 
  * Variable 0 (hidden) is categorical with categories
    distributed by the multinomial distribution `[0.25, 0.25, 0.5]`.
  * Variable 1 (covariate) is continuous (since its
    corresponding prob_category is None).
  * Variable 2 (treatment) is categorical and treatment assignment is equal
    between the treatment groups. 
* `treatment_methods` means that treatment will be assigned by percentiles using a 
  Gaussian distribution.
* All variables have signal to noise ratio of 
  *signal* / (*signal*+*noise*) = 0.9.
* `treatment_importance = 0.8` indicates that the outcome will be affected 80% 
  by treatment and 20% by all other predecessors.
* Effect size won't be manipulated into a specific desired value (since
  it is None).
* Outcome will be binary.

The data that is generated contains:
* `X` contains all the data generated (including latent variables,
  treatment assignments and outcome)
* `prop` contains the propensities
* `y0` and `y1` hold the counterfactual outcomes without and with treatment, 
respectively.

### Additional examples
A more elaborate example that includes using existing data 
is available in the example notebook.

## License
Datasets are provided under [Community Data License Agreement (CDLA)](https://cdla.io/).  
The ACIC16 dataset is provided under [CDLA-sharing](https://cdla.io/sharing-1-0/) license.  
The NHEFS dataset is provided under [CDLA-permissive](https://cdla.io/permissive-1-0/) license.  
Please see the full corresponding license within each directory.  

We thank the authors for sharing their data within this package.
