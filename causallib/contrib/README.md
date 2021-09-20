# Module `causallib.contrib`
This module currently includes additional causal methods contributed to the package 
by causal inference researchers other than `causallib`'s core developers.

The causal models in this module can be slightly more novel then in the ones in `estimation` module. 
However, they should largely adhere to `causallib` API 
(e.g., `IndividualOutcomeEstimator` or `WeightEstimator`).
Since code here is more experimental, 
models might also require additional (and less trivial) package dependencies, 
or have less test coverage.  
Well-integrated models could be transferred into the main `estimation` module in the future.

## Contributed Methods
Currently contributed methods are:

1. Adversarial Balancing: implementing the algorithm described in 
   [Adversarial Balancing for Causal Inference](https://arxiv.org/abs/1810.07406).
   ```python
   from causallib.contrib.adversarial_balancing import AdversarialBalancing
   ```
1. Interpretable Subgroup Discovery in Treatment Effect Estimation: 
   implementing the heterogeneous effect mixture model (HEMM) presented in 
   [Interpretable Subgroup Discovery in Treatment Effect Estimation with Application to Opioid Prescribing Guidelines](https://arxiv.org/pdf/1905.03297.pdf) 
   ```python
   from causallib.contrib.hemm import HEMM
   ```
1. Matching Estimation/Transform using `faiss`.

   Implemented a nearest neighbors search with API that matches `sklearn.NearestNeighbors`
   but is powered by [faiss](https://github.com/facebookresearch/faiss) for GPU
   support and much faster search on CPU as well.
   
   ```python
   from causallib.contrib.faissknn import FaissNearestNeighbors
   ```

## Dependencies
Each model might have slightly different requirements.  
Refer to the documentation of each model for the additional packages it requires.  
  
Requirements for `contrib` models are concentrated in `contrib/requirements.txt` 
and can be automatically installed using the extra-requirements `contrib` flag:  
```shell script
pip install causallib[contrib] -f https://download.pytorch.org/whl/torch_stable.html
```  
The `-f` find-links option is required to install PyTorch dependency.

## References

Ozery-Flato, M., Thodoroff, P., Ninio, M., Rosen-Zvi, M., & El-Hay, T. (2018). [Adversarial balancing for causal inference.](https://arxiv.org/abs/1810.07406) arXiv preprint arXiv:1810.07406.

Nagpal, C., Wei, D., Vinzamuri, B., Shekhar, M., Berger, S. E., Das, S., & Varshney, K. R. (2020, April). [Interpretable subgroup discovery in treatment effect estimation with application to opioid prescribing guidelines.](https://arxiv.org/pdf/1905.03297.pdf) In Proceedings of the ACM Conference on Health, Inference, and Learning (pp. 19-29).
