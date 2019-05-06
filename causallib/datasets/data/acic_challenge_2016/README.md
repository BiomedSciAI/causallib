# ACIC 2016 challenge data

This folder contains covariates, simulated treatment, and simulated response variables 
for the causal inference challenge in the 2016 Atlantic Causal Inference Conference. 

For each of 20 conditions, treatment and response data were simulated from real-world data 
corresponding to 4802 individuals and 58 covariates.


#### Files:
  x.csv - matrix of covariates; categorical variables are coded as A/B/C/..., 
          binary variables as 0/1, and real numbers are left alone
  zy_##.csv - the twenty sets of treatment and response variables corresponding to various simulation settings; 
              treatment is column "z" and response is column "y"

#### Cite:
If used for academic purposes, please consider citing the competition organizers:
```bibtex
@article{dorie2017automated,
  title={Automated versus do-it-yourself methods for causal inference: Lessons learned from a data analysis competition},
  author={Dorie, Vincent and Hill, Jennifer and Shalit, Uri and Scott, Marc and Cervone, Dan},
  journal={arXiv preprint arXiv:1707.02641},
  year={2017}
}
```
