# SpRCom

![animation](https://raw.githubusercontent.com/ckrapu/sprcom/master/data/animation.gif)

`sprcom` stands for **Sp**atial **R**egression of **Com**munities and is a Bayesian statistical package designed to streamline the interpretation and modeling of very high dimensional binary and count-valued data. The underlying model assumes a low-dimensional latent structure via communities or clusters that leads to a parsimonious model. `sprcom` is unique in that it can also account for the dependence of these communities on covariates! A number of utility and plotting functions are included to help visualize your results. `sprcom` is a wrapper for a [PyMC3](https://docs.pymc.io/) model and you can use any PyMC3 estimation method with it including Hamiltonian Monte Carlo and ADVI. In particular, it is especially well suited for GPU computing.

```python
covariates, response, adjacency = load_data(...)
n_communities = 5
model = spatial_community_regression(covariates, response, adjacency, n_communities)
with model:
  trace = pm.sample()
...
```
## Installation
You can install this package using pip: `pip install sprcom`. It requires several dependencies including PyMC3 and Seaborn.

We've included documentation to help you get up and running. Check out the `florabank1-tutorial` notebook for more details!

For questions or comments please contact Christopher Krapu at `ckrapu@gmail.com`.
