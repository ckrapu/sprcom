# SpRCom


<p align="center">
<img src="https://github.com/ckrapu/sprcom/blob/master/data/animation.gif">
</p>

`sprcom` stands for **Sp**atial **R**egression of **Com**munities and is a statistical  package designed to streamline the interpretation and modeling of very high dimensional binary and count-valued data. The underlying model assumes a low-dimensional latent structure via communities or clusters that leads to a parsimonious model. `sprcom` can also account for the dependence of these communities on covariates! A number of utility and plotting functions are included to help visualize your results. `sprcom` is a wrapper for a [PyMC3](https://docs.pymc.io/) model and you can use any PyMC3 estimation method with it including Hamiltonian Monte Carlo and ADVI.

```python
covariates, response, adjacency = load_data(...)
n_communities = 5
model = spatial_community_regression(covariates, response, adjacency,n_communities)
with model:
  trace = pm.sample()
...
```



We've included documentation to help you get up and running. Check out the `florabank1-tutorial` notebook for more details!

For questions or comments please contact Christopher Krapu at `ckrapu@gmail.com`.
