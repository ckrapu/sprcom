import numpy             as np
import pymc3             as pm

from scipy.linalg  import eigvalsh
from utils         import CAR

def community_regression_model(X,Y,C,W,setting = 'mvcar',response='bernoulli'):
    """
    Generates a PyMC3 model for fitting the spatial community regression model.

    Arguments
    ---------
    X : 2D Numpy array
        Covariates with shape [n_sites, n_covariates]
    Y : 2D Numpy array
        Response variable with shape [n_sites, n_species]
    C : int
        Number of communities
    W : 2D Numpy array
        Binary adjacency matrix
    community_effect_setting : string
        Type of model to implement. Choices are 'mvcar', 'car', and 'none'.
        'mvcar' implements the linear model of coregionalization to obtain
        correlated spatial processes underlying the community scores. 'car'
        treats these spatial processes as independent. 'none' removes them
        entirely.
    response : str
        Distribution appropriate for the response contained in <Y>.
        Currently supported options are 'bernoulli' for binary data and
        'poisson' for count-valued data

    Returns
    -------
    model : PyMC3 Model object

    """

    # Extract the number of sampling units (N), covariates (P) and species (S)
    N,P = X.shape
    _,S = Y.shape

    # For fastest GPU performance, these variables must be cast
    # to 32-bit float representation
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    W = W.astype(np.float32)

    # eigenvalues of D^−1/2 * W * D^−1/2
    # we pre-process this for the CAR initialization
    D = W.sum(axis=0)
    D_W = np.diag(D)
    Dinv_sqrt = np.diag(1 / np.sqrt(D))
    DWD = np.matmul(np.matmul(Dinv_sqrt, W), Dinv_sqrt)
    lam = eigvalsh(DWD)
    with pm.Model() as model:
        # Removes the community spatial effect
        if setting.lower() is 'none':
            community_effect = 0

        # Uncorrelated conditional autoregressions for each community
        elif setting.lower() is 'car':
            rho             = pm.Uniform('rho', shape = C)
            pe_var          = pm.InverseGamma('pe_var', alpha = 0.01, beta = 0.01, shape = C)
            univariate_CARs = [CAR('car_{0}'.format(c),alpha = rho[c],W=W,tau=pe_var[c,],shape = N,lam=lam) for c in range(C)]
            community_effect= pm.Deterministic('community_effect',pm.math.stack(univariate_CARs,axis = 1))

        # Correlated multivariate conditional autoregression for all communities
        # simultaneously
        elif setting.lower() is 'mvcar':
            rho             = pm.Uniform('rho', shape = C)
            univariate_CARs = [CAR('car_{0}'.format(c),alpha = rho[c],W=W,tau=1,shape = N,lam=lam) for c in range(C)]
            raw_plot_effect = pm.math.stack(univariate_CARs,axis = 1)
            packed_A        = pm.LKJCholeskyCov('packed_A', n=C,eta=1,
             sd_dist=pm.HalfCauchy.dist(2.5))
            A                = pm.Deterministic('A',pm.expand_packed_triangular(C,packed_A))
            community_effect = pm.Deterministic('community_effect',pm.math.dot(raw_plot_effect,A))

        # Scalar intercept applied to all species
        intercept = pm.Normal('intercept', sd = 10)

        # Regression coefficients linking covariaties
        # to community scores
        beta_var  = pm.InverseGamma('beta_var', alpha=0.01, beta=0.01)
        beta_raw  = pm.Normal('beta_raw',shape = [C,P])
        beta      = pm.Deterministic('beta', beta_raw*(beta_var**0.5))
        theta     = pm.Deterministic('theta',pm.math.dot(X, beta.T) + community_effect)

        phi = pm.HalfNormal('phi', shape=[C,S])
        mu  = pm.math.dot(theta, phi) + intercept

        if response.lower() == 'bernoulli':
            p        = pm.math.sigmoid(mu)
            response = pm.Bernoulli('response', p=p, observed=Y)

        elif response.lower() == 'poisson':
            rate     = pm.math.exp(mu)
            response = pm.Poisson('response',mu=rate)
    return model
