import numpy             as np
import pymc3             as pm
import matplotlib.pyplot as plt
import theano.tensor     as tt

from scipy.linalg  import eigvalsh
from scipy.sparse  import csr_matrix
from theano.sparse import as_sparse_variable
from theano.sparse import dot as t_sparse_dot
from theano        import shared
from seaborn       import heatmap

def expand_packed(n,packed):
    """
    Unpacks the entries of a lower triangular matrix from one dimension
    to two dimensions.Identical to PyMC3 expand_packed except it operates on
    numpy arrays rather than theano tensors.

    Parameters
    ----------
    n : int
        The dimension of the matrix to be unpacked
    packed : 1D array-like
        The array of matrix elements to be unpacked

    Returns
    -------
    out : 2D Numpy array
        Square, lower triangular matrix

    """
    out = np.zeros([n,n])
    ctr = 0
    for i in range(n):
        for j in range(i+1):
            out[i,j] = packed[ctr]
            ctr+=1
    return out

def get_neighbors(state):
    """
    For a 2D lattice, this creates a numpy array in which
    each entry is a list of tuples indicating the indices of its
    neighbors.

    Parameters
    ----------
    state : 2D Numpy array
        Array with the shape that should be copied
        for the resulting 2D lattice

    Returns
    -------
    neighbors : 2D Numpy array with list entries
        2D array indicating for each row, column position
        which matrix elements are its neighbors
    """

    y,x = state.shape

    # This array's entries are a list of neighbor
    # points for each site
    neighbors = np.empty([x,y],dtype=object)

    # Precalculate the list of neighbors for each cell so
    # this doesn't need to be recalculated on the fly.
    for i in range(y):
        for j in range(x):
            neighbors[i,j] = []
            if i is not 0:
                # add the neighbor above
                neighbors[i,j].append((i-1,j))
            if j is not 0:
                # add the neighbor to the left
                neighbors[i,j].append((i,j-1))
            if i is not y-1:
                # add the neighbor below
                neighbors[i,j].append((i+1,j))
            if j is not x-1:
                # add the neighbor to the right
                neighbors[i,j].append((i,j+1))
    return neighbors

def adjacency_from_1d(neighbors_1d):
    """
    Takes a 1D list of lists where the i-th inner list contains
    the indices of the i-th site's neighbors and converts it into
    an adjacency matrix.

    Parameters
    ----------
    neighbors_1d : list of arrays
        Sequence of lists with entries indicating the row and
        column index of neighbors in an adjacency matrix

    Returns
    -------
    W : 2D Numpy array
        binary adjacency matrix created according to the specification
        in neighbors_1d
    """
    n = len(neighbors_1d)
    W = np.zeros([n,n])
    for i in range(n):
        pairs = neighbors_1d[i]
        for j in pairs:
            W[i,j] = 1
    return W

def coverage(samples,true,width=90):
    """
    Calculates the fraction of variables with estimates
    falling within sample credible intervals.

    Parameters
    ----------
    samples : 2D Numpy array
        Array of samples across multiple variables
         with shape [n_samples, n_variables]
    true : 1D Numpy array
        Array of true values with shape [n_variables]
    width : integer
        Percentile from 0-100 specifying the width of
        the empirical credible interval to use.

    Returns
    -------
    fraction_covered : float
        The fraction of true variable values which
        fell within the sampled credible interval

    """

    qs   = [(100-width)/2,width+(100-width)/2]
    perc = np.percentile(samples,qs,axis=0)
    fraction_covered = np.mean((true>perc[0,:]) & (true<perc[1,:]))
    return fraction_covered

def sample_CAR(W,rho,tau):
    """
    Samples a realization of a conditional autoregression
    / Markov random field with correlation rho and precision tau.

    Parameters
    ----------
    W : 2D Numpy array
        Binary adjacency matrix with zeros on the diagonal
    rho : float
        Number between -1 and 1 specifying the spatial correlation
    tau : float
        Scale of the CAR realization in terms of the precision

    Returns
    -------
    sample : 2D Numpy array
        A random draw from the conditional autoregression /
        Markov random field defined by the input parameters
    """

    n   = W.shape[0]
    D   = W.sum(axis=0)
    D_W = np.diag(D)
    prec = tau * (D_W - rho * W)
    cov  = np.linalg.inv(prec)
    L    = np.linalg.cholesky(cov)
    w    = np.random.randn(n)
    sample = np.dot(L,w)
    return sample

def create_W_grid(width):
    """
    Creates an adjacency matrix for a square grid defined
    in terms of its width

    Arguments
    ---------
    width : int
        row and column dimension of the desired square
        adjacency matrix

    Returns
    -------
    W : 2D Numpy array
        Square binary adjacency matrix

    """
    # Create 2d lattice that gets unraveled into 1D
    grid         = np.arange(width**2).reshape([width,width])

    # Identify indices of neighbors for each entry
    neighbors    = get_neighbors(grid).ravel()
    neighbors_1d = [[grid[x] for x in l] for l in neighbors]
    grid         = grid.ravel()

    # Create adjacency matrix
    W            = adjacency_from_1d(neighbors_1d)
    return W

def diverse_community_matrix(S,C,multiplier=1,width=1):
    """
    Quick way to make a species/community matrix
    with diverse structure across communities. This matrix has rows that
    individually look like normal PDFs which are offset to make them dissimilar.

    Arguments
    ---------
    S : int
        Number of species
    C : int
        Number of communities, i.e. clusters of species
    multiplier : float
        Scale of the matrix entries
    width : float
        Spread of species across the community vector. Increase this
        to force each community to have more significant species.

    Returns
    -------
    sc : 2D Numpy array
        Matrix with dimensions [n_species,n_communities] where each
        community is represented by a positive vector with shape [n_species].
        These per-community vectors can have their inner product adjusted by
        the width parameter.

    """


    sc = np.zeros([S,C])
    x = np.linspace(-2,2,S)
    y = np.exp(-(1./width**2)*x**2)
    shift = int(S/C)
    for c in range(C):
        sc[:,c] = np.roll(y,shift=-shift*c,axis=0)
    sc = sc.T * multiplier
    return

def empirical_coverage(point,trace,width=90):
    """
    Calculates empirical coverage rates for all variables
    in a trace.

    Arguments
    ---------
    point : dict
        Dictionary mapping variable names to their true values.
        These names must be the same as in <trace>.
    trace : PyMC3 MultiTrace
        Posterior samples with variable names identical to <point>
    width : int
        Number from 0-100 indicating the nominal width of the credible
        intervals used to calculate the coverage.

    Returns
    -------
    results : dict
        Dictionary of empirical coverage rates
    """

    rvs = point.keys()
    results = {}

    total_coverage = 0
    total_size     = 0

    for rv in rvs:
        try:
            true     = point[rv]
            samples  = trace[rv]
            size     = np.product(true.shape)
            rv_coverage = coverage(samples,true,width=width)
            results[rv] = {'coverage':rv_coverage,'size':size}

            total_coverage += rv_coverage * size
            total_size     += size
        except:
            pass
    results['total']={'coverage':total_coverage/total_size,'size':total_size}
    return results

def tjur_r2(p, obs):
    """
    Gives the Tjur R-squared for a binary observation array
    <obs> and a array of values on [0,1] in <p> with the same
    shape as <obs>

    Arguments
    ---------
    p : Numpy array
        Predicted probabilities generated by a model
    obs : Numpy array
        Binary observations

    Returns
    -------
    r2 : float
        Tjur R-squared
    """

    obs = obs.astype(bool)
    p_vec = p.ravel()
    obs_vec = obs.ravel()
    r2 = np.mean(p_vec[obs_vec])- np.mean(p_vec[~obs_vec])
    return r2


def get_reordering(samples, true, C):
    """
    Calculates a reindexing that will align <samples>
    and <true> along their first non-sample dimension.
    This reindexing is determined by maximizing the correlation
    between vectors of the posterior mean of <samples> with vectors in <true>

    Arguments
    ---------
    samples : 3D Numpy array
        Posterior estimates of variables
        with dimension [n_samples, n_variables1, n_variables2]
    true : 2D Numpy array
        True values of variables with dimension [n_variables1, n_variables2]

    Returns
    -------
    permutation : 1D Numpy array
        Sequence of indices which, if used to index into samples.mean(),
        would rearrange them to share the same indices as <true>.
    """

    posterior_mean      = samples.mean(axis=0)
    corr                = np.corrcoef(true,posterior_mean)[0:C,C:2*C]
    inverse_permutation = np.linalg.inv((corr == corr.max(axis=0)).astype(int))
    permutation         = np.argmax(inverse_permutation,axis = 0)
    return permutation


def print_species(mean_phi, species_index,name_col='scientificname',
                  num_to_show=10):
    """
    Prints off the main species for a community matrix.

    Parameters
    ----------
    mean_phi : 2D Numpy array
        Posterior mean estimate of a species-community matrix with shape
        [n_species, n_communities]
    species_index : Pandas dataframe
        Pandas dataframe listing the integer index of a species along with
        its name
    name_col : string
        Name of the column in <species_index> which lists the name
    num_to_show : int
        Number of species to print for each community

    """
    C = mean_phi.shape[0]
    for c in range(C):
        top_indices = np.argsort(mean_phi[c,:])[-num_to_show::]
        top_species = species_index.iloc[top_indices][name_col].values
        print('\nSpecies for community {0}'.format(c))
        for species in top_species:
            print('\t',species)

def coefficient_plot(samples,x_labels,figsize =(5,3)):
    """
    Generates a plot showing the relation between communities and regression
    coefficients. Asterisks / stars indicate variables that are significant
    at the 2 sigma level.

    Parameters
    ----------
    samples : 3D Numpy array
        coefficient samples with shape [n_samples, n_covariates, n_communities]
    x_labels : List of strings
        Names of the covariates
    figsize : tuple
        Size of the desired plot

    Returns
    -------
    fig : Matplotlib Figure
        Figure object holding the plot
    ax : Matplotlib axes
        Axes object showing the plot

    """

    fig = plt.figure(figsize = figsize)

    stdevs = samples.std(axis = 0)
    means  = samples.mean(axis = 0)
    is_sig = np.abs(means) > (2*stdevs)
    C = samples.shape[1]
    ax = heatmap(samples.mean(axis=0),square=True,xticklabels=x_labels,cmap='RdBu_r',
               linewidth=0.4,linecolor='k')
    xs,ys = np.where(~is_sig.T)
    plt.scatter(xs+0.5,ys+0.5,marker='*',color='k',s=70)
    plt.ylim(-0.5,C+0.5); plt.ylabel('Community')

    return fig,ax

class CAR(pm.distributions.distribution.Continuous):
    """
    Conditional Autoregressive (CAR) distribution for PyMC3. Copied from
    documentation authored by Junpeng Lao at
    https://docs.pymc.io/notebooks/PyMC3_tips_and_heuristic.html.

    Parameters
    ----------
    alpha : float
        Spatial autocorrelation
    W : 2D Numpy array
        Binary adjacency matrix
    tau : Scale of CAR variables specified in terms of
        precision at each location
    """

    def __init__(self, alpha=None, W=None, tau=None,lam=None,sparse=False,
                *args, **kwargs):
        self.alpha  = tt.as_tensor_variable(alpha)
        self.tau    = tt.as_tensor_variable(tau)
        self.sparse = sparse
        self.D = D = W.sum(axis=0)
        self.D_W = np.diag(D)
        self.n = self.D.shape[0]
        self.median = self.mode = self.mean = np.zeros(self.n)
        super(CAR, self).__init__(*args, **kwargs)

        # eigenvalues of D^−1/2 * W * D^−1/2
        if lam is None:
            Dinv_sqrt = np.diag(1 / np.sqrt(D))
            DWD = np.matmul(np.matmul(Dinv_sqrt, W), Dinv_sqrt)
            self.lam = eigvalsh(DWD)
        else:
            self.lam = lam

        # sparse representation of W
        if sparse:
            w_sparse = csr_matrix(W)
            self.W   = as_sparse_variable(w_sparse)
        else:
            self.W = shared(W)
        self.D_tt = tt.as_tensor_variable(D)
        self.D    = D

    def logp(self, x):
        """Calculates log-likelihood for CAR random variable"""
        logtau = self.n * tt.log(self.tau)
        logdet = tt.log(1 - self.alpha * self.lam).sum()
        if self.sparse:
            Wx = t_sparse_dot(self.W, x)
        else:
            Wx = tt.dot(self.W,x)
        tau_dot_x = self.D_tt * x.T - self.alpha * Wx.ravel()
        logquad = self.tau * tt.dot(x.ravel(), tau_dot_x.ravel())

        return 0.5*(logtau + logdet - logquad)

    def random(self, point=None, size=None):
        """Random draw from a CAR-distributed vector"""
        alpha,tau = draw_values([self.alpha,self.tau],point=point,size=size)
        prec = tau * (self.D_W-alpha * self.W)
        cov  = np.linalg.inv(prec)
        L    = np.linalg.cholesky(cov)
        w    = np.random.randn(self.n)
        return np.dot(L,w)

def simulate_community_regression(N=625,S=500,C=5,P=5,seed=827,
                                  community_effect_setting = 'mvcar'):
    """
    Generates a synthetic dataset for the spatial community regression model.
    The covariates and regression coefficients are assumed to be IID normally
    distributed with a variance of 1.

    Parameters
    ----------
    N : int
        Number of desired observation sites
    S : int
        Number of desired species
    C : int
        Number of communities
    P : int
        Number of covariates
    seed: int
        Random seed used for reproducibility
    plot_effect_setting : string
        Choice of model type to simulate under; choices are 'mvcar', 'car'
        or 'none'. 'mvcar' implements the linear model of coregionalization as
        described in the Krapu 2019 paper while 'car' implements uncorrelated
        conditional autoregressions. 'none' omits any plot-level community
        effect.

    Returns
    -------
    data : dict
        Dictionary mapping variable names to their simulated values.


    """
    width = int(N ** 0.5)
    data = {}

    np.random.seed(seed)

    data['X']         = np.random.randn(N,P)
    data['beta']      = np.random.randn(C,P)
    data['intercept'] = np.random.randn()

    if community_effect_setting.lower() is 'none':
        data['plot_effect'] = 0
        data['W'] = None

    elif community_effect_setting in ['car','mvcar']:
        data['W']     = create_W_grid(width)
        D             = data['W'].sum(axis=0)
        D_W           = np.diag(D)
        data['rho']   = np.random.uniform(low = 0.9,high = 1.0,size=C)
        data['tau']   = np.ones(C)
        data['plot_effect'] = np.zeros([N,C])

        for c in range(C):
            prec = data['tau'][c] * (D_W - data['rho'][c] *data['W'])
            cov  = np.linalg.inv(prec)
            L    = np.linalg.cholesky(cov)
            data['plot_effect'] [:,c] = np.dot(L,np.random.randn(N))

        if community_effect_setting is 'mvcar':
            A = np.triu(np.random.randn(C,C))
            data['A'] = A
            data['plot_effect'] = np.dot(data['plot_effect'],A)


    data['theta'] = np.dot(data['X'],data['beta'].T) + data['plot_effect']
    data['phi']   = np.abs(np.random.randn(C,S))

    mu        = np.dot(data['theta']+data['alpha'],data['phi']) + intercept
    data['p'] = 1 / (1 + np.exp(-mu))

    data['Y']   = np.random.binomial(1,data['p'])
    return data

    def spatial_community_regression(X,Y,C,W,setting = 'mvcar',response='bernoulli'):
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
            if setting.lower() == 'none':
                community_effect = 0

            # Uncorrelated conditional autoregressions for each community
            elif setting.lower() == 'car':
                rho             = pm.Uniform('rho', shape = C)
                pe_var          = pm.InverseGamma('pe_var', alpha = 0.01, beta = 0.01, shape = C)
                univariate_CARs = [CAR('car_{0}'.format(c),alpha = rho[c],W=W,tau=pe_var[c,],shape = N,lam=lam) for c in range(C)]
                community_effect= pm.Deterministic('community_effect',pm.math.stack(univariate_CARs,axis = 1))

            # Correlated multivariate conditional autoregression for all communities
            # simultaneously
            elif setting.lower() == 'mvcar':
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
