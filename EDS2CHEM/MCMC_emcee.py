# %% 
import numpy as np
import scipy.interpolate as interpolate

def draw_proj_box(img, x0, y0, x1, y1, phi = 45, value = 1, pstep = 1):
    """Function to draw a vector between two positions in an image as well 
    as a projection box at a given angle and of a given size in order to
    extract the data as a profile. Median filter is used for the final
    profile across the projection.

    Args:
        img (_type_): _description_
        x0 (_type_): _description_
        y0 (_type_): _description_
        x1 (_type_): _description_
        y1 (_type_): _description_
        phi (int, optional): _description_. Defaults to 45.
        value (int, optional): _description_. Defaults to 1.
        pstep (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    v = np.asarray([x1, y1]) - np.asarray([x0,y0])
    theta = np.arccos(v[1]/(np.sqrt(v[1]**2+v[0]**2)))
    phi = theta + np.pi*(phi/180)
    #phi = theta + phi
    perp = np.asarray([np.sin(phi), np.cos(phi)])
    #perp = np.asarray([-v[1], v[0]])/(np.sqrt(v[1]**2+v[0]**2))#
    max_val = np.max(img[~np.isnan(img)])

    mat = np.zeros_like(img)
    step_number  = int(abs(x0-x1) + abs(y0-y1)) #Number of steps
    if step_number < 2:
            step_number = 2
    else:
            pass
    step_size = 1.0/step_number #Increment size
    p = [] #Point array (you can return this and not modify the matrix in the last 2 lines)
    t = 0.0 #Step current increment
    for i in range(step_number):
        p.append([int(round(x1 * t + x0 * (1 - t))), int(round(y1 * t + y0 * (1 - t)))])
        t+=step_size

    conc = []
    #create array of all steps perpendicular to vector between two end points
    nsteps = np.linspace(-pstep, pstep, 2*pstep+1)
    for item in p:
        mat = np.zeros_like(img)
        #coords perpendicular to vector
        coords = np.asarray([item+j*perp for j in nsteps], dtype = "int64")
        #get all vals
        mat[coords[:,0], coords[:,1]] = value
        vals = img[mat.astype("bool")]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
             vals = np.asarray([-10*max_val])
        else:
             pass
        #only want median to get rid of noise
        conc.append(np.median(vals))
    
    return np.asarray(conc)

def align_once(inputmatrix, Nvalues, theta):
    """ Simplified version of the align function below - this is very 
    useful to extract data from an image post-MCMC fitting to showcase the
    different/varierty of samples that form the posterior.

    Args:
        inputmatrix (_type_): _description_
        Nvalues (_type_): _description_
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    n = Nvalues
    ax = int(theta[0])
    ay = int(theta[1])
    bx = int(theta[2])
    by = int(theta[3])

    vals = draw_proj_box(inputmatrix, ax, ay, bx, by, 90, pstep = 5)
    steps = np.linspace(0, len(vals)-1, len(vals))
    steps = steps[~np.isnan(vals)]
    vals = vals[~np.isnan(vals)]
    if 0 in steps:
        pass
    else:
        if steps[0]-1 != 0:
            steps = np.concatenate([[0, steps[0]-1], steps])
            vals = np.concatenate([[0,0], vals])
        else:
            steps = np.concatenate([[0], steps])
            vals = np.concatenate([[0], vals])

    interpolation = interpolate.interp1d(steps, vals)
    trial_x = interpolation(np.linspace(0, len(vals)-1, n))

    return trial_x


def align(inputmatrix, Nvalues,theta, **kwargs):
    """Function used during the MCMC fitting procedure to form the
    joint posterior distribution of samples. Samples are drawn to 
    extract the profile using function above. Then the straight line model
    is used to model the calibration using sampled parameters.

    Args:
        inputmatrix (_type_): _description_
        Nvalues (_type_): _description_
        theta (_type_): _description_

    Returns:
        _type_: _description_
    """
    slope = theta[5]
    intercept = theta[6]
    #ax, ay, bx, by, ww, slope, intercept = P[0:Nvalues]
    ww = int(theta[4])
    n = Nvalues
    ax = int(theta[0])
    ay = int(theta[1])
    bx = int(theta[2])
    by = int(theta[3])
    phi = theta[7]

    vals = draw_proj_box(inputmatrix, ax, ay, bx, by, phi, pstep = ww)
    steps = np.linspace(0, len(vals)-1, len(vals))
    steps = steps[~np.isnan(vals)]
    vals = vals[~np.isnan(vals)]
    if 0 in steps:
        pass
    else:
        if steps[0]-1 != 0:
            steps = np.concatenate([[0, steps[0]-1], steps])
            vals = np.concatenate([[0,0], vals])
        else:
            steps = np.concatenate([[0], steps])
            vals = np.concatenate([[0], vals])

    interpolation = interpolate.interp1d(steps, vals)
    trial_x = interpolation(np.linspace(0, len(vals)-1, n))

    return trial_x*slope + intercept


class logfuncs():
    """
    New object created to house all the likelihood functions used
    during the MCMC fitting procedure.
    """
     
    def __init__(self,pmin, pmax):
          self.pmin = pmin
          self.pmax = pmax


    def log_prior(self,theta):
        """
        Log-likelihood function for the prior enforcing the bounds to each
        parameter as set by the user.

        Args:
            theta (_type_): _description_

        Returns:
            _type_: _description_
        """
        ax, ay, bx, by, ww, m, b, phi = theta
        if self.pmin[0] < ax < self.pmax[0] and self.pmin[1] < ay < self.pmax[1] and self.pmin[2] < bx < self.pmax[2] and self.pmin[3] < by < self.pmax[3] and self.pmin[4] < ww < self.pmax[4] and self.pmin[5] < m < self.pmax[5] and self.pmin[6] < b < self.pmax[6] and self.pmin[7] < phi <self.pmax[7]:
            return 0.0
        return -np.inf

    def log_likelihood(self,theta, x, y, yerr):
        """Model log-likelihood based on model fit. We use
        a gaussian likelihood due to its simple nature.

        Args:
            theta (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            yerr (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            model = align(x, len(y), theta )
        except IndexError:
            return -np.inf
        sigma2 = yerr**2 + model**2
        #sigma2 = model**2
        return -0.5 * np.sum(np.divide((y - model)**2, yerr**2))
    

    def log_probability(self,theta, x, y, yerr):
        """ Overall log probability of sampled parameters, taking into account
        both the modelling and the prior likelihoods.

        Args:
            theta (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            yerr (_type_): _description_

        Returns:
            _type_: _description_
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, yerr)
    
class simple_logfuncs():
    """Separate object for the likelihood functions when only sampling
    for straight line fit; a simple calibration scenario.
    """
     
    def __init__(self,pmin, pmax):
          #super().__init___(pmin, pmax)
          self.pmin = pmin
          self.pmax = pmax


    def log_prior(self,theta):
        """
        Log-likelihood function for the prior enforcing the bounds to each
        parameter as set by the user.

        Args:
            theta (_type_): _description_

        Returns:
            _type_: _description_
        """
        m, b = theta
        if self.pmin[0] < m < self.pmax[0] and self.pmin[1] < b < self.pmax[1]:
            return 0.0
        return -np.inf

    def log_likelihood(self,theta, x, y, yerr):
        """Model log-likelihood based on model fit. We use
        a gaussian likelihood due to its simple nature.

        Args:
            theta (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            yerr (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            model = x*theta[0] + theta[1]
        except IndexError:
            return -np.inf
        sigma2 = yerr**2 + model**2
        #sigma2 = model**2
        return -0.5 * np.sum(np.divide((y - model)**2, yerr**2))
    

    def log_probability(self,theta, x, y, yerr):
        """ Overall log probability of sampled parameters, taking into account
        both the modelling and the prior likelihoods.

        Args:
            theta (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            yerr (_type_): _description_

        Returns:
            _type_: _description_
        """
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, yerr)
    

def MCMC_run(x,y, uncert, params, pmin, pmax, return_ = False, name = "mcmc"):
    """Auxiliary function to tie all the others together and facilitate the MCMC
    fitting procedure using the emcee library. Parameters are initialized based on a maximum 
    likelihood fit around the initial parameters set by the user. Samples are then drawn as
    according to the MCMC procedure to create the final joint posterior distribution for the
    calibration model.

    Args:
        x (_type_): _description_
        y (_type_): _description_
        uncert (_type_): _description_
        params (_type_): _description_
        pmin (_type_): _description_
        pmax (_type_): _description_
        return_ (bool, optional): _description_. Defaults to False.
        name (str, optional): _description_. Defaults to "mcmc".

    Returns:
        _type_: _description_
    """

    names   = ["ax","ay", "bx", "by", "ww", "m", "b", "phi"]

    funcs = logfuncs(pmin,pmax)
    from scipy.optimize import minimize
    nll = lambda *args: -1*funcs.log_likelihood(*args)
    initial = np.array(params) + 0.1 * np.random.randn(8)
    soln = minimize(nll, initial, args=(x, y, uncert))
    
    print(soln.x)

    import emcee
    from multiprocessing import Pool
    pos = np.asarray(soln.x) + 1e-5 * np.random.randn(17, 8)
    nwalkers, ndim = pos.shape

    filename = str(name) + ".h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
        nwalkers, ndim, funcs.log_probability, args=(x, y, uncert), backend = backend, pool = pool
        )
        sampler.run_mcmc(pos, 10000, progress=True)
    #, moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2)]
    samples = sampler.get_chain()
    import pickle
    with open("emcee_res.pkl", "wb") as f:
         pickle.dump(samples, f)

    if return_ is True:
         return sampler


def Simple_MCMC_run(x,y, uncert, params, pmin, pmax, return_ = False, name = "mcmc"):
    """Auxiliary function largely identical to above used for the MCMC fit of a simple calibration
    model where only the straight line fit between the two variables are fitted - not positions or anything else.
    Maximum likelihood parameters around the user's initial guess are used to initialize the model; note
    only "m" and "c" parameters are ever changed.

    Args:
        x (_type_): _description_
        y (_type_): _description_
        uncert (_type_): _description_
        params (_type_): _description_
        pmin (_type_): _description_
        pmax (_type_): _description_
        return_ (bool, optional): _description_. Defaults to False.
        name (str, optional): _description_. Defaults to "mcmc".

    Returns:
        _type_: _description_
    """

    names   = ["ax","ay", "bx", "by", "ww", "m", "b", "phi"]

    funcs = simple_logfuncs(pmin,pmax)
    from scipy.optimize import minimize
    nll = lambda *args: -1*funcs.log_likelihood(*args)
    initial = np.array(params) + 0.1 * np.random.randn(2)
    soln = minimize(nll, initial, args=(x, y, uncert))
    
    print(soln.x)

    import emcee
    from multiprocessing import Pool
    pos = np.asarray(soln.x) + 1e-5 * np.random.randn(17, 2)
    nwalkers, ndim = pos.shape

    filename = str(name) + ".h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
        nwalkers, ndim, funcs.log_probability, args=(x, y, uncert), backend = backend, pool = pool
        )
        sampler.run_mcmc(pos, 10000, progress=True)
    #, moves=[(emcee.moves.DEMove(), 0.8),(emcee.moves.DESnookerMove(), 0.2)]
    samples = sampler.get_chain()
    import pickle
    with open("emcee_res.pkl", "wb") as f:
         pickle.dump(samples, f)

    if return_ is True:
         return sampler