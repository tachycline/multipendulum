""" Diagnostic routines for the multipendulum package.
Most are written with parallel execution in mind, so imports are
handled at the function level instead of the module level.
"""

def sampled_lyapunov_estimate(ic, pert=1e-10, nperts=100):
    """make a statistically robust estimate of lyapunov exponent.
    
    sample different perturbations with a single set of bounds.
    Written with imports for ease in using in parallel.
    
    Parameters:
    -----------
    ic : tuple of tuples of floats
         initial conditions for the unperturbed case
         example: ((0.1, 0.1), (0.0, 0.0))         
    pert : float
         magnitude of the perturbation
    nperts : integer
         number of perturbations to put in the ensemble    
    """
    
    import numpy as np
    import multipendulum as mp
    from scipy.optimize import curve_fit
    
    lambdas = []
    powers = []
    logs = 0
    lins = 0
    
    def linearfunc(x, a, b):
        """Linear fit function."""
        return a*x + b
    
    def logfunc(x, a, b):
        """Logarithmic fit function."""
        return a + b*np.log10(x)

    baseline = mp.run_mp(ic)
    
    perts = np.ones(nperts)*pert
    for pert in perts:
        perturbed = mp.run_mp(ic, pert)
            
        deltazvec = baseline.timeseries - perturbed.timeseries
        deltaz = np.sqrt(np.sum(deltazvec**2, axis=1))
            
        lowerthreshold = 1e2*pert
        upperthreshold = 1e7*pert

        # possible outcomes: 
        # 1. we never reach the lower threshold
        #    -> return a slope of zero without trying to fit.
        # 2. we end up between the upper and lower thresholds
        #    -> use the lower threshold and max value as bounaries
        # 3. we pass the upper threshold.
        #    -> use the two thresholds as bounaries
        maxval = max(deltaz)
        loweridx = np.argmax(deltaz > lowerthreshold)
        
        if maxval > lowerthreshold:
            if maxval < upperthreshold: # outcome 2
                upperidx = np.argmax(deltaz)
            else: # outcome 3
                upperidx = np.argmax(deltaz > upperthreshold)
        
            xdata = baseline.times[loweridx:upperidx]
            ydata = np.log10(deltaz[loweridx:upperidx])
            
            # linear fit for exponential growth
            try:
                popt, pcov = curve_fit(linearfunc, xdata, ydata)
            except: # the fit fails for some reason
                popt = [0,0]
            lambdas.append(popt[0])
            linres = ydata - linearfunc(xdata, *popt)
            linerror = linres.dot(linres)
            
            # log fit for power law growth
            try:
                poptplaw, pcovplaw = curve_fit(logfunc, xdata, ydata)
            except:
                poptplaw = [0,0]
            powers.append(poptplaw[0])
            logres = ydata - logfunc(xdata, *poptplaw)
            logerror = logres.dot(logres)

            if logerror < linerror:
                logs += 1
            else:
                lins += 1
                
        else: # outcome 1 above
            lambdas.append(0)
            powers.append(0)
            
    lambdas = np.array(lambdas)
    powers = np.array(powers)
        
    pmean = np.mean(powers)
    pstd = np.std(powers)
    lmean = np.mean(lambdas)
    lstd = np.std(lambdas)
    energy = baseline.efuncs['E'](*(np.array(ic).flat))
    
    output = {"pmean" : pmean,
              "pstd" : pstd/np.sqrt(len(powers)),
              "lmean" : lmean,
              "lstd" : lstd/np.sqrt(len(lambdas)),
              "logs" : logs,
              "lins" : lins,
              "energy" : energy,
              "ic" : ic}
    return output


# box counting dimension
def count_boxes(pend, nside):
    cols = list(pend.q) + list(pend.p)

    boxids = []
    for col in cols:
        minval = pend.timedf[col].min()
        maxval = pend.timedf[col].max()
        delta = maxval - minval
        spacing = delta/nside
        boxids.append(((pend.timedf[col]-minval)//spacing).astype(int))
    N = len(set(zip(*boxids)))
    
    return np.log(N)/np.log(1/spacing)

def MB_dimension(ic):
    """Estimate the Minkowski–Bouligand dimension by box counting
    
    Parameters:
    -----------
    ic : tuple of tuples of floats
        initial conditions for the system
        example: ((0.1, 0.1), (0.0, 0.0))

    Returns:
    --------
    float The estimated box counting dimension
    """
    import numpy as np
    import multipendulum as mp
    from scipy.optimize import curve_fit
    
    pend = mp.run_mp(ic, tmax=1000, nsteps=100000)
    nsides = np.linspace(5, 75, 25).astype(int)
    dim = [count_boxes(double, nside) for nside in nsides]
    minval = double.timedf[col].min()
    maxval = double.timedf[col].max()
    delta = maxval - minval

    popt, pcov = curve_fit(linear, delta/nsides, dim)
    return popt[1]

