import math
import numpy as np
import scipy.stats
import scipy.optimize
import scipy.integrate
import cosmolopy.distance as cd
import cosmolopy.constants as cc
import pdb

#import nondet as nd
#import limits as lims

cosmo = {'omega_M_0':0.3,'omega_lambda_0':0.7,'h':0.7}
cosmo = cd.set_omega_k_0(cosmo)
# Sorta arbitrary value of M_0
pivot = 10.6

class Memoize:
    """
    Define a class: Memoize 
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not args in self.cache: self.cache[args] = self.func(*args)
        return self.cache[args]

    def reset(self):
        self.cache={}
        
def poisson_interval(k, sigma):
    """
    Returns the poisson error interval for N=k and 
    2-sided CL=sigma using chisquared ppf expression
    """
    a = 1.-sigma
    lolim = scipy.stats.chi2.ppf(a/2, 2*k) / 2
    uplim = scipy.stats.chi2.ppf(1-a/2, 2*k + 2) / 2
    lolim[k==0] = 0.
    return np.array([lolim,uplim])
    
def get_quad_args():
    """
    Returns parameters for **quad_args
    """
    epsa, epsr, subdiv = 1e-6, 1e-6, 250
    quad_args = {'limit':subdiv,'epsrel':epsr,'epsabs':epsa}
    return quad_args

def lum_dist(z):
    """
    Return the luminosity distance as redshift z
    in units of 'cm'.
    """
    return cd.luminosity_distance(z,**cosmo) * cc.Mpc_cm
    
def co_vol(z):
    """
    Returns the differential comoving volume
    at redshift z in units of 'Mpc^3'.
    """
    return cd.diff_comoving_volume(z,**cosmo)

def power_law(mass, params):
    """
    From Shen 2003
    <R> = R_0*(M/M_sun)^beta
    mass should be in units of M_sun?
    """
    [R_0, beta] = params
    return R_0*(mass)**beta
    
def calc_R_bar(M,beta,R_0):
    """
    Returns log10(R_bar) for use in log-normal / bivariate fnc

    From Huang et al 2013 and Shen 2003:
    R_bar = R_0*(M/M_0)^beta

    log version:
    log10(R_bar) = log10(R_0) + beta*(log10(M) - log10(M_0))

    where R_0 and beta are fitting parameters;
    M_0 is arbitrarily choosen (as in Shen2003) to be 10.6
    """
    #return beta*(M - pivot) + math.log10(R_0)# + pivot
    return beta*M + math.log10(R_0)

def exp_o3(ha,power,ratio):
    """
    Returns the expected OII lum given Ha lum
    and line parameters
    """
    return (ha - pivot - math.log10(ratio))/power + pivot

def gauss(x,x0,sig):
    """
    Returns the evaluation at x of a Gaussian centered
    at x0 with stdev sig
    """
    return np.exp( -0.5 * (x-x0) * (x-x0) / sig / sig )
    
def log_gauss(x,x0,sig):
    """
    Returns the evaluation at x of a log-Gaussian
    centered at x0 with stdev sig
    """
    return np.exp( -0.5 * np.log(10)**2 * (x-x0)*(x-x0) / sig/sig )

def sch_shape(M, alpha, Mstar):
    """
    Returns the UNnormalized MASS Schechter Function shape
    given M, M*, and alpha (faint end slope)
    """
    x = math.pow(10., M - Mstar)
    return math.log(10) * math.pow(x,alpha+1.) * math.exp(-x)

def sch(o3, P):
    """
    Returns the UNnormalized Schechter Function value
    given a luminosity and a set of parameters
    """
    (alpha, Lst) = P
    return sch_shape(o3, alpha, Lst)
    
def norm_sch(x, alpha, Lst, phi):
    """
    Returns the Normalized Schechter Function value
    given a luminosity and a set of parameters
    """
    x = np.power(10.,x - Lst)
    return 10**phi * np.log(10) * np.power(x,alpha+1.) * np.exp(-x)
    
def bivar(R,M,P):
    """
    Returns the bivariate func evaluation given
    Ha and OIII luminosities (in log10) and a set of parameters
    
    P = ( alpha, Mstar, beta, R_0, sig )
    """
    (alpha, Mstar, beta, R_0, sig) = P

    sch_term = sch_shape(M, alpha, Mstar)

    R_bar = calc_R_bar(M, beta, R_0)
    gauss_term = log_gauss(R, R_bar, sig) * math.log(10.) \
                 /math.sqrt(2.*math.pi)/sig
    #print "R:", R, "R_bar:", R_bar, "Gauss term:",gauss_term

    return sch_term * gauss_term

def get_sangle( n_fields=1 ):
    """
    Returns the solid angle for the survey
    """
    sr_in_deg2 = (math.pi/180.)**2
    #s_angle = (3.5/3600.) * sr_in_deg2 * n_fields
    #return s_angle
    
    return 8400.*sr_in_deg2

def collapse_blf(P, phi, correct, sig=2.5):
    """
    Returns a function that collapses the bivariate LF to Halpha 
    dimension
    """
    integrand = lambda o3, ha: bivar(ha, o3, P) * (10**phi)    
    ndcor_func = nd.get_ndcor_ha_func(sig=sig)

    def collapse(ha):
        lolim, hilim = 35., 50.
        if correct: return scipy.integrate.quad(integrand, lolim, 
                                hilim, args=(ha,))[0] * ndcor_func(ha)
        else: return scipy.integrate.quad(integrand, lolim, 
                                          hilim, args=(ha,))[0]
              
    return np.vectorize(collapse)

def veff(z0,z1,s_angle,L=None,flim=None):
    """
    Returns the effective volume in the redshift range [z0,z1] for a given L
    """    
    def criterion(z,L):
        return math.log10(lims.get_flux(L,z)) - math.log10(flim)

    if flim is not None:
        vol = L*0
        for i in range(len(L)):
            znew = min( z1, scipy.optimize.brentq( criterion, 0.01, 10, args=(L[i]),) )
            if znew > z0: vol[i] = scipy.integrate.quad(co_vol, z0, znew)[0] * s_angle
            else: vol[i] = np.NaN
    else: vol = scipy.integrate.quad(co_vol, z0, z1)[0] * s_angle

    return vol 

def find_levels(hist, conf):
    """
    Returns levels at which requested confidence is reached.
    """
    if (np.sum(hist) - 1.) > 1e-6:
        raise Exception("Error in find_levels(): Provided histogram is not normalized.")
        
    levels = []
    crit = lambda x,c: np.sum(hist[hist >= x]) - c
    for c in conf:
        level = scipy.optimize.brentq(crit, 0, 1, args=(c,))
        levels.append(level)
    return levels  
