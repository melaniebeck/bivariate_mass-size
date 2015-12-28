import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
import scipy.optimize as opt
from astropy.table import Table
import pdb
from cosmolopy.luminosityfunction import schechterL
import likelihood_2d as ll2d
import probability_2d as prob2d
import functions as fns
import scipy.stats
import scipy.integrate
from scipy.stats import kde

plt.rc('font', **{'size':16.,'weight':400})
plt.rc('axes', **{'linewidth': 2.0})
plt.rc('xtick', **{'major.width': 1.2})
plt.rc('ytick', **{'major.width': 1.2})

def schechter(mass, alpha, Mstar):
    x = np.power(10., mass-Mstar)
    return  np.log(10.)*np.power(x,alpha+1)*np.exp(-x)

def lognormal(mass, size, sigma_R, R_0, beta):
    R_bar = peak_of_size_dist(mass, R_0, beta)
    logfunc = (np.log10(size/R_bar)/sigma_R)**2
    return 1/(size*sigma_R*np.sqrt(2*np.pi)) * np.exp(-0.5*logfunc)

def peak_of_size_dist(mass, R_0, beta, M_0=10.6):
    return R_0*(M/M_0)**beta

def calculate_R_bar(mass, R_0, beta, M_0=10.6):
    return np.log10(R_0) + beta*(mass - M_0) 

def bivariate_mass_size(mass, size, alpha, Mstar, beta, R_0, sigma_lnR):
    a = np.power(10., mass-Mstar)
    b = np.log(10)/(sigma*np.sqrt(2*np.pi))

    R_bar = calculate_R_bar(mass, M_0, R_0, beta)

    c = (size-R_bar)**2/(2*(sigma_lnR/np.log(10))**2)

    return np.log(10)*np.power(x,alpha+1)*np.exp(-x)*b*np.exp(-c)

def plot_2d_bivar(data, coeff, name=None):
    # coeff is a list which contains all the necessary parameters to describe 
    # the bivariate size-mass function. 
    # The last value in the list contains phi_*
    phi = coeff[5]
    params = coeff[:5]
    print phi, params

    rlims = [np.min(data['PETROR50_R_KPC_LOG']), \
             np.max(data['PETROR50_R_KPC_LOG']) ]
    #rlims = [0., 1.3]
    mlims = [np.min(data['MODE']), np.max(data['MODE'])]
    #mlims = [10., 12.0]

    # grid over the parameter space
    xpos, ypos = np.mgrid[rlims[0]:rlims[1]:50j, mlims[0]:mlims[1]:80j]
    pos = np.vstack([xpos.ravel(), ypos.ravel()])
    zpos = np.array([10**(phi)*fns.bivar(x,y,params) for x,y 
                     in zip(pos[0],pos[1])])
    #val = np.reshape(np.log10(zpos), xpos.shape)
    val = np.reshape(zpos, xpos.shape)

    values = np.vstack([data['PETROR50_R_KPC_LOG'], data['MODE']])
    

    fig = plt.figure(figsize=(10,10))

    ax3 = fig.add_subplot(221)
    extents = np.array([mlims[0], mlims[1], rlims[0], rlims[1]])
    img = ax3.imshow(val, cmap=plt.cm.hot_r, origin='lower', 
                     interpolation='none', extent=extents)  #
    ax3.contour(ypos, xpos, val, 10, cmap=plt.cm.bone, lw=2)
    cbar = fig.colorbar(mappable=img)
    ax3.set_title('Model')

    ax = fig.add_subplot(222)
    H, x_, y_ = np.histogram2d(data['PETROR50_R_KPC_LOG'], data['MODE'], 
                               bins=25, normed=True)
    img = ax.imshow(H, origin='lower', interpolation='none', 
                    extent=extents, cmap=plt.cm.Blues)
    levels = [-2., -2.5, -3.0, -3.5, -4., -4.5, -5.]
    ax.contour(ypos, xpos, val, 10, cmap=plt.cm.bone, lw=2)
    plt.ylabel('Size [log(kpc)]')
    plt.xlabel('Mass [log(Stellar Mass)]')
    plt.title('data (hist2d)')
    cbar = fig.colorbar(mappable=img)
    
    #--------------------------------------------------------------------#

    ax4 = fig.add_subplot(223)
    extents = np.array([rlims[0], rlims[1], mlims[0], mlims[1]])
    img = ax4.imshow(val.T, cmap=plt.cm.hot_r, origin='lower', 
                     interpolation='none', extent=extents)# 
    ax4.contour(xpos.T, ypos.T, val.T, 10, cmap=plt.cm.bone, lw=2)
    ax4.set_title('Model')

    ax2 = fig.add_subplot(224)
    plt.xlabel('Size [log(kpc)]')
    plt.ylabel('Mass [log(Stellar Mass)]')
    plt.title('data (Gaussian KDE)')

    k = scipy.stats.gaussian_kde(values)
    xi, yi = np.mgrid[rlims[0]:rlims[1]:50j, mlims[0]:mlims[1]:80j]
    zpos = np.vstack([xi.flatten(), yi.flatten()])
    norm = len(data)/k.integrate_box([rlims[0],mlims[0]],[rlims[1],mlims[1]])

    zi = norm*k(zpos)/fns.veff(.005,0.1,fns.get_sangle())


    img = ax2.pcolormesh(xi, yi, zi.reshape(xi.shape))
    plt.axis([rlims[0], rlims[1], mlims[0], mlims[1]])
    #levels = [-2., -2.5, -3.0, -3.5, -4., -4.5, -5.]
    ax2.contour(xpos.T, ypos.T, val.T, 10, cmap=plt.cm.bone, lw=2)
    cbar = fig.colorbar(img)

    plt.tight_layout()
    plt.savefig('bivar_plot2d_%s.png'%name)
    plt.show()
    pdb.set_trace()

def minimize_plot2d(data, x0, bounds, name=None):
    coeff = ll2d.minimize(data, 0., 0., x0, bounds, log=name)
    plot_2d_bivar(data, coeff, name=name)


# Huang et al. 2013:
# P = [alpha, Mstar, R_0, sigma_R, beta] = [-1.7, -21., 0.21", 0.7, 0.3]

data = Table.read('zoo2MainSpecz_Ancillary_Masses_cut2.fits')

# Split by concentration index (CI = 2.86)
CI = data['PETROR90_R']/data['PETROR50_R']
early1 = np.where(CI > 2.86)
late1 = np.where(CI < 2.86)

early2 = np.where(CI > 2.6)
late2 = np.where(CI < 2.6)

print "size limits:", np.log10(np.min(data['PETROR50_R_KPC'])), \
    np.log10(np.max(data['PETROR50_R']))
print "mass limits:", np.min(data['MODE']), np.max(data['MODE'])

data['PETROR50_R_KPC_LOG'] = np.log10(data['PETROR50_R_KPC'])

# initial parameters (schechter, lognormal)
# P = (alpha, M_0, beta, R_0, sig) 
x0 = (-1.5, 10.6, .25, 2., .9)
bounds_e = [(-2., -0.1),(10.3, 11.5),(.1, 1.0),(.1, 3.),(.2, 3.)]
bounds_l = [(-2., -0.1),(10.3, 11.5),(.1, 1.0),(.1, 5.),(.2, 3.)] 

#P = (-0.10000000000000001, 11.5, 2.0, 4.75, 0.20000000000000001)
#num = fns.bivar(3.49, 12.37, P)
#print num
#pdb.set_trace()

minimize_plot2d(data[early2], x0, bounds_e, name='e2_run1')
minimize_plot2d(data[late2], x0, bounds_l, name='l2_run1')


#coeff = [-0.1, 10.7304885784, 0.1952174371, 3.5563765795, 0.3685764135]
#phi = prob2d.phi(coeff, data)
#phi = -2.6400451783
#coeff.append(phi)

pdb.set_trace()
