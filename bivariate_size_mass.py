import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
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
import shen


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

def plot_2dbivar_check(data, coeff, name=None):
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
    
    CS = plt.contour(xpos.T, ypos.T, val.T, 10, cmap=plt.cm.bone, lw=2)
    plt.close()

    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(1,1, wspace=0.0, hspace=0.0)
    
    ax1 = plt.subplot(gs[0,0])
    ax1.set_xlabel('Size [log(kpc)]')
    ax1.set_ylabel('Mass [log(M / M_sun)]')
    ax1.set_title('Stuff')

    k = scipy.stats.gaussian_kde(values)
    xi, yi = np.mgrid[rlims[0]:rlims[1]:50j, mlims[0]:mlims[1]:80j]
    zpos = np.vstack([xi.flatten(), yi.flatten()])
    norm = len(data)/k.integrate_box([rlims[0],mlims[0]],[rlims[1],mlims[1]])

    zi = norm*k(zpos)/fns.veff(.005,0.1,fns.get_sangle())

    img = ax1.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=plt.cm.Oranges)
    plt.axis([rlims[0], rlims[1], mlims[0], mlims[1]])

    levels = [.1, .25, .5, .68, .9, .99]
    #CS = ax1.contour(xpos.T, ypos.T, val.T, 10, cmap=plt.cm.bone, lw=2)
    CS2 = ax1.contour(CS, levels=CS.levels[::2], cmap=plt.cm.bone, 
                      linewidths=1.3)
    cbar = fig.colorbar(img)

    #extents = np.array([rlims[0], rlims[1], mlims[0], mlims[1]])
    #img = ax4.imshow(val.T, cmap=plt.cm.hot_r, origin='lower', 
    #                 interpolation='none', extent=extents)# 
    #ax4.contour(xpos.T, ypos.T, val.T, 10, cmap=plt.cm.bone, lw=2)
    #ax4.set_title('Model')

    plt.tight_layout()
    plt.savefig('bivar_plot2d_%s.png'%name)
    plt.show()
    pdb.set_trace()

def plot_2dbivar_compare(data, coeff, text, title=None, name=None):
    # coeff is a list which contains all the necessary parameters to describe 
    # the bivariate size-mass function. 
    # The last value in the list contains phi_*

    rlims = [0., 1.3]
    mlims = [10., 12.0]
    #rlims = [np.min(data['PETROR50_R_KPC_LOG']), \
    #         np.max(data['PETROR50_R_KPC_LOG']) ]
    #mlims = [np.min(data['MODE']), np.max(data['MODE'])]

    fig = plt.figure(figsize=(10,10))
    #gs = gridspec.GridSpec(2,2, wspace=0.0, hspace=0.0)
    levels=[-4., -3.5, -3., -2.5, -2.]

    i = 0
    for d, c in zip([data[0], data[1]],[coeff[0], coeff[1]]):
        # CALCULATE THE BIVARATE MODEL OVER THE PLANE
        phi = c[5]
        params = c[:5]

        # grid over the parameter space
        xi, yi = np.mgrid[rlims[0]:rlims[1]:50j, mlims[0]:mlims[1]:50j]
        pos = np.vstack([xi.ravel(), yi.ravel()])

        # calculate model values -- these generate contours
        zz = np.log10(np.array([10**(phi)*fns.bivar(x,y,params) for x,y 
                            in zip(pos[0],pos[1])]))
        #zz[zz<-5.] = np.nan
        zi_model = np.reshape(zz, xi.shape)

        # shitty contours based on the model
        #fig2 = plt.figure()
        #CS = plt.contour(xi.T, yi.T, zi_model.T, levels)
        #plt.close()
        
        # KDE OF THE DATA
        values = np.vstack([d['PETROR50_R_KPC_LOG'], d['MODE']])
        k = scipy.stats.gaussian_kde(values)
        norm = len(d)/k.integrate_box([rlims[0],mlims[0]],[rlims[1],mlims[1]])
        zi_data = np.log10(norm*k(pos)/fns.veff(.005,0.1,fns.get_sangle()))


        # BEGIN FIGURE
        ax = fig.add_subplot(211+i)
        if i != 0:
            ax.set_xlabel('log R$_e$ [kpc]', family='serif')
        else:
            if title:
                ax.text(.5, 1.05, title, horizontalalignment='center', 
                        fontsize=20, transform=ax.transAxes)
        ax.set_ylabel('log M [M/M$_{sun}$]', family='serif')
        
        img = ax.pcolormesh(xi, yi, zi_data.reshape(xi.shape),
                            cmap=plt.cm.hot_r, vmin=-4.5, vmax=-2.2)
        #, norm=LogNorm(vmin= zi_data.min(), vmax=zi_data.max())

        plt.axis([rlims[0], rlims[1], mlims[0], mlims[1]])
        
        CS2 = ax.contour(xi.T, yi.T, zi_model.T, levels, cmap=plt.cm.hot_r, 
                         linewidths=2)

        #ticks = 10**np.array([-4.5, -4., -3.5, -3., -2.5, -2.])
        #labels = [format(np.log10(t), '.2f') for t in ticks]
        cbax = fig.colorbar(img)
        #cbax.ax.set_yticklabels(labels)
        from matplotlib import ticker
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbax.locator=tick_locator
        cbax.update_ticks()
        cbax.ax.set_ylabel('log Number Density [Mpc$^{-3}$]', 
                           size=16, family='serif')#dlogM dlogR

        ax.text(.1, 11.7, '%s'%text[i], color='k', size=16)

        i+=1
        
        #pdb.set_trace()
    #extents = np.array([rlims[0], rlims[1], mlims[0], mlims[1]])
    #img = ax4.imshow(val.T, cmap=plt.cm.hot_r, origin='lower', 
    #                 interpolation='none', extent=extents)# 
    #ax4.contour(xpos.T, ypos.T, val.T, 10, cmap=plt.cm.bone, lw=2)
    #ax4.set_title('Model')

    plt.tight_layout()
    plt.savefig('bivar_plot2d_%s_orange.png'%name)
    plt.show()
    pdb.set_trace()

def minimize_plot2d(data, x0, bounds, name=None):
    print name
    coeff = ll2d.minimize(data, 0., 0., x0, bounds, log=name)
    plot_2dbivar_check(data, coeff, name=name)



def main():
    
    # Huang et al. 2013:
    # P = [alpha, Mstar, R_0, sigma_R, beta] = [-1.7, -21., 0.21", 0.7, 0.3]
    
    #print "size limits:", np.log10(np.min(data['PETROR50_R_KPC'])), \
        #    np.log10(np.max(data['PETROR50_R']))
    #print "mass limits:", np.min(data['MODE']), np.max(data['MODE'])
    
    
    pdb.set_trace()
    data = Table.read('zoo2MainSpecz_Ancillary_Masses_cut2.fits')
    
    """
    # Split by concentration index (CI = 2.86)
    CI = data['PETROR90_R']/data['PETROR50_R']
    early1 = np.where(CI > 2.86)
    late1 = np.where(CI < 2.86)
    
    early2 = np.where(CI > 2.6)
    late2 = np.where(CI < 2.6)
    """
    
    #"""
    # Split by u-r color  ( u-r = 2.2)
    ur = data['PETROMAG_U']-data['PETROMAG_R']
    early = np.where(ur < 2.2)
    late = np.where(ur > 2.2)
    #"""
    
    
    """
    # Split by g-r color (g-r = 0.7)
    gr = data['PETROMAG_G']-data['PETROMAG_R']
    early = np.where(gr < 0.7)
    late = np.where(gr > 0.7)
    """
    
    """
    # Split by GZ2 visual morphology
    feat = shen.get_disks(data)
    smooth = shen.get_ellipticals(data)
    """
    
    
    data['PETROR50_R_KPC_LOG'] = np.log10(data['PETROR50_R_KPC'])
    
    # initial parameters (schechter, lognormal)
    # P = (alpha, M_0, beta, R_0, sig) 
    x0 = (-1.5, 10.6, .25, 2., .9)
    bounds_e = [(-2., -0.1),(10.3, 11.5),(.1, 1.0),(.1, 3.),(.2, 3.)]
    bounds_l = [(-2., -0.1),(10.3, 11.5),(.1, 1.0),(.1, 5.),(.2, 3.)] 
    
    name = 'ur_early'
    pdb.set_trace()
    minimize_plot2d(data[early], x0, bounds_e, name=name)
    minimize_plot2d(data[late], x0, bounds_l, name='ur_late')
    
    pdb.set_trace()
    
    #plot_2dbivar_compare([data[late2], data[early2]], [Pl, Pe], 
    #                     ['CI < 2.6', 'CI > 2.6'], name='CI_2.6')


if __name__ == '__main__':
    main()


