import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from scipy.stats import kde
from astropy.table import Table

import likelihood_2d as ll2d
import functions as fns
import probability_2d as prob2d

plt.rc('font', **{'size':24.,'weight':540})
plt.rc('axes', **{'linewidth': 3.9})
plt.rc('lines', **{'linewidth':3.})
plt.rc('xtick', **{'major.width': 2.5})
plt.rc('ytick', **{'major.width': 2.5})


def get_disks(data):
    featured = np.where(
        (data['t01_smooth_or_features_a02_features_or_disk_debiased']
         >=0.430) &
        (data['t01_smooth_or_features_a02_features_or_disk_count']>=10) )
        #(data['t02_edgeon_a05_no_debiased']>=0.715) &
    return featured

def get_ellipticals(data):
    ellipts = np.where(
        (data['t01_smooth_or_features_a01_smooth_count']>=10) &
        (data['t01_smooth_or_features_a01_smooth_debiased']>=0.463) )
    return ellipts

def plot_2dbivar_check(data, coeff, name=None):
    # coeff is a list which contains all the necessary parameters to describe 
    # the bivariate size-mass function. 
    # The last value in the list contains phi_*
    phi = coeff[5]
    params = coeff[:5]

    rlims = [np.min(data['PETROR50_R_KPC_LOG']), \
             np.max(data['PETROR50_R_KPC_LOG']) ]
    #rlims = [0., 1.3]
    mlims = [np.min(data['MODE']), np.max(data['MODE'])]
    #mlims = [10., 12.0]

    levels=[-4., -3.5, -3., -2.5, -2.]

    # grid over the parameter space
    xi, yi = np.mgrid[rlims[0]:rlims[1]:50j, mlims[0]:mlims[1]:50j]
    pos = np.vstack([xi.ravel(), yi.ravel()])
    
    # calculate model values -- these generate contours
    zz = np.log10(np.array([10**(phi)*fns.bivar(x,y,params) for x,y 
                            in zip(pos[0],pos[1])]))
    zi_model = np.reshape(zz, xi.shape)
        
    # KDE OF THE DATA
    values = np.vstack([data['PETROR50_R_KPC_LOG'], data['MODE']])
    k = kde.gaussian_kde(values)
    norm = len(data)/k.integrate_box([rlims[0],mlims[0]],[rlims[1],mlims[1]])
    zi_data = np.log10(norm*k(pos)/fns.veff(.02,0.1,fns.get_sangle()))

    #------------------- BEGIN FIGURE ------------------------
    fig = plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(1,1, wspace=0.0, hspace=0.0)
    
    ax1 = plt.subplot(gs[0,0])
    ax1.set_xlabel('Size [log(kpc)]')
    ax1.set_ylabel('Mass [log(M / M_sun)]')

    img = ax1.pcolormesh(xi, yi, zi_data.reshape(xi.shape), 
                         cmap=plt.cm.afmhot_r, vmin=-4.5, vmax=-2.2)
    plt.axis([rlims[0], rlims[1], mlims[0], mlims[1]])

    #levels = [.1, .25, .5, .68, .9, .99]
    CS = ax1.contour(xi.T, yi.T, zi_model.T, levels, cmap=plt.cm.bone, lw=2)

    #ax2 = plt.subplot(gs[0,1])
    #ax2.pcolormesh(xi, yi, zi_model.T.reshape(xi.shape), cmap=plt.cm.Oranges)
    #ax2.axis([rlims[0], rlims[1], mlims[0], mlims[1]])
    #CS = plt.contour(yi.T, xi.T, zi_model.T, 10, cmap=plt.cm.bone, lw=2)

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

    fig = plt.figure(figsize=(12,15))
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
        k = kde.gaussian_kde(values)
        norm = len(d)/k.integrate_box([rlims[0],mlims[0]],[rlims[1],mlims[1]])
        zi_data = np.log10(norm*k(pos)/fns.veff(.02,0.1,fns.get_sangle()))


        # BEGIN FIGURE
        ax = fig.add_subplot(211+i)
        if i != 0:
            ax.set_xlabel('log R$_{50}$ [kpc]', family='serif')
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
                         linewidths=3.)

        cbax = fig.colorbar(img)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbax.locator=tick_locator
        cbax.update_ticks()
        cbax.ax.set_ylabel('log Number Density [Mpc$^{-3}$] dlogM dlogR', 
                           size=18, family='serif')#

        ax.text(.1, 11.7, '%s'%text[i], color='k', size=26, weight=535)

        i+=1
        

    plt.tight_layout()
    plt.savefig('bivar_plot2d_%s_orange.png'%name)
    plt.show()
    #pdb.set_trace()


def minimize_plot2d(data, x0, bounds, name=None):
    coeff = ll2d.minimize(data, 0., 0., x0, bounds, log=name)
    plot_2dbivar_check(data, coeff, name=name)


data = Table.read('zoo2MainSpecz_Ancillary_Masses_cut2.fits')
data['PETROR50_R_KPC_LOG'] = np.log10(data['PETROR50_R_KPC'])


# Split by concentration index (CI = 2.86)
#"""
CI = data['PETROR90_R']/data['PETROR50_R']
"""
early = np.where(CI > 2.6)
late = np.where(CI < 2.6)

##Pe = [-0.1, 10.83284494, 0.378171563, 2.780258918, 0.2670760691,-2.8937194140]
##Pl = [-0.40963428, 10.6243711,0.237021830, 4.619651378,0.30432605,-3.02032141]
Pe = [-0.40351359,10.97414469,0.37482638,0.000296086,0.26746686,-2.97779065]
Pl = [-0.4096560,10.623896095, 0.23694097, 0.0142239710, 0.30433650,-3.020135]
plot_2dbivar_compare([data[late],data[early]], [Pl, Pe], 
                     ['CI < 2.6  (late type)', 'CI > 2.6  (early type)'], 
                     title=None, name='CIsplit_new')
#"""

# Split by u-r color  ( u-r = 2.2)
#"""
ur = data['PETROMAG_U']-data['PETROMAG_R']
"""
late = np.where(ur < 2.2)
early = np.where(ur > 2.2)

#Pe = [-0.1, 10.8378753402, 0.3772260140,2.9107492119,0.300851208, -2.84588877]
#Pl = [-0.1, 10.4059393342, 0.2316821481,4.6239232272, 0.3064851250,-2.9966379]
#Pe = [-0.47101960,10.8577362, 0.297429549, 0.0021207456,0.29967796,-2.91540652]
Pe = [-0.43384542,10.9261923, 0.387708418, 0.000225180,0.29540305,-2.9253182]
Pl = [-0.07625976,10.3997770, 0.23169741,0.016181267,0.306473909,-2.99497934]
#plot_2dbivar_compare([data[late], data[early]], [Pl, Pe],
#                     ['u - r < 2.2  (late type)', 'u - r > 2.2  (early type)'],
#                     title=None, name='ursplit_new2')
#"""

# Split by g-r color (g-r = 0.7)
#"""
gr = data['PETROMAG_G'] - data['PETROMAG_R']
"""
early = np.where(gr > 0.7)
late = np.where(gr < 0.7)

##Pe = [-0.1,10.8304478349,0.3842122932,2.9281852988,0.2922444503,-2.8204111540]
##Pl = [-0.244921, 10.3962361, 0.2485336754, 4.816482442, 0.307306366,-3.036758]
Pe = [-0.01, 10.8004062, 0.38439804,0.00024653, 0.292302049, -2.8066131092]
Pl = [-0.2453965,10.396375777, 0.24850780,0.011182265,0.307247042,-3.036821]
plot_2dbivar_compare([data[late],data[early]], [Pl, Pe], 
                     ['g - r < 0.7  (late type)', 'g - r > 0.7  (early type)'],
                     title=None, name='grsplit_new')
#"""

# Split by GZ2  morphology
"""
late = get_disks(data)
early = get_ellipticals(data)

#Pe = [-0.1, 10.8059143, 0.3310917164,2.8291546253, 0.303010387, -3.037072950]
#Pl = [-0.1, 10.67395533, 0.157066960,4.2076078381, 0.3543562389, -2.94510106]
Pe = [-0.3147779,10.894069815,0.33179461,0.0008600825,0.30209906,-3.08789534]
Pl = [-0.01,10.64676755,0.157014841,0.091133216,0.3543111375,-2.9341713953]
plot_2dbivar_compare([data[late],data[early]], [Pl, Pe], 
                     ['GZ2: Featured', 'GZ2: Smooth'], 
                     title=None, name='GZ2split_new')
#"""

# ========= COMPARE PARAMETERS WITH LITERATURE =============
xx = np.linspace(10.e9,10.e13,100)
xxs = np.linspace(10.e9, 10.e11, 100)
xxl = np.linspace(10.e7, 10.e11, 100)

# early types
e_shen = [2.88e-6, 0.56]
e_lange_ur = [7.32e-5, .44]
e_lange_gi = [8.25e-5, .44]
e_lange_vis = [4.19e-5, .46]
my_ur_e = [0.000225180, 0.387708418]
my_gi_e = [0.00024653, 0.38439804]
my_vis_e = [0.0008600825, 0.33179461]
my_ci_e = [0.000296086,0.37482638]
earlys = [np.where(CI > 2.6), np.where(ur > 2.2), np.where(gr > 0.7),  
          get_ellipticals(data)]
pars_early = [e_shen, e_lange_ur, e_lange_gi, e_lange_vis]
mypars_early = [my_ci_e, my_ur_e, my_gi_e, my_vis_e]


# late types
l_lange_ur = [13.63e-3, 0.25]
l_lange_gi = [13.98e-3, 0.25]
l_lange_vis = [37.24e-3, 0.20]
l_lange_sersic = [27.72e-3, 0.21]
my_ci_l = [0.0142239710, 0.23694097]
my_ur_l = [0.016181267, 0.23169741]
my_gi_l = [0.011182265,0.24850780]
my_vis_l = [0.091133216,0.157014841]

lates = [np.where(CI < 2.6), np.where(ur < 2.2), np.where(gr < 0.7),  
         get_disks(data)]
pars_late = [l_lange_sersic, l_lange_ur, l_lange_gi, l_lange_vis]
mypars_late = [my_ci_l, my_ur_l, my_gi_l, my_vis_l]

labels = ['CI', 'u-r', 'g-r', 'GZ2']

mass=data['MODE']
size=data['PETROR50_R_KPC']

fig = plt.figure(figsize=(10,15))
gs = gridspec.GridSpec(4,2, wspace=0., hspace=0.)
xi, yi = np.mgrid[10.:12:50j, .1:18.5:50j]
pos = np.vstack([xi.ravel(), yi.ravel()])

for i in range(4):
    # KDE OF THE DATA
    values = np.vstack([mass[earlys[i]], size[earlys[i]]])
    k = kde.gaussian_kde(values)
    zi = k(pos).reshape(xi.shape)

    ax = fig.add_subplot(gs[i,0])
    #ax.scatter(mass[earlys[i]], size[earlys[i]], marker='.', 
    #           edgecolor='', color='r', alpha=.5)
    if i == 0:
        ax.plot(np.log10(xxl), fns.power_law(xxl,pars_early[i]),  
                           color='k', ls='-.', label='Shen2003')
    else:
        ax.plot(np.log10(xxl), fns.power_law(xxl,pars_early[i]), 
                color='k', ls='--', label='Lange2015')

    ax.plot(np.log10(xx), fns.power_law(xx, mypars_early[i]), 
                        color='k', ls='-', label='This work')
    ax.contour(xi, yi, zi, levels=[.01, 0.1, .25, .5], colors='r', 
               linewidths=1.5)

    ax.text(10.1, 10., labels[i], size=18)

    ax.set_ylim(.5, 18.)
    ax.set_xlim(10.,12.)
    ax.set_yscale('log')

    ax.set_yticks([1, 10])
    ax.set_yticklabels(['1', '10'])

    if i != 3:
        ax.xaxis.set_visible(False)
    else:
        ax.set_xticks([10.2, 10.7, 11.2, 11.7])
        ax.set_xlabel('Mass')
    ax.set_ylabel('R$_{50}$ [kpc]')

for i in range(4):
    # KDE OF THE DATA
    values = np.vstack([mass[lates[i]], size[lates[i]]])
    k = kde.gaussian_kde(values)
    zi = k(pos).reshape(xi.shape)

    ax = fig.add_subplot(gs[i,1])
    #ax.scatter(mass[lates[i]], size[lates[i]], marker='.', 
    #           edgecolor='', color='b', alpha=.5)
    ax.plot(np.log10(xxl), fns.power_law(xxl,pars_late[i]), color='k', 
            ls='--', label='Lange2015')
    ax.plot(np.log10(xx), fns.power_law(xx, mypars_late[i]), color='k', 
            ls='-', label='This work')
    ax.contour(xi, yi, zi, levels=[.01, 0.1, .25], colors='b', 
               linewidths=1.5)

    ax.set_ylim(.5, 18.)
    ax.set_xlim(10.,12.)
    ax.set_yscale('log')
    ax.yaxis.set_visible(False)

    if i != 3: 
        ax.xaxis.set_visible(False)
    else: 
        ax.set_xticks([10.2, 10.7, 11.2, 11.7])
        ax.set_xlabel('Mass')

    if i == 0:
        artist = plt.Line2D((0,1),(0,0), color='k', ls='-.')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([artist]+handles, ['Shen2003']+labels, loc='lower right', 
                  fontsize=16)

plt.tight_layout()
plt.savefig('size_mass_litcompare.png')
plt.show()

pdb.set_trace()
exit()
# ========= MAXIMIZE LOG-LIKELIHOOD FUNCTION =============

#params = [-0.4828199,10.83686115,0.27742707,0.0032888,0.2752564]
#params.append(prob2d.phi(params, data[early]))
#plot_2dbivar_check(data[early], params)
#pdb.set_trace()

#============== P = (alpha, M_0, beta, R_0, sig) =========================

# initial parameters (schechter, lognormal)
# P = (alpha, M_0, beta, R_0, sig) 
x0 = (-.5, 10.8, .25, .0002, .47)
bounds_e = [(-.5, -.01),(10.5, 11.),(.1, 1.0),(.00001, .01),(.1, 1.)]
bounds_l = [(-1., -0.01),(10., 11.5),(.01, 2.0),(.0001, 1.),(.02, 3.)] 

minimize_plot2d(data[early], x0, bounds_e, name='early_ur_run2a')
minimize_plot2d(data[late], x0, bounds_l, name='late_ur_run2a')

pdb.set_trace()
