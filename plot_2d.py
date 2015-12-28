import math, pyfits
import numpy as np
import scipy.stats
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt
import cosmolopy.distance as cd
from mpl_toolkits.axes_grid1 import make_axes_locatable

import completeness as cmpl
import functions as fns
import limits as lims
import nondet as nd
import probability_2d as probs_2d
from pars_2d import *

plt.rc('font', **{'size':16.,'weight':400})
plt.rc('axes', **{'linewidth': 2.0})
plt.rc('xtick', **{'major.width': 1.2})
plt.rc('ytick', **{'major.width': 1.2})

def comp_ha(x,z):
    return cmpl.get_comp_flux(comp_func_flux_ha, x, z)

def setup_figure():

    fig,ax = plt.subplots(1,1,figsize=(10,11.5),dpi=75,tight_layout=False)
    ax.set_aspect(1.)
    ax.fill([50,lims.get_lim_interval(avg_flim_o3, 0.8)[0],lims.get_lim_interval(avg_flim_o3, 0.8)[0],35,35,50],
             [lims.get_lim_interval(avg_flim_ha, 0.8)[0],lims.get_lim_interval(avg_flim_ha, 0.8)[0],50,50,35,35], 
             'k', lw=0, alpha=0.15)
    ax.fill([50,lims.get_lim_interval(avg_flim_o3, 1.2)[0],lims.get_lim_interval(avg_flim_o3, 1.2)[0],35,35,50],
             [lims.get_lim_interval(avg_flim_ha, 1.2)[0],lims.get_lim_interval(avg_flim_ha, 1.2)[0],50,50,35,35], 
             'k', lw=0, alpha=0.1)
    ax.set_xlim(40.5,43.5)
    ax.set_ylim(40.5,43.5)
    ax.set_xticks([40.5,41,41.5,42,42.5,43])
    ax.set_yticks([41,41.5,42,42.5,43])
    ax.set_xlabel(r'$\mathrm{log\ L}_{\mathrm{[OIII]}}\ [ergs/s]$', fontsize=20)
    ax.set_ylabel(r'$\mathrm{log\ L}_{\mathrm{H \alpha}}\ [ergs/s]$', fontsize=20)

    fig1,ax1 = plt.subplots(1,1,figsize=(8,5),dpi=75,tight_layout=True)
    ax1.set_xlim(40.75,43.5)
    ax1.set_xticks([41,41.5,42,42.5,43])
    ax1.set_ylim(-4.75,-1.1)
    ax1.set_yticks([-4,-3,-2])
    ax1.set_ylabel(r'$\mathrm{log\ } \Psi\ \mathrm{dlog \ L_{[OIII]}}$', fontsize=20)
    ax1.set_xlabel(r'$\mathrm{log\ L}_{\mathrm{[OIII]}}\ [ergs/s]$', fontsize=20)

    fig2,ax2 = plt.subplots(1,1,figsize=(8,5),dpi=75,tight_layout=True)
    ax2.set_xlim(40.75,43.5)
    ax2.set_xticks([41,41.5,42,42.5,43])
    ax2.set_ylim(-4.75,-1.1)
    ax2.set_yticks([-4,-3,-2])
    ax2.set_ylabel(r'$\mathrm{log\ } \Psi\ \mathrm{dlog \ L_{H\alpha}}$', fontsize=20)
    ax2.set_xlabel(r'$\mathrm{log\ L}_{\mathrm{H \alpha}}\ [ergs/s]$', fontsize=20)
    
    return (fig, fig1, fig2), (ax, ax1, ax2)
    
def plot_points(data, axis):

    data_non = data[ data['fo3'] == 0 ]
    data_det = data[ data['fo3'] != 0 ]    

    #for i in range(len(data_non)):
    #     axis.arrow(data_non[i]['dLo3']+np.log10(2.5), data_non[i]['Lha'], -0.05, 0,
    #                head_length=0.015,head_width=0.015,lw=0.25,fc='k',ec='k',alpha=0.4)
    
    axis.errorbar(data_det['Lo3'],data_det['Lha'],
                   xerr=data_det['dLo3'],yerr=data_det['dLha'],
                   color='k',fmt='ko',lw=0.6,markersize=0,capsize=0,alpha=0.75)
    axis.scatter(data_det['Lo3'],data_det['Lha'],color='k',lw=0,s=18,alpha=0.75)
    
def plot_hist(data, P, axis, color, lw, alpha, oiii, correct=False):
    
    bins = np.arange(40.5,43.501,0.3)
    binc = 0.5*(bins[1:]+bins[:-1])
    dbin = bins[1:] - bins[:-1]    
    tot_lim = [35,50]

    if oiii:
        line, flim = 'Lo3', avg_flim_o3
        cut_lim = lims.get_lim_interval(flim, 0.8)
        #comp = cmpl.get_comp(comp_func, (1+data['z'])*data['o3_ew'], data['o3_sn'])
        cor =  np.array([scipy.integrate.nquad(fns.bivar, [cut_lim,[x,y]], args=(P,))[0] for x,y in zip(bins[:-1],bins[1:])])
        cor /= np.array([scipy.integrate.nquad(fns.bivar, [tot_lim,[x,y]], args=(P,))[0] for x,y in zip(bins[:-1],bins[1:])])
    else:
        line, flim = 'Lha', avg_flim_ha
        cut_lim = lims.get_lim_interval(flim, 0.8)
        #comp = cmpl.get_comp(comp_func, (1+data['z'])*data['ha_ew'], data['ha_sn'])
        cor =  np.array([scipy.integrate.nquad(fns.bivar, [[x,y],cut_lim], args=(P,))[0] for x,y in zip(bins[:-1],bins[1:])])
        cor /= np.array([scipy.integrate.nquad(fns.bivar, [[x,y],tot_lim], args=(P,))[0] for x,y in zip(bins[:-1],bins[1:])])
    
    comp = comp_ha(data['Lha'], data['z'])
    hist, _ = np.histogram(data[line], bins=bins, weights=1./comp)
    hist, dbin, binc, cor = hist[hist!=0], dbin[hist!=0], binc[hist!=0], cor[hist!=0]
    err = fns.poisson_interval(hist,0.6827)
    err[0], err[1] = np.log10(hist) - np.log10(err[0]), np.log10(err[1]) - np.log10(hist)
    hist = hist / dbin / fns.veff(0.8, 1.2, s_angle, binc, flim) / cor
    
    if correct:
        ndcor_func = nd.get_ndcor_ha_func(sig=2.5)
        hist = hist * ndcor_func(binc)    
    
    axis.scatter(binc, np.log10(hist), c=color, s=25, lw=0, alpha=alpha)
    axis.errorbar(binc, np.log10(hist), yerr=err, fmt='ko', color=color, markersize=0, lw=lw, alpha=alpha)
    
def plot_2dhist(data, fig, axis):

    values = np.vstack([data['Lo3'],data['Lha']])
    kernel = scipy.stats.gaussian_kde(values, bw_method='scott')
    norm = len(data)/kernel.integrate_box([40.5,40.5],[43.5,43.5])

    xpos, ypos = np.mgrid[40.5:43.5:100j,40.5:43.5:100j]
    pos = np.vstack([xpos.ravel(),ypos.ravel()])
    comp = comp_ha(pos[1,:],1.0)
    kde = np.reshape(np.log10(norm * kernel(pos) / fns.veff(0.8,1.2,s_angle) / comp), xpos.shape)
    img = axis.imshow(kde.T,cmap=plt.cm.hot_r,origin='lower',interpolation='none',
                       vmin=-5.0,vmax=-2.0,extent=[40.5,43.5,40.5,43.5])

    cbaxes = fig.add_axes([0.15, 0.075, 0.7, 0.02])
    cbax = fig.colorbar(mappable=img, cax=cbaxes, ticks=[-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1], orientation='horizontal')
    cbax.ax.set_xlabel(r'$\mathrm{log \ Number \ density \ [\mathrm{Mpc}^{-3}] \ dlogL_{H\alpha} \ dlogL_{[OIII]}}$', fontsize=16)

def plot_bivar_contours(P, phi, axis):
    
    xpos, ypos = np.mgrid[40.5:43.5:250j,40.5:43.5:250j]
    pos = np.vstack([xpos.ravel(),ypos.ravel()])
    val = np.array([10**phi*fns.bivar(y,x,P) for x,y in zip(pos[0],pos[1])])
    val = np.reshape(np.log10(val), xpos.shape)
    axis.contour(xpos, ypos, val, levels=[-2.5,-3.0,-3.5,-4.0,-4.5,-5.0],
                 cmap=plt.cm.hot_r, vmin=-5.0, vmax=-2.0, linewidths=2)

def get_LF(coeff,L):
    LF_o3 = fns.norm_sch(L,*coeff)
    return LF_o3
    
def get_LF_collapse(coeff,L,o3_sig,correct):
    P, phi = coeff
    collapse_fn = fns.collapse_blf(P, phi, correct=correct, sig=o3_sig)
    LF_ha = collapse_fn(L)
    return LF_ha

def plot_LF(coeff, coeff1=None, coeff2=None, 
            axis=None, c=None, lw=None, ls=None, alpha=None, label=None, 
            o3_sig=None, collapse=False, correct=False):

    L = np.arange(40.5,43.5,0.05)

    if not collapse:
        LF = get_LF(coeff,L)
        if coeff1 is not None: LF1 = get_LF(coeff1,L)
        if coeff2 is not None: LF2 = get_LF(coeff2,L)

    else:
        LF = get_LF_collapse(coeff,L,o3_sig,correct)
        if coeff1 is not None: LF1 = get_LF_collapse(coeff1,L,o3_sig,correct)
        if coeff2 is not None: LF2 = get_LF_collapse(coeff2,L,o3_sig,correct)

    axis.plot(L, np.log10(LF), color=c, lw=lw, ls=ls, label=label, alpha=1.0)
    if coeff1 is not None: axis.fill_between(L,np.log10(LF),np.log10(LF1),color=c,alpha=alpha)
    if coeff2 is not None: axis.fill_between(L,np.log10(LF),np.log10(LF2),color=c,alpha=alpha)

def plot_LF_error(best_P, best_phi, chain, o3_sig, ax1, ax2, alpha):

    L = np.arange(40.5,43.5,0.1)
    LF_o3 = get_LF(coeff=(best_P[0],best_P[1],best_phi), L=L)
    LF_ha = get_LF_collapse(coeff=(best_P,best_phi), L=L, o3_sig=o3_sig, correct=False)
    LF_ha_cor = get_LF_collapse(coeff=(best_P,best_phi), L=L, o3_sig=o3_sig, correct=True)
    
    for P in [chain[i,:] for i in 10000+(np.random.rand(500)*(len(chain)-10000))]:
        P, phi = P[:-1], P[-1]
        o3 = get_LF(coeff=(P[0],P[1],phi), L=L)
        ha = get_LF_collapse(coeff=(P,phi), L=L, o3_sig=o3_sig, correct=False)
        ha_cor = get_LF_collapse(coeff=(P,phi), L=L, o3_sig=o3_sig, correct=True)
        LF_o3 = np.vstack((LF_o3,o3))
        LF_ha = np.vstack((LF_ha,ha))
        LF_ha_cor = np.vstack((LF_ha_cor,ha_cor))
    
    ll_o3, ul_o3 = np.percentile(LF_o3, (16,84), axis=0)
    ll_ha, ul_ha = np.percentile(LF_ha, (16,84), axis=0)
    ll_ha_cor, ul_ha_cor = np.percentile(LF_ha_cor, (16,84), axis=0)

    ax1.fill_between(L,np.log10(ll_o3),np.log10(ul_o3),color='k',alpha=alpha)
    ax2.fill_between(L,np.log10(ll_ha),np.log10(ul_ha),color='k',alpha=alpha)
    ax2.fill_between(L,np.log10(ll_ha_cor),np.log10(ul_ha_cor),color='r',alpha=alpha)

def make_plot(wispdata, P, phi, fname=None):
    
    (fig, fig1, fig2), (ax, ax1, ax2) = setup_figure()
    
    chain = np.genfromtxt('mcmc_chain_2d.dat')
    plot_LF_error(P, phi, chain, o3_sig=2.5, ax1=ax1, ax2=ax2, alpha=0.2)
            
    plot_2dhist(wispdata[wispdata['Lo3'] != 0], fig, ax)
    plot_points(wispdata, ax)
    plot_bivar_contours(P, phi, ax)

    #plot_hist(wispdata[wispdata['Lo3'] != 0], P=P, axis=ax1, color='k', lw=1.0, alpha=0.8, oiii=True)
    #plot_hist(wispdata[wispdata['Lo3'] != 0], P=P, axis=ax2, color='k', lw=1.0, alpha=0.8, oiii=False, correct=False)
    #plot_hist(wispdata[wispdata['Lo3'] != 0], P=P, axis=ax2, color='r', lw=1.0, alpha=0.8, oiii=False, correct=True)

    plot_LF(coeff=(P[0],P[1],phi),
            axis=ax1, c='k', lw=3, ls='-', alpha=0.2, label="OIII This Paper (0.8<z<1.2)")
    plot_LF(coeff=(-1.4,42.34,-3.19),
            coeff1=(-1.4+0.15,42.34-0.06,-3.19+0.09),
            coeff2=(-1.4-0.15,42.34+0.06,-3.19-0.09),
            axis=ax1, c='b', lw=1.5, ls='--', alpha=0.1, label="OIII Colbert (2013) (0.7<z<1.5)")

    plot_LF(coeff=(P,phi), 
            axis=ax2, c='k', lw=3, ls='-', alpha=0.2, label=r"H$\alpha$ This Paper (0.8<z<1.2)", 
            o3_sig=2.5, collapse=True, correct=False)
    plot_LF(coeff=(P,phi), 
            axis=ax2, c='r', lw=3, ls='-', alpha=0.2, label=r"H$\alpha$ This Paper (0.8<z<1.2)"+"\n (corrected)", 
            o3_sig=2.5, collapse=True, correct=True)
    plot_LF(coeff=(-1.43,42.18,-2.70),
            coeff1=(-1.43+0.17,42.18-0.10,-2.70+0.12),
            coeff2=(-1.43-0.17,42.18+0.10,-2.70-0.12),
            axis=ax2, c='b', lw=1.5, ls='--', alpha=0.1, label=r"H$\alpha$ Colbert (2013) (0.9<z<1.5)")
    
    ax1.legend(fontsize=14)
    ax2.legend(fontsize=14)

    if fname is not None:
        fig.savefig('plots/fit_bivar.png',bbox_inches='tight')
        fig1.savefig('plots/fit_oiii_'+fname+'.png')
        fig2.savefig('plots/fit_ha_'+fname+'.png')
    else: plt.show()

wispdata = np.genfromtxt('linelists/ha_oiii_z1p0.dat',
                          dtype=[('par',int),('obj',int),('z',float),('zerr',float),('fha',float),('dfha',float),
                                 ('Lha',float),('dLha',float),('ha_fwhm',float),('ha_ew',float),('ha_sn',float),
                                 ('fo3',float),('dfo3',float),('Lo3',float),('dLo3',float),('o3_fwhm',float),
                                 ('o3_ew',float),('o3_sn',float)],unpack=True)
                          
# pivot=40
P = (-1.5283327959, 42.1130223641, 1.1281714447, 0.2811299898, 0.9231545176)
phi = -2.9540416975

make_plot(wispdata,P,phi,fname='2d')
