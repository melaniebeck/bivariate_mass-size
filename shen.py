from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import cosmolopy.distance as cd
import cosmolopy.magnitudes as cm
import pdb
from scipy.stats import kde
import calc_kcor as KC

'''
As an excersize I'm following along with Shen's paper and plotting what he
plots so that I have a generic understanding of the task ahead of me. 
'''

plt.rc('font', **{'size':18.,'weight':400})
plt.rc('axes', **{'linewidth': 2.0})
plt.rc('xtick', **{'major.width': 1.2})
plt.rc('ytick', **{'major.width': 1.2})

def density(x, y, limits, numbins):

    notnans = np.where((~np.isnan(x) & ~np.isnan(y)))
    x, y = x[notnans], y[notnans]

    xsize = np.abs(limits[1]-limits[0])
    ysize = np.abs(limits[3]-limits[2])

    if xsize > ysize:
        aspect = ysize/xsize
    else:
        aspect = xsize/ysize

    gridx = np.linspace(limits[0], limits[1], numbins)
    gridy = np.linspace(limits[2], limits[3], numbins)

    grid, xedges, yedges = np.histogram2d(x, y, bins=[gridx, gridy])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    #plt.figure()
    #plt.imshow(grid.T, origin='low', extent=extent, interpolation='nearest',
    #           aspect=aspect)
    #plt.show()
    return grid, extent, aspect

def gen_bins(data, binsize):
    return np.arange(np.min(data), np.max(data)+binsize, binsize)

def redshift_limits(magnitude, redshift, mag_min=11., mag_max=17.2):
    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'h':0.7}

    dL = cd.luminosity_distance(redshift, **cosmo)
    dL_min = dL*10**(-0.2*(magnitude-mag_min))
    dL_max = dL*10**(-0.2*(magnitude-mag_max))

    return dL_min, dL_max

def calculate_mass_limits(mass, magnitude, mag_limit=17.2):
    log_mass_limit = mass + 0.4*(magnitude-mag_limit)
    return log_mass_limit

def restframe_mag(apparant_mag, color, redshift, filter_name, color_name):
    cosmo = {'h': 0.72, 'omega_M_0': 0.3, 'omega_lambda_0': 0.7, 
             'omega_k_0': 0.0}

    return apparant_mag - cm.distance_modulus(redshift,**cosmo) - \
        KC.calc_kcor(filter_name, redshift, color_name, color)

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

def complicated_mass_cut():
    mass_limits = calculate_mass_limits(data['MODE'], data['PETROMAG_R'])
    data['M_lim'] = mass_limits
    
    # sort by magnitude
    data.sort('PETROMAG_R')
    
    # select faintest 20%
    faintest_twenty_percent = len(data) - int(.2*len(data))
    faintest = data[faintest_twenty_percent:]
    
    colors = ['0.5' for i in range(faintest_twenty_percent)]
    colors = colors + ['b' for i in range(len(data) - faintest_twenty_percent)]
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(121)
    ax1.scatter(data['REDSHIFT'], data['MODE'], c='0.5', alpha=.3, marker='.',
                edgecolor='')
    ax1.set_xlim(-0.01,.25)
    ax1.set_ylim(5.5,12.5)
    ax1.set_xlabel('z')
    ax1.set_ylabel('log Stellar Mass')
    ax2 = fig.add_subplot(122)
    ax2.scatter(data['REDSHIFT'], data['MODE'], c='0.5', alpha=.3, marker='.',
                edgecolor='')
    ax2.scatter(data['REDSHIFT'], data['M_lim'], c=colors, alpha=.3, marker='.',
                edgecolor='')
    ax2.set_xlim(-0.01,.25)
    ax2.set_ylim(5.5,12.5)
    ax2.set_xlabel('z')
    plt.show()
    

def apply_cuts(data):
    print "Starting with: ", len(data)
    
    data = data[data['MODE']>0]
    print "After removing those which do not have stellar masses: ", len(data)
    
    #basic_properties(data, filename='full')
    #mass_mag_redshift(data, filename='full')
    
    # cut in redshift
    data1 = data[np.where((data['REDSHIFT']<=0.1) & (data['REDSHIFT']>0.02))]
    print "0.02 < z <= 0.1: ", len(data1)
    
    # cut in magnitude (r band)
    data2 = data1[np.where((data1['PETROMAG_R']<=17.2) & (data1['PETROMAG_R']>=15))]
    print "15. <= m_r <= 17.2: ", len(data2)
    
    # cut in size
    data3 = data2[np.where(data2['PETROR50_R']>=1.5)]
    print "R_50 >= 1.5: ", len(data3)
    
    # cut in surface brightness
    data4 = data3[np.where(data3['MU50_R']<=23.0)]
    print "u_50 <= 23: ", len(data4)
    
    # cut in mass
    data5 = data4[np.where(data4['MODE']>10.)]
    print "log M_sun > 10: ", len(data5)
    
    #basic_properties(data5, filename='cuts')
    #mass_mag_redshift(data5, filename='cuts')
    #plot_size_mass(data5, filename='cuts2')
    #plot_URZcolors(data, data5)
    
    #Table.write(data5, 'zoo2MainSpecz_Ancillary_Masses_cut2.fits')
    #pdb.set_trace()
    return data5
    
def basic_properties(data, filename=None):
    # ---------------- HISTOGRAMS of BASIC PROPERTIES ----------------------
    fig = plt.figure(figsize=(10,10))

    ax1 = fig.add_subplot(221)
    ax1.hist(data['PETROR50_R'], normed=True, histtype='step', lw=2, color='k',
             bins=gen_bins(data['PETROR50_R'],.5))
    ax1.axvline(x=1.5, color='r', ls='--', lw=2)
    ax1.set_xlabel(r'$R_{50,r}$')
    ax1.set_xlim(0,12)
    
    ax2 = fig.add_subplot(222)
    ax2.hist(data['PETROMAG_R'], normed=True, histtype='step', lw=2, color='k',
             bins=gen_bins(data['PETROMAG_R'], .1))
    ax2.axvline(x=17.2, color='r', ls='--', lw=2)
    ax2.axvline(x=15., color='r', ls='--', lw=2)
    ax2.set_xlabel('r [mag]')
    ax2.set_xlim(13,19)
    
    ax3 = fig.add_subplot(223)
    ax3.hist(data['MU50_R'], normed=True, bins=gen_bins(data['MU50_R'], .1),
             histtype='step', lw=2, color='k')
    ax3.axvline(x=23.0, color='r', ls='--', lw=2)
    ax3.set_xlabel(r'$\mu_{50,r}$')
    
    ax4 = fig.add_subplot(224)
    ax4.hist(data['REDSHIFT'], normed=True, histtype='step', lw=2, color='k',
             bins=gen_bins(data['REDSHIFT'], .005))
    ax4.axvline(x=0.1, color='r', ls='--', lw=2)
    ax4.axvline(x=0.02, color='r', ls='--', lw=2)
    ax4.set_xlabel('z')
    
    plt.tight_layout()
    plt.savefig('basic_properties_%s.png'%filename)
    plt.close()

def mass_mag_redshift(data, filename=None):
    rz_color = data['PETROMAG_G']-data['PETROMAG_Z']
    gi_color = data['PETROMAG_G']-data['PETROMAG_I']

    colors = data['MODE']/13.
    #plt.hist(colors)
    #plt.show()
    #plt.close()


    # ----------------- MASS & MAGNITUDE VS REDSHIFT -------------------------
    fig = plt.figure(figsize=(10,10))
    
    ax1 = fig.add_subplot(211)
    plt1 = ax1.scatter(data['REDSHIFT'], data['PETROMAG_R'], 
                       marker='.', c=colors, edgecolor='')
    #ax1.axhline(y=17., color='k', ls='--')
    ax1.axhline(y=17.2, color='k', ls='-.')
    ax1.axvline(x=0.1, color='k', ls='--')
    ax1.axvline(x=0.02, color='k', ls='--')
    ax1.set_xlim(-0.01, .26)
    ax1.set_ylim(10, 18)
    ax1.set_xlabel('z')
    ax1.set_ylabel('m$_r$')
    
    ticks = np.arange(6., 13.)/13.
    #print ticks
    labels = [format(t*13., '.1f') for t in ticks]
    cbax = fig.colorbar(plt1, ticks=ticks)
    cbax.ax.set_ylabel('log($M_{sun}$)')
    cbax.ax.set_yticklabels(labels)
    
    
    ax2 = fig.add_subplot(212)
    okay = np.where( (gi_color > 0.) & (gi_color < 2.) )
    
    colors = gi_color[okay]/2.
    
    plt2 = ax2.scatter(data['MODE'][okay], data['REDSHIFT'][okay], marker='.', 
                       c=colors, edgecolor='')
    ax2.axhline(y=0.1, color='k', ls='--')
    ax2.axhline(y=0.02, color='k', ls='--')
    ax2.axvline(x=10., color='k', ls='--')
    ax2.set_xlim(6, 12.5)
    ax2.set_ylim(-0.01, .26)
    ax2.set_xlabel('Stellar Mass, log $M_*$')
    ax2.set_ylabel('z')
    
    ticks = np.arange(0., 2.,.5)/2.
    labels = [format(t*2., '.1f') for t in ticks]
    cbax = fig.colorbar(plt2, ticks=ticks)
    cbax.ax.set_ylabel('g - i')
    cbax.ax.set_yticklabels(labels)
    
    plt.tight_layout()
    plt.savefig('mass_mag_z_%s.png'%filename)
    plt.show()
    plt.close()

def plot_size_mass(data, filename):

    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    ax1.scatter(np.log10(data['PETROR50_R_KPC']), data['MODE'], marker='.', edgecolor='')
    #ax1.set_yscale('log')
    ax1.set_ylim(9.7, 12.)
    ax1.set_xlim(-0.2, 1.5)
    ax1.set_ylabel('log Stellar Mass', fontsize=16)
    ax1.set_xlabel('R$_{50,r}$ [kpc]', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('size_mass_%s.png'%filename)
    plt.show()
    plt.close()
    

def plot_UGRcolors(data, data5):
    ur_color = data['PETROMAG_U']-data['PETROMAG_R']
    gi_color = data['PETROMAG_G']-data['PETROMAG_I']
    rz_color = data['PETROMAG_R']-data['PETROMAG_Z']
    
    Mu = restframe_mag(data['PETROMAG_U'], ur_color, data['REDSHIFT'], 
                       'u', 'u - r')
    Mr = restframe_mag(data['PETROMAG_R'], ur_color, data['REDSHIFT'], 
                       'r', 'u - r')
    Mz = restframe_mag(data['PETROMAG_Z'], rz_color, data['REDSHIFT'], 
                       'z', 'r - z')
    
    ur_color_sub = data5['PETROMAG_U']-data5['PETROMAG_R']
    gi_color_sub = data5['PETROMAG_G']-data5['PETROMAG_I']
    rz_color_sub = data5['PETROMAG_R']-data5['PETROMAG_Z']
    
    Mu_sub = restframe_mag(data5['PETROMAG_U'], ur_color_sub, 
                           data5['REDSHIFT'],'u', 'u - r')
    Mr_sub = restframe_mag(data5['PETROMAG_R'], ur_color_sub, 
                           data5['REDSHIFT'],  'r', 'u - r')
    Mz_sub = restframe_mag(data5['PETROMAG_Z'], rz_color_sub, 
                           data5['REDSHIFT'], 'z', 'r - z') 
    
    #---------------------------------------------------------------------
    
    # restframe absolute magnitude / colors
    M_ur, M_ur_sub = Mu - Mr, Mu_sub - Mr_sub
    
    # apparent magnitudes / colors
    M_rz, M_rz_sub = Mr - Mz, Mr_sub - Mz_sub
    
    
    Mur_lims = [-1., 4.]
    Mrz_lims = [-0.5, 2.]
    
    good = np.where( (M_rz < 250.) & (M_ur < 75.))
    
    xi, yi = np.mgrid[Mrz_lims[0]:Mrz_lims[1]:40j,Mur_lims[0]:Mur_lims[1]:30j]
    
    fig = plt.figure(figsize=(15,15))
    
    x = np.linspace(0,2)
    y = 0.88*x+0.69
    
    # Full Sample - Restframe Magnitudes -------------------------------
    ax1 = fig.add_subplot(221)
    
    k1 = kde.gaussian_kde(np.vstack([M_rz[good], M_ur[good]]))
    z1 = k1(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    img = ax1.pcolormesh(xi, yi, z1, cmap=plt.cm.Blues)
    ax1.contour(xi.T, yi.T, z1.T, 10, cmap = plt.cm.bone, lw=2)
    #ax2.scatter(M_rz, M_ur)
    ax1.plot(x,y, color='k', lw = 2)
    ax1.plot([-0.5,1.],[1.3, 1.3], color='k', lw=2)
    plt.axis([Mrz_lims[0], Mrz_lims[1], Mur_lims[0], Mur_lims[1]])
    ax1.set_xlabel('M(r) - M(z)')
    ax1.set_ylabel('M(u) - M(r)')
    ax1.set_title("Full Sample (240K) RestFrame")
    
    
    # Full Sample - Apparent Magnitudes --------------------------------
    ax2 = fig.add_subplot(223)
    
    k2 = kde.gaussian_kde(np.vstack([rz_color, ur_color]))
    z2 = k2(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    img = ax2.pcolormesh(xi, yi, z2, cmap=plt.cm.Blues)
    ax2.contour(xi.T, yi.T, z2.T, 10, cmap = plt.cm.bone, lw=2)
    #ax1.scatter(rz_color, ur_color)
    ax2.plot(x,y, color='k', lw = 2)
    ax2.plot([-0.5,1.],[1.3, 1.3], color='k', lw=2)
    plt.axis([Mrz_lims[0], Mrz_lims[1], Mur_lims[0], Mur_lims[1]])
    ax2.set_xlabel('r - z')
    ax2.set_ylabel('u - r')
    ax2.set_title('Full Sample Observed')
    
    # Working Sample - Restframe Magnitudes -------------------------------
    ax3 = fig.add_subplot(222)
    
    k3 = kde.gaussian_kde(np.vstack([M_rz_sub, M_ur_sub]))
    z3 = k3(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    img = ax3.pcolormesh(xi, yi, z3, cmap=plt.cm.Blues)
    ax3.contour(xi.T, yi.T, z3.T, 10, cmap = plt.cm.bone, lw=2)
    #ax2.scatter(M_rz, M_ur)
    ax3.plot(x,y, color='k', lw = 2)
    ax3.plot([-0.5,1.],[1.3, 1.3], color='k', lw=2)
    plt.axis([Mrz_lims[0], Mrz_lims[1], Mur_lims[0], Mur_lims[1]])
    ax3.set_xlabel('M(r) - M(z)')
    ax3.set_ylabel('M(u) - M(r)')
    ax3.set_title("Working Sample (120K) RestFrame")
    
    
    # Working Sample - Apparent Magnitudes --------------------------------
    ax4 = fig.add_subplot(224)
    
    k4 = kde.gaussian_kde(np.vstack([rz_color_sub, ur_color_sub]))
    z4 = k4(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    img = ax4.pcolormesh(xi, yi, z4, cmap=plt.cm.Blues)
    ax4.contour(xi.T, yi.T, z4.T, 10, cmap = plt.cm.bone, lw=2)
    #ax1.scatter(rz_color, ur_color)
    ax4.plot(x,y, color='k', lw = 2)
    ax4.plot([-0.5,1.],[1.3, 1.3], color='k', lw=2)
    plt.axis([Mrz_lims[0], Mrz_lims[1], Mur_lims[0], Mur_lims[1]])
    ax4.set_xlabel('r - z')
    ax4.set_ylabel('u - r')
    ax4.set_title('Working Sample (120K) Observed')
    
    plt.tight_layout()
    plt.savefig('URZcolors.png')
    plt.show()

def plot_GAMAfig2(data):
    mass = data['MODE']
    ur = data['PETROMAG_U']-data['PETROMAG_R']
    gi = data['PETROMAG_G']-data['PETROMAG_R']
    conc = data['PETROR90_R']/data['PETROR50_R']


    fig = plt.figure(figsize=(8, 15))
     
    # ----------- Stellar Mass vs Concentration; Color = ur ---------------
    ax1 = fig.add_subplot(311)
    ax1.set_xlim(1.,5.)
    ax1.set_ylim(10.,12.)
    ax1.set_xlabel('CI')
    ax1.set_ylabel('log[Stellar Mass]')
    xi, yi = np.mgrid[np.min(conc):np.max(conc):30j, 
                      np.min(mass):np.max(mass):40j]
    k1 = kde.gaussian_kde(np.vstack([conc, mass]))
    z1 = k1(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    color = ur/np.max(ur)
    good = np.where( (color < .215) & (color > 0.02) )
    scatter = ax1.scatter(conc[good], mass[good], c=color[good], marker='.', 
                          edgecolor='')
    #img = ax4.pcolormesh(xi, yi, z4, cmap=plt.cm.Blues)
    ax1.contour(xi.T, yi.T, z1.T, 10, cmap = plt.cm.bone, lw=2)

    ticks = np.arange(0.1, 4.1, .5)/np.max(ur)
    labels = [format(t*np.max(ur), '.1f') for t in ticks]
    cbax = fig.colorbar(scatter, ticks=ticks)
    cbax.ax.set_ylabel('u - r')
    cbax.ax.set_yticklabels(labels)

    #"""
    # ---------------- ur vs Stellar Mass; color = conc -----------------
    ax2 = fig.add_subplot(312)
    ax2.set_xlim(10.,12.)
    ax2.set_ylim(0.,5.)
    ax2.set_xlabel('log[Stellar Mass]')
    ax2.set_ylabel('u - r')
    xi, yi = np.mgrid[np.min(mass):np.max(mass):30j, 
                      np.min(ur):np.max(ur):40j]
    k2 = kde.gaussian_kde(np.vstack([mass, ur]))
    z2 = k2(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    color = conc/np.max(conc)
    good = np.where(color < .35)
    scatter = ax2.scatter(mass[good], ur[good], c=color[good], marker='.', 
                          edgecolor='')
    ax2.contour(xi.T, yi.T, z2.T, 10, cmap = plt.cm.bone, lw=2)
    ax2.axhline(y=2.2, color='k')

    ticks = np.arange(1., 4.,.5)/np.max(conc)
    labels = [format(t*np.max(conc), '.1f') for t in ticks]
    cbax = fig.colorbar(scatter, ticks=ticks)
    cbax.ax.set_ylabel('CI')
    cbax.ax.set_yticklabels(labels) 

    #----------------- gi vs Stellar Mass; color = conc ------------------
    ax3 = fig.add_subplot(313)
    ax3.set_xlim(10.,12.)
    ax3.set_ylim(0.,2.)
    ax3.set_xlabel('log[Stellar Mass]')
    ax3.set_ylabel('g - i')
    xi, yi = np.mgrid[np.min(mass):np.max(mass):30j, 
                      np.min(gi):np.max(gi):40j]
    k3 = kde.gaussian_kde(np.vstack([mass, gi]))
    z3 = k3(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    color = conc/np.max(conc)
    good = np.where(color < .35)
    scatter = ax3.scatter(mass[good], gi[good], c=color[good], marker='.', 
                          edgecolor='')
    ax3.contour(xi.T, yi.T, z3.T, 10, cmap = plt.cm.bone, lw=2)
    ax3.axhline(y=.7, color='k')

    ticks = np.arange(1., 4.,.5)/np.max(conc)
    labels = [format(t*np.max(conc), '.1f') for t in ticks]
    cbax = fig.colorbar(scatter, ticks=ticks)
    cbax.ax.set_ylabel('CI')
    cbax.ax.set_yticklabels(labels) 
    #"""

    plt.tight_layout()
    plt.savefig('3D_morph.png')
    plt.show()
    plt.close()

def plot_GAMAfig4(data):
    ur = data['PETROMAG_U']-data['PETROMAG_R']
    gi = data['PETROMAG_G']-data['PETROMAG_R']
    conc = data['PETROR90_R']/data['PETROR50_R']

    fig = plt.figure(figsize=(10,15))

    ax1 = fig.add_subplot(311)
    bins = np.arange(np.min(conc), np.max(conc)+0.05, 0.05)
    ax1.hist(conc, normed=True, bins=bins, range=(0.,5.), histtype='step', lw=2)
    ax1.axvline(x=2.65, color='k', lw=2)
    ax1.set_xlim(1.5, 3.75)
    ax1.set_xlabel('CI')

    ax2 = fig.add_subplot(312)
    bins = np.arange(np.min(ur), np.max(ur)+0.05, 0.05)
    ax2.hist(ur, normed=True, bins=bins, range=(0., 5.), histtype='step', lw=2)
    ax2.axvline(x=2.2, color='k', lw=2)
    ax2.set_xlim(1.2,3.5)
    ax2.set_xlabel('u - r')

    ax3 = fig.add_subplot(313)
    bins = np.arange(np.min(gi), np.max(gi)+0.02, 0.02)
    ax3.hist(gi, normed=True, bins=bins, range=(0., 2.), histtype='step', lw=2)
    ax3.axvline(x=0.7, color='k', lw=2)
    ax3.set_xlim(0.3,1.1)
    ax3.set_xlabel('g - r')

    plt.tight_layout()
    plt.savefig('morph_cuts.png')
    plt.show()
    plt.close()


def plot_stuff_GZclasses(data):
    feat = get_disks(data)
    ells = get_ellipticals(data)

    combo = np.sort(np.concatenate([feat[0],ells[0]]))
    color = np.array(['b' if x in feat[0] else 'r' for x in combo])
    data = data[combo]

    import numpy.random as rand
    random_sample = rand.random_integers(0,len(data)-1, .5*len(data))
    
    ur = data['PETROMAG_U']-data['PETROMAG_R']
    gi = data['PETROMAG_G']-data['PETROMAG_R']
    conc = data['PETROR90_R']/data['PETROR50_R']
    mass = data['MODE']


    fig = plt.figure(figsize=(8,15))
    
    ax1 = fig.add_subplot(311)
    ax1.set_xlim(1.,5.)
    ax1.set_ylim(10.,12.5)
    ax1.set_xlabel('CI')
    ax1.set_ylabel('log[Stellar Mass]')
    xi, yi = np.mgrid[np.min(conc):np.max(conc):80j,
                      np.min(mass):np.max(mass):80j]
    k1 = kde.gaussian_kde(np.vstack([conc, mass]), bw_method='scott')
    z1 = np.reshape(k1(np.vstack([xi.flatten(), yi.flatten()])), xi.shape) 

    img = ax1.pcolormesh(xi, yi, z1, cmap=plt.cm.bone_r)
    plt.axes([1.,5., 10.,12.])
    levels = [.1, .32, .5, .75, .90]
    ax1.contour(xi.T, yi.T, z1.T, levels=levels, colors=('k',), 
                linewidths=(1.3,))
    scatter = ax1.scatter(conc[random_sample], mass[random_sample], 
                          c=color[random_sample], marker='.',
                          edgecolor='', alpha=0.5)
    ax1.axvline(x=2.65, color='k', lw=2, ls='--')

    #red = mpatches.Patch(color='red', label='Smooth')
    #blue = mpatches.Patch(color='blue', label='Featured')
    #plt.legend(handles=[red, blue])

    #"""
    # ---------------- ur vs Stellar Mass; color = conc -----------------
    ax2 = fig.add_subplot(312)
    ax2.set_xlim(10., 12.5)
    ax2.set_ylim(0.,4.5)
    ax2.set_xlabel('log[Stellar Mass]')
    ax2.set_ylabel('u - r')
    xi, yi = np.mgrid[10.:12.5:60j,0.:5.:60j]
    k2 = kde.gaussian_kde(np.vstack([mass, ur]))
    z2 = k2(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    img = ax2.pcolormesh(xi, yi, z2, cmap=plt.cm.bone_r)
    ax2.contour(xi.T, yi.T, z2.T, levels=levels, colors=('k',), 
                linewidths=(1.3,))
    scatter = ax2.scatter(mass[random_sample], ur[random_sample], 
                          c=color[random_sample], marker='.', edgecolor='')
    levels = [.1, .32, .5, .75, .90]
    ax2.axhline(y=2.2, color='k', lw=2, ls='--')


    #----------------- gi vs Stellar Mass; color = conc ------------------
    ax3 = fig.add_subplot(313)
    ax3.set_xlim(10.,12.5)
    ax3.set_ylim(0.,1.5)
    ax3.set_xlabel('log[Stellar Mass]')
    ax3.set_ylabel('g - i')
    xi, yi = np.mgrid[10.:12.5:60j, 0.:1.5:60j]
    k3 = kde.gaussian_kde(np.vstack([mass, gi]))
    z3 = k3(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    img = ax3.pcolormesh(xi, yi, z3, cmap=plt.cm.bone_r)
    ax3.contour(xi.T, yi.T, z3.T, levels=levels, colors=('k',), 
                linewidths=(1.3,))
    scatter = ax3.scatter(mass[random_sample], gi[random_sample], 
                          c=color[random_sample], marker='.', edgecolor='')
    ax3.axhline(y=.7, color='k', lw=2, ls='--')
    
    plt.tight_layout()
    plt.savefig('visual_morph.png')
    plt.show()
    pdb.set_trace()
    #"""  

# ======================================================================= #

# spectroscopically confirmed redshifts -- All Data for GZ2
#data = Table.read('zoo2MainSpecz_Ancillary_Masses.fits')
data = Table.read('zoo2MainSpecz_Ancillary_Masses_cut2.fits')
#data_cuts = apply_cuts(data)

# Data from above after cuts made and matched to my morphologies
#data_morph = Table.read('zoo2MainSpecz_Ancillary_Masses_cut2_morph.fits')

plot_stuff_GZclasses(data)
pdb.set_trace()

plot_GAMAfig2(data)
plot_GAMAfig4(data)

pdb.set_trace()

full_c = data['PETROR90_R']/data['PETROR50_R']
work_c = data_cuts['PETROR90_R']/data_cuts['PETROR50_R']
my_c_elipt = data_morph['r80']/data_morph['r20']
my_c_circ = data_morph['r80_c']/data_morph['r20_c']

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.hist(full_c, bins=100, color='y', range=(1., 7.), alpha=.6,
         label='Full Sample / SDSS CI (R90/R50)')

#ax2 = fig.add_subplot(312)
ax1.hist(work_c, bins=100, color='r', range=(1., 7.), alpha=.6,
         label='Working Sample / SDSS CI (R90/R50)')

#ax3 = fig.add_subplot(313)
ax1.hist(my_c_elipt, bins=100, color='b', range=(1., 7.), alpha=.65,
         label='Working Sample / My CI (R80/R20 - elipt)') 
ax1.hist(my_c_circ, bins=100, color='g', range=(1., 7.), alpha=.65,
         label='Working Sample / My CI (R80/R20 - circ)')

ax1.axvline(x=2.86, color='k', ls='--', lw=3)
ax1.text(2.9, 12000, 'CI = 2.86')
ax1.axvline(x=2.6, color='k', ls='--', lw=3)
ax1.text(2., 12000, 'CI = 2.6')
ax1.legend(loc='best')
ax1.set_xlabel('Concentration Index')
ax1.set_ylabel('Number of Galaxies')

plt.tight_layout()
plt.savefig('concentration.png')
plt.show()
pdb.set_trace()
