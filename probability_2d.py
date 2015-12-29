import math
import numpy as np
import scipy.integrate
import pdb

#import completeness as cmpl
import functions as fns
#import limits as lims
#from pars_2d import *

def lim_ha(ha_sig,z,par):
        flim = lims.get_sens('ha',ha_sig,z,par)
        return lims.get_lim_interval(flim, z)

def lim_o3(o3_sig,z,par):
        flim = lims.get_sens('o3',o3_sig,z,par)
        return lims.get_lim_interval(flim, z)

def comp_ha(x,z):
        return cmpl.get_comp_flux(comp_func_flux_ha, x, z)

comp_ha = fns.Memoize(comp_ha)

def calc_prob(P, ha_sig, o3_sig, entry):
        """
        Returns the numerator for the probability calculation of a detection for the 2-D case
	"""
        '''
        lim_entry_ha = lim_ha(ha_sig,entry['z'],entry['par'])
	lolim = max(lim_entry_ha[0], entry['Lha'] - \
                    (entry['dLha'] + math.log10(25.)))
	hilim = min(lim_entry_ha[1], entry['Lha'] + (entry['dLha'] + 
                                                     math.log10(25.)))
	lim_det_ha = [lolim, hilim] 
        #'''
        # size limits (Halpha ~ R_e)
	#lim_det_ha = [-0.6, 1.15]

        '''
	lim_entry_o3 = lim_o3(o3_sig,entry['z'],entry['par'])
	lolim = max(lim_entry_o3[0], entry['Lo3'] - \
                    (entry['dLo3'] + math.log10(25.)))
	hilim = min(lim_entry_o3[1], entry['Lo3'] + (entry['dLo3'] + 
                                                     math.log10(25.)))
	lim_det_o3 = [lolim, hilim] # put in size limits 1.5" ... ?  (put in log)
        '''
        #lim_det_o3 = [10., 12.8]

	#comp = cmpl.get_comp(comp_func, (1+entry['z'])*entry['ha_ew'], entry['ha_sn'])
	
        # numerator marginalized over error bars P=parameters, x/y = Halpha/OIII 
        #def num_int(x,y): return fns.bivar(x,y,P) #* fns.gauss(x,entry['Lha'],entry['dLha']) * fns.gauss(y,entry['Lo3'],entry['dLo3'])
        #quad_args = fns.get_quad_args()
        #print num_int
        #num = scipy.integrate.nquad(num_int, [lim_det_ha, lim_det_o3], opts=[quad_args,quad_args])[0]
        #num = num * comp
        
        """
        # redefine calculate denominator (outside above integral)
        def den_int(x,y): return fns.bivar(x,y,P) #* comp_ha(x,entry['z'])
        # send limits to this function -- integrate over all size/mass space
	den = scipy.integrate.nquad(den_int, [lim_entry_ha, lim_entry_o3], opts=[quad_args,quad_args])[0] # 10^-8 make less accurate
	"""
        #print "size (arcsec):", entry['PETROR50_R']
        #print "size (kpc):", entry['PETROR50_R_KPC']

        num = fns.bivar(entry['PETROR50_R_KPC_LOG'], entry['MODE'], P)
        #print num
        try: prob = math.log(num) #- math.log(den)
        except: 
                print num
                print P
                print "size (kpc):", entry['PETROR50_R_KPC']
                print "mass:", entry['MODE']
                exit()
        return prob
		
def phi(P, data):
        """
	Returns the normalization parameter 'phi' for a given N and parameters
	"""
	#lim_phi_ha = lambda z: lims.get_lim_interval(avg_flim_ha, z)
	#lim_phi_o3 = lambda z: lims.get_lim_interval(avg_flim_o3, z)
	
        lim_phi_ha = [np.min(data['PETROR50_R_KPC_LOG']), 
                      np.max(data['PETROR50_R_KPC_LOG'])]
        lim_phi_o3 = [np.min(data['MODE']), np.max(data['MODE'])]

        z_lim = [0.005, 0.1]
        quad_args = fns.get_quad_args()

	def den_iint(x,y,z): return fns.bivar(x,y,P) * fns.co_vol(z) #* comp_ha(x,z)
	def den_int(z): return scipy.integrate.nquad( den_iint, [lim_phi_ha, lim_phi_o3], args=(z,), opts=[quad_args,quad_args])[0]
	den = scipy.integrate.quad(den_int, z_lim[0], z_lim[1], **quad_args)[0]

	phi = len(data) / den / fns.get_sangle()
	return math.log10(phi)
        
