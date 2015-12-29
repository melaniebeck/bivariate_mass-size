import time
import math
import numpy as np
import scipy.optimize
import functions as fns
from multiprocessing import Queue, Process
import pdb

import probability_2d as probs_2d

def likelihood(P, data, ha_sig, o3_sig, log=None):
    """
    Returns the evaluation of the likelihood function for
    the 2D case. Multiprocesses the data set to calculate
    the numerator and denominator for the probability term.
    """
    #'''
    P = tuple(P)
    now1 = time.time()

    num_procs = 7
    split_data = np.array_split(data, num_procs)

    def slave(queue, chunk_data):
        for entry in chunk_data:
            assert entry['MODE'] != 0.
            # this line calls shit
            p= probs_2d.calc_prob(P, ha_sig, o3_sig, entry) 
            queue.put(p)
        queue.put(None)
        
    queue = Queue()
    procs = [Process(target=slave, args=(queue, chunk_data)) for chunk_data in split_data]

    finished, prob = 0, np.zeros(0)
    for proc in procs: proc.start()
    while finished < num_procs:
        items = queue.get()
        if items is None:
            finished += 1
        else:
            prob = np.append(prob, items)
    for proc in procs: proc.join()

    # prob is an array for all objects (numerators)
    assert len(prob) == len(data) 
    #'''
    """ For when you need to debug. Can't do it in 7 processors simultanesouly!
    P, prob = tuple(P), np.array([])
    now1 = time.time()
    for entry in data:
        p= probs_2d.calc_prob(P, ha_sig, o3_sig, entry) 
        prob = np.append(prob, p)
    #"""
    # divide by single denominator then...
    # redefine calculate denominator (outside above integral)
    def den_int(x,y): return fns.bivar(x,y,P) #* comp_ha(x,entry['z'])
    # send limits to this function -- integrate over all size/mass space
    lim_det_ha = [-0.6, 1.15]
    lim_det_o3 = [10., 12.8]
    quad_args = fns.get_quad_args()
    den = scipy.integrate.nquad(den_int, [lim_det_ha, lim_det_o3], 
                                opts=[quad_args,quad_args])[0] 
	
    prob = prob - math.log(den)
    #print prob, 10**prob
    
    loglike = np.sum(prob)

    now2 = time.time()
 
    log_string = '%15.10f %15.10f %15.10f %15.10f %15.10f : %20.10f : %.2f s' % (P[0], P[1], P[2], P[3], P[4], loglike, (now2-now1))

    if log != None:
        with open(log,'a') as f: f.write(log_string+'\n')
    else: pass  #print log_string
    
    return loglike # returns loglikelihood for ONE SET OF PARAMETERS, P

def minimize(data, ha_sig, o3_sig, x0, bounds, log=None):
    """
    Returns the value for P where the likelihood function
    is maximized (minimizes negative likelihood function)
    """

    if log != None:
        with open(log,'w') as f: f.write('# Log file for maximization of log-likelihood function for 2-D case \n \n')

    #x0 = (a, b, c, d, ...) 5 parameters
    #bounds = [(lowa, higha), (lowb, highb), ...]
    P = scipy.optimize.fmin_l_bfgs_b(lambda x: 1e6 - likelihood(x,data,ha_sig,o3_sig,log), x0=x0, bounds=bounds, approx_grad=True)[0]
    phi = probs_2d.phi(P, data)

    coeff = [P[0],P[1],P[2],P[3],P[4],phi]
    log_string = '%15.10f %15.10f %15.10f %15.10f %15.10f -- Norm.: %15.10f\n' % (P[0], P[1], P[2], P[3], P[4], phi)

    if log != None:
        with open(log,'a') as f: f.write(log_string)
    else: print log_string
        
    return coeff
