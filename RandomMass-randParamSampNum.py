#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 08:59:32 2022

@author: simone
"""

#importing all packages used 
import numpy as np
import matplotlib.pyplot as plt
import math
import emcee
import random
import time

from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy import interpolate
from scipy import integrate
                                      

sample_num = 100000

#extract power function
def extract_power(mass_arr, mfunc):
  """
  Function to extract power of galaxy cluster mass array, and normalize mfunc, power values are passed into the MCMC for sample.

  parameters:
  -----------
  mass_arr: 1d numpy array of cluster mass in 10^n M⊙ unit
  mfunc: 1d numpy array of halo numbr density in dN/dlog(M)(Mpc/ℎ)^(−3) unit

  mass_arr_p: 1d numpy array of mass of power = n
  mfunc_n: 1d numpy array of halo number density * 10^5 #mfunc unit doesnt work in this way, will be changed later
  mass_arr_log: 1d numpy array of natural log of mass
  """
  mass_arr_p = np.log10(mass_arr)
  mfunc_n = mfunc*(10**5)
  mass_arr_log = np.log(mass_arr)
  return mass_arr_p, mfunc_n, mass_arr_log

#the likelihood function
def lnpo(mass, min, max, test_fun):
  """
  likelihood function used by MCMC

  parameters:
  -----------
  mass: a float or a 1d numpy array of cluster mass in 10^n M⊙ unit
  """
  if (mass < min) or (mass > max):
    return -np.inf
  return math.log(test_fun(mass)) #log likelihood is required by emcee

redshift = 0.0
#sample_num = 100000
filename = "rs_" + str(redshift) + "_sn_"  + "_mcmc_save.h5"
print(filename)

def interpolate_MCMC(mass_array_p, mass_arr_log, mfunc, mfunc_n, mass_range, sample_num, redshift):
  """
  interpolate and normalize mfunc_n, use the result as a likelihood function and perform MCMC method to get sample.

  parameters:
  -----------
  mass_arr_p: 1d numpy array of cluster mass power (for example, 10^14 M⊙ represented as 14 in arr)
  mfunc_n: 1d numpy array of halo number density * 10^5
  mass_range: a tuple of cluster masses, lower limit and upper limit for sampling
  sample_num: an integer of number of sample

  sample_chain.flatten(): an 1d numpy array of mass sampling, same unit as mass_arr_p
  """
  min, max = mass_range
  print(min, max)
  
  #new values for integration for sample number
  min2, max2 = np.log(np.power(10, mass_range))
  print (min2, max2)
  
  #interpolate for mass function for MCMC and mass function for sample number
  interpolate_mfunc = interpolate.interp1d(mass_array_p, mfunc_n)
  interpolate_mfunclog = interpolate.interp1d(mass_arr_log, mfunc)
  
  test_arr = np.linspace(min, max, 5000)
  f_test = interpolate_mfunc(test_arr)
  
  #normalize to likelihood function by divided by integration, looking for better method
  val, err = integrate.quad(interpolate_mfunc, min, max)
  print(val, err)
  
  #new values for sample number
  val2, err2 = integrate.quad(interpolate_mfunclog, min2, max2)
  sample_num2 = int(val2*10**9)
  
  test_fun = interpolate.interp1d(test_arr, (f_test/val))
  #plt.plot(test_arr, f_test/val, marker = 'o')
  #nval, nerr = integrate.quad(test_fun, min, max), 5000 test points will lead to error < 10^-4

  #create random walkers
  
  randomlist = []
  for i in range(20):
    n = random.uniform(min, max)
    randomlist.append(n)
  random_arr = np.array(randomlist)

  #backend setup
  ndim, nwalkers = 1, 20
  # name = "rs_" + str(redshift) + "_sn_" + str(sample_num)
  # file_name = "mcmc_save.h5"
  # backend = emcee.backends.HDFBackend(filename, name = name)
  # backend.reset(nwalkers, ndim)
  
  print(test_fun)

  #run MCMC

  processed_sample_num = sample_num2 // (ndim * nwalkers)
  p0 = random_arr.reshape((nwalkers, ndim))
  the_random = random.uniform(1.0, 100.0)
  locals()["sample" + str(the_random)] = emcee.EnsembleSampler(nwalkers, ndim, lnpo, args = [min, max, test_fun])
  t0=time.time()
  pos, prob, state = locals()["sample" + str(the_random)].run_mcmc(p0, 25000) #burn-in sequence
  locals()["sample" + str(the_random)].reset()
  t1=time.time()
  print( ' Done with burn-in: ', t1-t0)
  locals()["sample" + str(the_random)].run_mcmc(pos, processed_sample_num)
  t2=time.time()
  print( ' Done with MCMC: ', t2-t1)

  mass_chain = locals()["sample" + str(the_random)].chain
  locals()["sample" + str(the_random)].reset() #prevent storing error
  return test_fun, mass_chain.flatten()

def mass_sampling(mass_range, sample_num, params_full, redshift, mdef = '200c', model = 'bocquet16'):
  """
  the function to give back a sample of mass distribution based on halo mass function 
     
  Parameters:
  ----------- 
  mass_range: a tuple of cluster masses, lower limit and upper limit for sampling
  redshift: a float, 0.0 by deault
  sample_num: an integer of number of sample, 100000 by default
  mdef: The mass definition in which the halo mass M is given
  model: the halo mass function model used by colossus 

  mass_chain: a numpy array of length = sample_num.
  test_func: the likelihood function
  """
 
  import numpy as np
  min, max = mass_range
  mass_arr = np.logspace(min, max, num = 200, base = 10)
  print (params_full[0], params_full[1], params_full[2], params_full[3], params_full[4])
  print (mass_range)
  print (sample_num)
  params = {'flat': True, 'H0': params_full[2]*100, 'Om0': params_full[0], 'Ob0': params_full[1], 'sigma8': params_full[4], 'ns': params_full[3]}
  cosmology.setCosmology('myCosmo', **params)
  mfunc = mass_function.massFunction(mass_arr, redshift, mdef = mdef, model = model, q_out = 'dndlnM')
  print(mfunc)
  mass_arr_p, mfunc_n, mass_arr_log = extract_power(mass_arr, mfunc)
  test_func, prim_mass_sample = interpolate_MCMC(mass_arr_p, mass_arr_log, mfunc, mfunc_n, mass_range, sample_num, redshift)
  return test_func, prim_mass_sample

def unit_switch(prim_mass_sample):
  """
  switch from n to 10^n

  Parameters:
  ----------- 
  prim_mass_sample: a priliminary numpy array

  mass_sample: a numpy array of cluter mass after unit transformation
  """
  mass_sample = np.power(10, prim_mass_sample)
  return mass_sample

#sets up intial values
params_full = np.zeros((5,))
index = [0, 0.5]
mass_range = [13.0, 17.0]
mass_chain = []
num_variations = 2
richness_bins = [20, 45, 80, 200, 10000000]
richness_bin_len = len(richness_bins) - 1
arrays = np.zeros([len(mass_chain), 2*richness_bin_len * 20])
allcnts = np.zeros([num_variations, richness_bin_len  + 4])
params_total = np.zeros([num_variations, 8])


for v in range(num_variations):
    
    #creates a random set of parameters and gets a mass sample from them
    params_full = [np.random.uniform(low = 0.1, high = 0.5), np.random.uniform(low = 0.03, high = 0.07), np.random.uniform(low = 0.5, high = 0.9), np.random.uniform(low = 0.8, high = 1.2), np.random.uniform(low = 0.6, high = 1.0)]
    test_func, mass_chain = mass_sampling(mass_range, sample_num, params_full, redshift = 0.0)
    mass_chain = np.power(10, mass_chain)
    log_mass = np.log(mass_chain)
    
    
    #### see tis paper https://arxiv.org/pdf/1707.01907.pdf
    ### randomly generate a set of MA, MB and lnsigma according to a rough range from the above paper
    ### these will be new additional input parameters
    MA = 3.2 + 2.0*np.random.rand(1)
    MB = 0.99 + 0.5*np.random.rand(1)
    lnsigma =  0.456 + 0.3 * np.random.rand(1)
   
    
    for mm in range(4):
        params_total[v, mm] = params_full[mm]
    
    params_total[v, 5] = MA
    params_total[v, 6] = MB
    params_total[v, 7] = lnsigma
    print ("This is the array", params_total)
    
    
    mu = MA + MB*(log_mass - np.log(3.0*10**14.0))
    sigma = lnsigma
    lnL = np.random.normal(mu, sigma, len(mu))
    lamda = np.exp(lnL)
    print(np.exp(MA + MB*(np.log(2.0*10.0**13.0) - np.log(3.0*10**14.0))), np.exp(MA), np.min(lamda), np.max(lamda))

    #creates a scatter plot of the mass vs richness     
    plt.scatter(log_mass, np.log10(lamda), s = 5, alpha = 0.2)
    plt.xlabel('Mass (ln(Msun/h))')
    plt.ylabel('Richness (lnlamda)')
    plt.show()
    
    #count halos according to richness (lamda) bins 
    for kk in range(richness_bin_len):
        mass_lo=richness_bins[kk]
        mass_up=richness_bins[kk+1]

        ind, = np.where( (lamda >= mass_lo) & (lamda < mass_up))
        cnt=len(ind)
        mass_aver = np.log10(np.mean(mass_chain[ind]) )
        print (np.shape(allcnts))
        allcnts[v, kk] = cnt
        allcnts[v, kk+4] = mass_aver
        print ("This is the average mass", mass_aver)
        
        print ("this is the counts array", allcnts)
        
np.savetxt('RichnessCounts', allcnts)
np.savetxt('Params', params_total)