## Calculating the bivariate mass-size function 
Code here is thanks to Vihang Mehta and was used in his paper (Mehta et al. 2015). 
* `functions.py` contains various functions necessary for calculations (including schechter function and log-normal)
* `probability_2d.py` calculates the probability that a galaxy with mass M has size R
* `likelihood_2d.py` calculates the log likelihood function for all galaxies in the sample given parameters P; minimizes function to determine best fit parameters
* `bivariate.py` is a wrapper for the above; splits galaxy sample into early and late types using a variety of morphological indicators; has various plotting functions to explore the dataset and compare the fitting to the data

Parameters fit include alpha, M*, beta, R_0, sigma
