# RandomRichness

#Purpose: Create a random sample of masses and richnesses of galaxy clusters at a large scale to use as a training library for a SBI algorithm. Needed a larger sample size for training as Latin Hypercube Quijote simulations only have ~2000 simulations.

#Build Status: complete or near completion

#Important packages to understnad for this code are scipy, emcee, and colossus. Credit goes to Hanzhi Tan for their mass sampling code this is a google doc which explains the fucntions used. https://docs.google.com/document/d/1-GXMtPnOCkAZW28RbWyu4dj3Xu8xjFHVv2d_C-VYr1A/edit?usp=sharing
## There is a small adjustment to Hanzhi's code where the sample number used in MCMC is calculates based on the mass function found from the mass sample of colossus.

#This code uses numpy random to randomly generate the 5 cosmological paramters as follows Ωm : [0.1 − 0.5]; Ωb :  [0.03 − 0.07]; h :   [0.5 − 0.9]; ns :  [0.8 − 1.2]; σ8 :  [0.6 − 1.0]. Redshift and orignal sample number are inputted by user and set to 0 and 100000 by default. These values are passed into colossus where a mass sample is output, then scipy interpolates and integrates a likelihood fucntion and sample number for MCMC. This is the final mass chain and then richness is assigned according to this paper https://arxiv.org/pdf/1707.01907.pdf. The number of masses are then counted based on richness bins and the counts are saved. 

#Major credit to Hanzhi Tan for the mass sampling code which was the basis for most of this and Yuanyuan Zhang and Moonzarin Reza for their advice with the code.
