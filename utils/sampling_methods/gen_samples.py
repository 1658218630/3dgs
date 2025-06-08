# gen_samples.py

import numpy as np
from fibonacci_BigFloat import sample_gauss

if __name__ == "__main__":
    N = 100                            
    D = 3                              
    mean = np.zeros(D, dtype=np.float64)
    cov  = np.eye(D,    dtype=np.float64)

    samples = sample_gauss(D, N, mean, cov)  

    outfile = "stdnormal3D_samples.txt"
    samples_t = samples.T 

    np.savetxt(outfile, samples_t, fmt="%.8e") 
    print(f'Generated {N} samples and saved at {outfile}')
