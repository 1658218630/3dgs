# test_fibonacci_bigfloat.py

import numpy as np
from numpy.linalg import eigvals, norm, inv, det
from fibonacci_BigFloat import sample_gauss

def is_pos_def(x: np.ndarray) -> bool:
    return np.all(eigvals(x) > 0)

def kl_divergence(cov1: np.ndarray, cov2: np.ndarray) -> float:
    """
    KL divergence D_{KL}[N(0, cov1) || N(0, cov2)]
    """
    dim = cov1.shape[0]
    term1 = np.trace(inv(cov2) @ cov1)
    term2 = np.log(det(cov2) / det(cov1))
    return 0.5 * (term1 - dim + term2)

def main():
    # ----------------------------------------
    # Parameters
    # ----------------------------------------
    dim = 3                    # dimensionality
    L   = 100                  # number of samples
    mu  = np.array([1.0, 2.0, 3.0])  # target mean
    C   = np.array([
        [3.535007045438822, 0.5303248580074816, 0.5797800830155548],
        [0.5303248580074816, 3.401792608270529, 0.4253584296303486],
        [0.5797800830155548, 0.4253584296303486, 3.723805074536865]
    ])

    # ----------------------------------------
    # Sanity check
    # ----------------------------------------
    assert is_pos_def(C), "Covariance matrix must be positive definite."

    # ----------------------------------------
    # Draw samples
    # ----------------------------------------
    samples = sample_gauss(dim, L, mu, C)  # shape (dim, L)
    print("type(samples):", type(samples))
    print("samples.shape:", samples.shape)

    # ----------------------------------------
    # Compute sample mean & covariance
    # ----------------------------------------
    sample_mean = samples.mean(axis=1)             # shape (dim,)
    D = samples - sample_mean[:, None]             # demeaned
    sample_cov = D @ D.T / (L - 1)                 # sample covariance

    # ----------------------------------------
    # Print results
    # ----------------------------------------
    print("\nTrue mean:     ", mu)
    print("Sample mean:   ", sample_mean)
    print("Mean error ‖m-μ‖₂:", norm(sample_mean - mu))
    print("Difference (mean):", sample_mean - mu)

    print("\nTrue covariance:\n", C)
    print("Sample covariance:\n", sample_cov)

    kl = kl_divergence(sample_cov, C)
    print("\nKL divergence Dₖₗ(sample_cov ∥ C):", kl)

    print("\nCovariance difference (sample_cov – C):\n", sample_cov - C)

if __name__ == "__main__":
    main()
