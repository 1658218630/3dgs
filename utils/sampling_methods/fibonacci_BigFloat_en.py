import numpy as np
import warnings
from scipy.special import erfinv
from sympy import isprime

__all__ = ['sample_gauss']

def fibonacci_eigen(D: int) -> np.ndarray:
    """
    Compute the D×D orthogonal matrix V based on generalized Fibonacci grids.
    :param D: Matrix dimension
    :return: A D×D orthogonal matrix V
    """
    # Special construction when dimension D is 4
    if D == 4:
        # Golden ratio and related constants
        p  = (1 + np.sqrt(5.0)) / 2       # Golden ratio φ
        ap = 3 + np.sqrt(5.0)             # Constant for computation
        am = 3 - np.sqrt(5.0)
        bp = np.sqrt(6 * (5 + np.sqrt(5.0)))
        bm = np.sqrt(6 * (5 - np.sqrt(5.0)))

        # Compute intermediate variables v1, v2, v3, v4
        v1 = (am - bm) / 4
        v2 = (ap - bp) / 4
        v3 = -1 / v1
        v4 = -1 / v2

        # Normalization factors g, h
        g = 1 / np.sqrt((1 + v3**2) * (1 + p**2))
        h = 1 / np.sqrt((1 + v4**2) * (1 + p**2))

        # Construct orthogonal matrix V
        V = np.array([
            [ p * g,              h,            p * v3 * g,   v4 * h],
            [ g,                 -p * h,        v3 * g,      -p * v4 * h],
            [-p * v3 * g,       -v4 * h,        p * g,         h],
            [-v3 * g,            p * v4 * h,    g,           -p * h]
        ], dtype=float)
    else:
        # General case: construct D-dimensional discrete cosine matrix
        i1 = np.arange(1, D+1, dtype=float)
        j1 = i1.copy()
        theta = np.outer(2*i1 - 1, 2*j1 - 1) * np.pi / (4*D + 2)
        V = np.cos(theta)
        # Normalize columns to unit norm
        col_norms = np.linalg.norm(V, axis=0)
        V = V / col_norms

        # Warn if (2*D + 1) is not prime, as primes yield optimal performance
        if not isprime(2*D + 1):
            warnings.warn("2 * D + 1 should be prime for optimal performance")

    return V

def sample_gauss(
    Dimension: int,
    NSamples: int,
    Mean: np.ndarray,
    Covariance: np.ndarray,
    Rescale: bool = True
) -> np.ndarray:
    """
    Perform deterministic Gaussian sampling using generalized Fibonacci grids.
    Returns a D×L array of samples.
    :param Dimension: Dimension D of the random variable
    :param NSamples: Number of samples L to generate
    :param Mean: Mean vector of length D
    :param Covariance: Covariance matrix of size D×D
    :param Rescale: Whether to apply boundary rescaling
    :return: A D×L array of generated samples
    """
    # Initialize dimension and sample count
    D = Dimension
    L = NSamples
    # Reshape mean into a column vector
    mu = Mean.reshape(D, 1)
    # Copy covariance matrix and convert to float
    C  = Covariance.copy().astype(float)

    # 1. Compute Fibonacci orthogonal basis
    V = fibonacci_eigen(D)
    # outer is used to scale grid extent
    outer = np.max(np.sum(np.abs(V), axis=1))

    # 2. Build uniform grid of points
    L0 = int(np.ceil(L ** (1.0 / D)))  # Points per dimension
    spc = 1.0 / L0                     # Grid spacing
    extra = 2                          # Extra boundary buffer
    L1 = int(np.ceil(outer / spc)) + extra
    # Ensure parity of L and L1 matches for symmetry
    if (L % 2) != (L1 % 2):
        L1 += 1

    # 2.1 Construct 1D grid and center it
    vec = np.arange(0, L1, dtype=float) * spc
    vec -= vec.mean()

    # 2.2 Build full D-dimensional Cartesian grid and flatten to (D, L1^D)
    mesh = np.meshgrid(*([vec]*D), indexing='ij')
    xy_reg = np.vstack([m.flatten() for m in mesh])

    # 3. Rotate grid points into Fibonacci basis
    xy_rot = V @ xy_reg

    # —— First filtering: keep points where dims 2..D lie in [-0.5, 0.5] ——
    mask_sub = np.all((xy_rot[1:] <= 0.5) & (xy_rot[1:] >= -0.5), axis=0)
    xy_sub  = xy_rot[:, mask_sub]

    # —— Second filtering: within subspace, keep all dims in [-0.5, 0.5] ——
    mask_full = np.all((xy_sub <= 0.5) & (xy_sub >= -0.5), axis=0)
    count_in  = mask_full.sum()
    # Ensure parity consistency
    assert (count_in % 2) == (L % 2), "Parity mismatch of points"

    # Compute how many symmetric points to add or remove
    n_add = (L - count_in) // 2

    # —— Third step: adjust around first-dimension borders ——
    xs = xy_sub[0, :]
    order = np.argsort(xs)
    xs_sorted = xs[order]

    plus_border  = np.searchsorted(xs_sorted,  0.5, side='right') - 1
    minus_border = np.searchsorted(xs_sorted, -0.5, side='left')

    if n_add > 0:
        # Add points: take nearest pairs just outside each border
        idx_plus  = order[plus_border+1 : plus_border+1+n_add]
        idx_minus = order[minus_border-n_add : minus_border]
        mask_full[idx_plus]  = True
        mask_full[idx_minus] = True
    elif n_add < 0:
        # Remove points: drop extra pairs near borders
        rem1 = order[plus_border+n_add+1 : plus_border+1]
        rem2 = order[minus_border : minus_border-n_add]
        mask_full[rem1] = False
        mask_full[rem2] = False

    # —— Fourth step: select exactly L points ——
    xy = xy_sub[:, mask_full]
    assert xy.shape[1] == L, f"Sample count mismatch: got {xy.shape[1]} vs {L}"

    # 4.1 Center selected points
    xy -= xy.mean(axis=1, keepdims=True)

    # 4.2 Optional rescale to fit within [-0.5, 0.5]
    if L > 1 and Rescale:
        border_wanted = 0.5 - 1.0 / (2*L)
        fac = np.max(np.abs(xy), axis=1, keepdims=True) / border_wanted
        xy /= fac

    # 5. Map uniform grid to Gaussian via inverse error function
    xy_equal = xy + 0.5
    xy_std   = np.sqrt(2.0) * erfinv(2*xy_equal - 1.0)
    # Normalize each dimension's standard deviation
    fac_mm   = xy_std.std(axis=1, keepdims=True)
    xy_stdmm = xy_std / fac_mm

    # 6. Eigen-decompose covariance and apply transform
    C += np.eye(D) * 1e-10  # Ensure numerical stability
    w, P = np.linalg.eigh(C)  # Eigenvalues w and eigenvectors P
    if np.any(w < 0):
        raise ValueError("Covariance matrix is not positive semi-definite.")

    # Final Gaussian samples
    xy_Gauss = P @ np.diag(np.sqrt(w)) @ xy_stdmm + mu
    return xy_Gauss
