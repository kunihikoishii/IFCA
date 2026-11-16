"""

Functions for IFCA analysis

"""
from pathlib import Path

from numba import njit
import numpy as np
from scipy.optimize import fmin_bfgs
from scipy.stats import special_ortho_group


@njit("Tuple((u4[:],u4[:,:],u4[:,:,:]))(u8,u8[:],u8[:],u8,u8)")
def _make_histograms(num_ch, timestamp, nanotime, bin_size, gap):
    """Make 1D, 2D, 3D histograms from photon data using numba.
    
    Parameters
    ----------
    num_ch : unsigned int
        Number of channels.
    timestamp : ndarray of unsigned int
        List of photon arrival times (or macrotime). (unit: clock)
    nanotime : ndarray of unsigned int
        List of emission delay times (or microtime). (unit: channel)
    bin_size : unsigned int
        Binning width. (unit: clock)
    gap : unsigned int
        A photon pair or a photon triplet is counted only if the
        intervals between photons are larger than `gap`. (unit: clock)

    Returns
    -------
    v, M, T : ndarray of unsigned int
        1D (vector), 2D (matrix), 3D (tensor) histogram.
    """
    num_photon = len(timestamp)
    v = np.zeros((num_ch), dtype=np.uint32)
    M = np.zeros((num_ch, num_ch), dtype=np.uint32)
    T = np.zeros((num_ch, num_ch, num_ch), dtype=np.uint32)
    photon_first = 0
    photon_last = 0

    for bin_start in range(0, timestamp[num_photon-1] - bin_size, bin_size):
        while timestamp[photon_first] < bin_start:
            photon_first += 1

        while timestamp[photon_last] < bin_start + bin_size:
            photon_last += 1

        for i in range(photon_first, photon_last):
            timestamp_i = timestamp[i]
            nanotime_i = nanotime[i]
            v[nanotime_i] += 1
            for j in range(i+1, photon_last):
                timestamp_j = timestamp[j]
                nanotime_j = nanotime[j]
                if timestamp_j - timestamp_i > gap:
                    M[nanotime_i, nanotime_j] += 1
                    for k in range(j+1, photon_last):
                        timestamp_k = timestamp[k]
                        nanotime_k = nanotime[k]
                        if timestamp_k - timestamp_j > gap:
                            T[nanotime_i, nanotime_j, nanotime_k] += 1

    return v, M, T


def make_tensors(
        d_photon: dict,
        bin_time, clock, gap, tcspc_num_ch, tcspc_bin_factor, detector_list,
        ):
    """Make 1D, 2D, 3D corrlation histograms from photon data.

    Parameters
    ----------
    d_photon : dict
        timestamp : ndarray of unsigned int
            List of photon arrival times (or macrotime). (unit: clock)
        nanotime : ndarray of unsigned int
            List of emission delay time (or microtime). (unit: channel)
        detector : ndarray of unsigned int
            List of detectors that detect the photons.
    bin_time : float
        Binning width. (unit: second)
    clock : float
        Clock period. (unit: second)
    gap : unsigned int
        A photon pair or a photon triplet is counted only if the
        intervals between photons are larger than `gap`. (unit: clock)
    tcspc_num_ch : unsigned int
        Number of TCPSC channels in the original nanotime data.
    tcspc_bin_factor : unsigned int
        Number of TCSPC channels to be combined together. This is
        for reducing the size of tensors.
    detector_list : List of unsigned int
        List of detectors used in the analysis.

    Returns
    -------
    d_tensor : dict
        vector : (N,) ndarray of float
            1D histogram normalized to a single bin.
        matrix : (N, N) ndarray of float
            2D histogram normalized to a single bin. The uncorrelated
            part is subtracted. 
        tensor : (N, N, N) ndarray of float
            3D histogram normalized to a single bin. The uncorrelated
            part is subtracted.
    bin_num : unsigned int
        Number of bins analyzed.
    """
    bin_size = np.uint64(np.round(bin_time/clock))
    num_ch_per_detector = np.uint64(np.ceil(tcspc_num_ch/tcspc_bin_factor))
    num_detectors = len(detector_list)
    num_ch = np.uint64(num_ch_per_detector * num_detectors)

    timestamp = d_photon["timestamp"].astype(np.uint64)
    nanotime = d_photon["nanotime"].astype(np.uint64)
    detector = d_photon["detector"].astype(np.uint64)

    is_valid = np.isin(detector, detector_list)
    timestamp = timestamp[is_valid]
    nanotime = nanotime[is_valid]
    detector = detector[is_valid]
    detector_index = np.zeros((np.amax(detector_list)+1), dtype=np.uint64)
    for i in range(0, num_detectors):
        detector_index[detector_list[i]] = i
    detector = detector_index[detector]
 
    nanotime //= tcspc_bin_factor
    nanotime += num_ch_per_detector * detector

    v, M, T = _make_histograms(num_ch, timestamp, nanotime, bin_size, gap)

    bin_num = np.int64(timestamp[-1] // bin_size)
    v = v.astype("f8") / bin_num
    M = M.astype("f8") / bin_num
    T = T.astype("f8") / bin_num

    M = (M + M.T) * bin_size**2 / ((bin_size-gap) * (bin_size-gap-1))
    T = T + np.transpose(T, (0, 2, 1)) \
        + np.transpose(T, (1, 0, 2)) + np.transpose(T, (1, 2, 0)) \
        + np.transpose(T, (2, 0, 1)) + np.transpose(T, (2, 1, 0))
    T *= bin_size**3 \
         / ((bin_size-gap*2) * (bin_size-gap*2-1) * (bin_size-gap*2-2))

    corM = M - np.einsum("i,j->ij", v, v)
    corT = T - np.einsum("ij,k->ijk", M, v) - np.einsum("ij,k->kij", M, v) \
           - np.einsum("ij,k->jki", M, v) + 2*np.einsum("i,j,k->ijk", v, v, v)

    d_tensor = dict()
    d_tensor["vector"] = v
    d_tensor["matrix"] = corM
    d_tensor["tensor"] = corT

    return d_tensor, bin_num

   
def accumulate_tensors(loader, args, path, file_name_pattern, **param):
    """Accumulate 1D, 2D, 3D histograms from multiple data files.

    Parameters
    ----------
    loader : function
        Appropriate function for loading photon data as a dictionary.
    args : tuple
        Additional parameters to be passed to the loader function.
    path : string
        Path to the folder in which photon data are stored.
    file_name_pattern : str
        Filter string to select files using wildcards.
    param : dict
        Additional parameters to be passed to `make_tensors` function.

    Returns
    -------
    d_acc_tensor : dict
        Dictionary returned from `make_tensors` is accumulated in
        `d_acc_tensor` and normalized.    
    """
    files = Path(path).glob(file_name_pattern)
    photons = np.int64(0)
    acc_bin_num = np.int64(0)
    d_acc_tensor = dict()
    num_files = 0

    for num_files, file in enumerate(files, 1):
        file_name = file.name

        print("Loading", file_name, end="... ")
        d_photon = dict(zip(("timestamp", "detector", "nanotime"),
                        loader(path+file_name, *args)))
 
        print(len(d_photon["timestamp"]),
                  "photons were found. Making tensors",
                  end="... ")
        d_tensor, bin_num= make_tensors(**param, d_photon=d_photon)
        if num_files == 1:
            d_acc_tensor = d_tensor.copy()
            d_acc_tensor["vector"] *= bin_num
            d_acc_tensor["matrix"] *= bin_num
            d_acc_tensor["tensor"] *= bin_num
        else:
            d_acc_tensor["vector"] += d_tensor["vector"] * bin_num
            d_acc_tensor["matrix"] += d_tensor["matrix"] * bin_num
            d_acc_tensor["tensor"] += d_tensor["tensor"] * bin_num

        print("Done.")
        photons += len(d_photon["timestamp"])
        acc_bin_num += bin_num
    
    if num_files == 0:
        print("No matching file found.")
        return
    d_acc_tensor["vector"] /= acc_bin_num
    d_acc_tensor["matrix"] /= acc_bin_num
    d_acc_tensor["tensor"] /= acc_bin_num

    print(num_files, "files loaded.", photons, "photons were found in", acc_bin_num, "bins in total.")
    print("Average photon counts / bin =", photons/acc_bin_num, ".")
    
    return d_acc_tensor, num_files, photons, acc_bin_num


def eigenvalue_analysis(d_tensor: dict):
    """Do error normalization and eigenvalue analysis.

    Parameters
    ----------
    d_tensor : dict
        vector : (N,) ndarray of float
            1D histogram. The square root of `vector` is regarded as
            the standard deviation of the data.
        matrix : (N, N) ndarray of float
            2D histogram.
        tensor : (N, N, N) ndarray of float
            3D histogram.

    Returns
    -------
    dict
        vector : (N,) ndarray of float
            Vector used for normalization.
        T_normalized : (N, N, N) ndarray of float
            Error-normalized 3D histogram.
        eigenvalue : (N,) ndarray of float
            List of eigenvalues of error-normalized 2D histogram.
            They are sorted in the descending order.
        eigenvector : (N, N) ndarray of float
            List of eigenvectors of error-normalized 2D histogram.
    """
    v = d_tensor["vector"].astype("f8")
    norm_factor = np.zeros(len(v))
    np.sqrt(v, where=v>0, out=norm_factor)
    np.reciprocal(norm_factor, where=norm_factor>0, out=norm_factor)

    M_norm = np.einsum("ij,i,j->ij", d_tensor["matrix"],
                       norm_factor, norm_factor)

    T_norm = np.einsum("ijk,i,j,k->ijk", d_tensor["tensor"],
                       norm_factor, norm_factor, norm_factor)

    eig_val, eig_vec = np.linalg.eigh(M_norm)

    eig_val = eig_val[::-1]
    eig_vec = eig_vec[:, ::-1]

    return {"T_normalized": T_norm,
            "eigenvalue": eig_val,
            "eigenvector": eig_vec,
            "v": v}


@njit("f8(f8[:],i8,i8,i8,f8[:,:],f8[:,:],f8[:,:,:])")
def _residual(x, n, p, q, U, W, TB):
    """Calculate the fitting residual in the Jacobi method using numba

    Parameters
    ----------
    x : (N,) array of float
        `x[0]` : rotation angle of the Givens matrix.
        `x[m, m>0]` : m-th diagonal element of a diagonal tensor.
    n : int
        Dimension of the Givens matrix.
    p, q : int
        Specify the rows and columns rotated by the Givens matrix.
    U : (N, N) ndarray of float
        Orthogonal matrix representing the result of the previous runs.
    W : (N, N) ndarray of float
        Diagonal matrix with diagonal elements equal to the square root
        of the eigenvalues of a 2D histogram.
    TB : (N, N, N) array of float
        Reduced 3D tensor. Obtained by multiplying selected
        eigenvectors of a 2D histogram to a 3D histogram.

    Returns
    -------
    float
        The sum of squared residuals of fitting to `TB`.
    """
    angle = x[0]
    G = np.identity(n)
    G[p, p] = np.cos(angle)
    G[q, p] = np.sin(angle)
    G[p, q] = -np.sin(angle)
    G[q, q] = np.cos(angle)
    #M = G @ U @ W
    WUT = np.zeros((n, n), dtype=np.float64)
    for i in range(0, n):
        for j in range(0, n):
            WUT[i, j] = W[i, i] * U.T[i, j]
    M = G @ WUT.T

    TB_sim = np.zeros((n, n, n), dtype=np.float64)
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                for m in range(0, n):
                    TB_sim[i, j, k] += x[m+1] * M[m, i] * M[m, j] * M[m, k]

    return np.sum((TB-TB_sim)**2)


def find_IFC(T_normalized, eigenvalue, eigenvector, v, n,
             maxiter=5000, tol=1e-13, gtol=1e-7, trial=1):
    """Find independent fluorescence components using Jacobi method.
    
    Parameters
    ----------
    T_normalized : (N, N, N) ndarray of float
        Error-normalized 3D histogram.
    eigenvalue : (N,) ndarray of float
        List of eigenvalues of an error-normalized 2D histogram
        sorted in the descending order.
    eigenvector : (N, N) ndarray of float
        List of eigenvectors of an error-normalized 2D histogram.
    v : (N,) ndarray of float
        1D histogram used for error normalization.
    n : int
        Number of significant independent components. Usually judged
        from the eigenvalues.
    maxiter : int, default 5000
        Maximum number of iterations in the Jacobi method.
    tol : float, default 1e-13
        Tolerance for judging convergence in the Jacobi method
    gtol : float, default 1e-7
        Gradient tolerance for judging convergence in numerical
        optimization using fmin_bfgs.
    trial : int, default 1
        Number of trials with different initial guesses. It can help
        find the global minimum when a single trial tends to be trapped
        in a local minimum.
    
    Returns
    -------
    dict
        observed_tensor : (N, N, N) ndarray of float
        fitted_tensor : (N, N, N) ndarray of float
        rotation_matrix : (N, N) ndarray of float
        reduced_eigenvectors : (M, N) ndarray of float
        square_root_reduced_eigenvalues : (N, N) ndarray of float
        number_of_molecules_from_M&T : (N,) ndarray of float
        number_of_molecules_from_v&M : (N,) ndarray of float
        single_molecule_decay : (M, N) ndarray of float
    """
    B = eigenvector[:,:n]
    W = np.diag(np.sqrt(eigenvalue[:n]))
    inv_W = np.diag(1 / np.sqrt(eigenvalue[:n]))
    TB = np.tensordot(np.tensordot(np.tensordot(T_normalized,
         B, axes=(2, 0)), B, axes=(1, 0)), B, axes=(0, 0))

    U_best = np.identity(n)
    H_best = np.ones(n, dtype=np.float64)
    f_best = _residual(np.zeros(n+1), n, 0, 1, U_best, W, TB)
    for i in range(0, trial):
        U = special_ortho_group.rvs(n)
        H = np.ones(n, dtype=np.float64)
        x0 = np.zeros(n+1, dtype=np.float64)
        f = _residual(x0, n, 0, 1, U, W, TB)
        for iter in range(1, maxiter+1):
            for p in range(0, n-1):
                for q in range(p+1, n):
                    for m in range(0, n):
                        x0[m+1] = H[m]
                    xopt, fopt, *_ = fmin_bfgs(
                        _residual,
                        x0,
                        args=(n, p, q, U, W, TB),
                        gtol=gtol,
                        full_output=True,
                        disp=False,
                        )
                    angle = xopt[0]
                    G = np.identity(n)
                    G[p, p] = np.cos(angle)
                    G[q, p] = np.sin(angle)
                    G[p, q] = -np.sin(angle)
                    G[q, q] = np.cos(angle)
                    U = G @ U
                    for m in range(0, n):
                        H[m] = xopt[m+1]
            if (f - fopt) < tol:
                f = fopt
                break
            f = fopt
        print("Trial", i+1, ": Residual =", f, ", number of sweeps =", iter)
        if f < f_best:
            f_best = f
            U_best = U.copy()
            H_best = H.copy()
    print("Residual, best of", trial, "trials =", f_best)

    U = U_best
    H = H_best
    for m in range(0, n):
        U[m, :] *= np.sign(H[m])
        H[m] = np.abs(H[m])

    i = np.argsort(H)
    U = U[i, :]
    H = H[i]

    HUW = np.einsum("m,mi,mj,mk->ijk", H, U @ W, U @ W, U @ W)

    n_mol_MT = np.array([1 / H[m]**2 for m in range(0, n)])

    n_mol_vM = np.square(U @ inv_W @ B.T @ np.sqrt(v)) 

    sm_v = np.diag(np.sqrt(v)) @ B @ W @ U.T @ np.diag(1 / np.sqrt(n_mol_MT))

    return {"observed_tensor": TB,
            "fitted_tensor": HUW,
            "rotation_matrix": U,
            "reduced_eigenvectors": B,
            "square_root_reduced_eigenvalues": W,
            "number_of_molecules_from_M&T": n_mol_MT,
            "number_of_molecules_from_v&M": n_mol_vM,
            "single_molecule_decay": sm_v}