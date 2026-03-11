import numpy as np 
from cosmotracer.tcn import calculate_steady_state_concentration

from scipy.optimize import differential_evolution
from scipy.stats import linregress
from scipy.interpolate import (
    UnivariateSpline,
    RegularGridInterpolator
)

from multiprocessing import Pool

import gc

def _grid_worker(args):

    i, j, csyn, esyn, C_meas, Mcomp, n_repeats = args

    draws = draw_composites(
        values=csyn,
        weights=esyn,
        m_grains_per_composite=Mcomp,
        n_draws=n_repeats
    )

    W = np.zeros(n_repeats)

    for k in range(n_repeats):
        W[k] = wasserstein2_distance(
            u_values=draws[k],
            v_values=C_meas
        )
    
    return i, j, W


def wasserstein2_distance(u_values, v_values, u_weights=None, v_weights=None):
    """Wasserstein-2 distance between two 1D distributions."""
    # Sort and normalize weights
    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)
    u_values = np.asarray(u_values)[u_sorter]
    v_values = np.asarray(v_values)[v_sorter]

    if u_weights is None:
        u_weights = np.ones(len(u_values)) / len(u_values)
    else:
        u_weights = np.asarray(u_weights)[u_sorter]
        u_weights = u_weights / u_weights.sum()

    if v_weights is None:
        v_weights = np.ones(len(v_values)) / len(v_values)
    else:
        v_weights = np.asarray(v_weights)[v_sorter]
        v_weights = v_weights / v_weights.sum()

    # Build CDFs, then sample both quantile functions on a common grid
    u_cdf = np.concatenate([[0], np.cumsum(u_weights)])
    v_cdf = np.concatenate([[0], np.cumsum(v_weights)])
    all_quantiles = np.unique(np.concatenate([u_cdf, v_cdf]))

    # Interpolate inverse CDFs (quantile functions) on the common grid
    u_quantile = np.interp(all_quantiles, u_cdf, np.concatenate([[u_values[0]], u_values]))
    v_quantile = np.interp(all_quantiles, v_cdf, np.concatenate([[v_values[0]], v_values]))

    # Integrate |Q_u - Q_v|^2 over [0,1]
    deltas = np.diff(all_quantiles)
    integrand = np.square(u_quantile[:-1] - v_quantile[:-1])
    return np.sqrt(np.dot(integrand, deltas))

def draw_composites(
    values: np.ndarray,
    weights: np.ndarray,
    m_grains_per_composite: np.ndarray,
    n_draws: int = 1,
):
    """
    Draw synthetic composite samples.

    Parameters
    ----------
    weights : array
        Sampling probabilities (usually erosion rates).
    n_draws : int
        Number of independent composite draws.

    Returns
    -------
    out : ndarray
        Shape (n_draws, n_composites) containing mean composite
        concentrations. If n_draws == 1, returns 1D array.
    """

    weights = weights / np.sum(weights)

    M = np.asarray(m_grains_per_composite)
    n_comp = len(M)

    grains_per_draw = np.sum(M)
    tot_draws = n_draws * grains_per_draw

    # --------------------------------------
    # Draw all grains in one vectorized call
    # --------------------------------------

    all_samples = np.random.choice(
        a=values,
        size=tot_draws,
        replace=True,
        p=weights
    )

    # --------------------------------------
    # Compute composite means
    # --------------------------------------

    # starting indices of each composite
    offsets = np.cumsum(np.r_[0, np.tile(M, n_draws)[:-1]])

    sums = np.add.reduceat(all_samples, offsets)

    # repeat composite sizes for each draw
    sizes = np.tile(M, n_draws)

    means = sums / sizes

    # reshape into draws × composites
    out = means.reshape(n_draws, n_comp)

    return out

def smooth_valley(K_vals, n_vals, W_vals, smooth_fact=0.01, n_points=200):
    """
    Smooth along a misfit valley in 2D (K,n) space and retain mappings
    to actual K and n for plotting or indexing.
    
    Parameters
    ----------
    K_vals, n_vals, W_vals : array-like
        Arrays of valley points (same length)
    smooth_fact : float
        Smoothing factor for UnivariateSpline
    n_points : int
        Number of points along the smoothed curve

    Returns
    -------
    K_min, n_min : float
        Values of K and n at the minimum of smoothed W
    s_smooth : ndarray
        Normalized distance along the valley (0..1)
    W_smooth : ndarray
        Smoothed W along s_smooth
    K_smooth, n_smooth : ndarray
        Interpolated K and n along s_smooth
    """
    
    K_vals = np.asarray(K_vals)
    n_vals = np.asarray(n_vals)
    W_vals = np.asarray(W_vals)
    
    # cumulative distance along valley
    dK = np.diff(K_vals, prepend=K_vals[0])
    dn = np.diff(n_vals, prepend=n_vals[0])
    s = np.cumsum(np.sqrt(dK**2 + dn**2))
    s = (s - s.min()) / (s.max() - s.min())  # normalize 0..1
    
    # smooth W along s
    spl = UnivariateSpline(s, W_vals, s=smooth_fact)
    s_smooth = np.linspace(0, 1, n_points)
    W_smooth = spl(s_smooth)
    
    # interpolate K and n along s_smooth
    K_smooth = np.interp(s_smooth, s, K_vals)
    n_smooth = np.interp(s_smooth, s, n_vals)
    
    # find minimum
    id_min = np.argmin(W_smooth)
    K_min = K_smooth[id_min]
    n_min = n_smooth[id_min]
    
    return K_min, n_min, s_smooth, W_smooth, K_smooth, n_smooth

class TCNDistributionFit:
    
    def __init__(
        self,
        measured_concentrations: np.ndarray,
        grains_per_composite: int|np.ndarray,
        steepness_indices: np.ndarray,
        halflife: float = np.inf,
        density: float = 2.7
    ):
        
        """
        Note: depending on the application, C_true can be single grain measurements
        (in which case grains_per_composite=1), or composite samples with >1 grain each.
        May be an array if composites have different numbers of grains.
        """
        
        self._P_dict = {} 
        self._ks = steepness_indices
        self._C = measured_concentrations
        
        if hasattr(grains_per_composite, "__len__"):
            self._Mcomp = grains_per_composite
        else:
            self._Mcomp = np.full_like(
                measured_concentrations, 
                grains_per_composite
            )
        
        self._t12 = halflife
        self._rho = density
        
    
    def add_production_pathway(
        self,
        scaling_factors: np.ndarray,
        attenuation_length: float = 160.,
        reaction="sp",
        p_slhl=4.03,
    ):
        self._P_dict[reaction] = {
            "p": scaling_factors*p_slhl,
            "att": attenuation_length,
        }
    
    def _calculate_concs(
        self,
        n,
        logK
    ):
        
        e = 10**logK * self._ks ** n
        
        c_out = np.zeros_like(self._ks)
        
        for pw, attr in self._P_dict.items():
            c_out += calculate_steady_state_concentration(
                exhumation_rate=e,
                bulk_density=self._rho,
                production_rate=attr["p"],
                attenuation_length=attr["att"],
                halflife=self._t12
            )
        
        return c_out
    
    def calculate_W_gridsearch(
        self,
        n_vals,
        logK_vals,
        n_repeats=100,
        parallel=True,
        n_workers=None
    ):

        W_out = np.zeros((len(logK_vals), len(n_vals), n_repeats))

        tasks = []

        for i, logK in enumerate(logK_vals):
            for j, n in enumerate(n_vals):
                
                csyn = self._calculate_concs(n, logK)
                esyn = 10**logK * self._ks**n
                
                tasks.append((
                    i,
                    j,
                    csyn,
                    esyn,
                    self._C,
                    self._Mcomp,
                    n_repeats
                ))

        if parallel:

            with Pool(n_workers) as pool:
                results = pool.map(_grid_worker, tasks)

            for i, j, W in results:
                W_out[i, j, :] = W

        else:

            for task in tasks:
                i, j, W = _grid_worker(task)
                W_out[i, j, :] = W
        
        logK_min, n_min = self._grid_best_fit_smooth_center(
            W=np.median(W_out, axis=-1), 
            n_vals=n_vals,
            logK_vals=logK_vals
        )

        return W_out, logK_min, n_min

    def _grid_best_fit_smooth_center(
        self,
        W,
        n_vals,
        logK_vals,
        cutoff=np.inf
    ):
        i_order = []
        j_order = []
        for j in range(W.shape[1]):
            i_min = np.argmin(W[:,j])
            if W[i_min, j] < cutoff:
                i_order.append(i_min)
                j_order.append(j)
                
        res = linregress(n_vals[j_order], logK_vals[i_order])
        
        # sample along this line...
        n_out = n_vals[j_order]
        logk_out = res.slope * n_out + res.intercept

        # --- 2D interpolator ---
        interp = RegularGridInterpolator(
            (logk_out, n_vals),
            W,
            bounds_error=False,
            fill_value=np.nan
        )

        # points must be (N, 2) = (logK, n)
        points = np.column_stack([logk_out, n_out])
        W_out = interp(points)
        
        logK_min, n_min, s_smooth, W_smooth, K_smooth, n_smooth = smooth_valley(
            logk_out, n_out, W_out, smooth_fact=0.01
        )
        
        return logK_min, n_min


    def _misfit_diff_evolve(self, par):
        
        n_repeats = self._de_n_repeats

        logK, n = par

        esyn = 10**logK * self._ks ** n

        csyn = self._calculate_concs(n, logK)

        draws = draw_composites(
            values=csyn,
            weights=esyn,
            m_grains_per_composite=self._Mcomp,
            n_draws=n_repeats
        )

        W = np.zeros(n_repeats)

        for k in range(n_repeats):
            W[k] = wasserstein2_distance(
                draws[k,:],
                self._C
            )

        return np.median(W)
    
    def calculate_W_diff_evolve(
        self,
        bounds=(( -8, -3 ), (0.25, 1.75)),
        n_repeats=100,
        popsize=12,
        maxiter=40,
        workers=1
    ):
        
        self._de_n_repeats = n_repeats

        result = differential_evolution(
            func=self._misfit_diff_evolve,
            bounds=bounds,
            popsize=popsize,
            maxiter=maxiter,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            workers=workers
        )

        logK_best, n_best = result.x

        return {
            "logK": logK_best,
            "n": n_best,
            "misfit": result.fun
        }
                
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import cosmotracer as ct
    
    import time 
    import os 
    print(os.cpu_count())
    
    np.random.seed(1)
    
    # Test this on the synthetic example...
    
    wd = r"C:\Users\Lennart\OneDrive\Desktop\phd\Work\Writing\ModellingStudy\Data\Kauaii"
    
    out = ct.synthetic.parse_RunHandler_output(
        wd=r"C:\Users\Lennart\OneDrive\Desktop\phd\Work\Writing\ModellingStudy\Data\transconc_2d\Ubase1mmyr_m05_n1",
        fname="model_U0.001_Ksp1e-05_Kdiff0_n1_m0.5_dt1000_Tmax2000000.h5",
    )
    #  select time step where uplift is 0.0015 (half way propagated knickpoint)
    i_use = np.argmin((out["mean_true_exhum"]-0.0015)**2)
    P_dict={"sp": 4.09, "totmu": 0.024, "nmu": 0.027}
    att_dict={"sp": 160, "totmu": 4320, "nmu": 1510}
    halflife=1.5e6
    
    node_concs = out["core_node_conc"][i_use,:]
    node_exhum = out["core_node_exhum"][i_use, out["id_tracked_core"]]
    node_ksn = (node_exhum/1e-5)**(1/1)
    
    # Scaling
    sp_s = out["core_node_scaling_sp"][i_use,:]
    totmu_s = out["core_node_scaling_totmu"][i_use,:]
    nmu_s = out["core_node_scaling_nmu"][i_use,:]
    
    # create a synthetic set of composites
    grains_per_composite = [
        32, 36, 38, 30, 31, 33, 31, 
        31, 37, 36, 31, 24, 31, 27, 
        29, 26, 24, 27, 25, 36, 32, 
        38, 36, 29, 39, 40
    ]
    
    true_comps = draw_composites(
        values=node_concs,
        weights=node_exhum,
        m_grains_per_composite=grains_per_composite,
        n_draws=1
    )[0]

    
    cf = TCNDistributionFit(
        measured_concentrations=true_comps,
        grains_per_composite=grains_per_composite,
        steepness_indices=node_ksn,
        halflife=1.5e6,
        density=2.7
    )
    
    cf.add_production_pathway(
        scaling_factors=sp_s,
        attenuation_length=att_dict["sp"],
        reaction="sp",
        p_slhl=P_dict["sp"]
    )
    cf.add_production_pathway(
        scaling_factors=totmu_s,
        attenuation_length=att_dict["totmu"],
        reaction="totmu",
        p_slhl=P_dict["totmu"]
    )
    cf.add_production_pathway(
        scaling_factors=nmu_s,
        attenuation_length=att_dict["nmu"],
        reaction="nmu",
        p_slhl=P_dict["nmu"]
    )
    
    res = 100
    logK_vals = np.linspace(-7, -3, res)
    n_vals = np.linspace(0.25, 1.75, res)
    
    n_true_samples = 50
    
    lK_best = []
    n_best = []
    
    for i in range(n_true_samples):
        true_comps = draw_composites(
            values=node_concs,
            weights=node_exhum,
            m_grains_per_composite=grains_per_composite,
            n_draws=1
        )[0]
        cf._C = true_comps
    
        """start = time.time()
        W_raw, logK, n = cf.calculate_W_gridsearch(
            n_vals=n_vals,
            logK_vals=logK_vals,
            n_repeats=50,
            parallel=True,
            n_workers=5
        )
        end = time.time()
        
        print("Gridsearch time:", end-start, logK, n)"""
        
        # Use diff evolve
        start = time.time()
        out = cf.calculate_W_diff_evolve(
            n_repeats=200,
            workers=5
        )
        end = time.time()
        print("Diff evlove time:", end-start, out["logK"], out["n"])
        
        lK_best.append(out["logK"])
        n_best.append(out["n"])
        
        gc.collect()
    
    W_raw, logK, n = cf.calculate_W_gridsearch(
        n_vals=n_vals,
        logK_vals=logK_vals,
        n_repeats=50,
        parallel=True,
        n_workers=5
    )
    W_med = np.median(W_raw, axis=-1)
    
    plt.imshow(
        np.log10(W_med),
        extent=[
            n_vals.min(), n_vals.max(),
            logK_vals.min(), logK_vals.max()
        ],
        origin="lower",
        aspect="auto"
    )
    
    plt.scatter(n_best, lK_best)
    plt.show()
        

