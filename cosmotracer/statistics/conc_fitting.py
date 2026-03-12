import numpy as np 
from cosmotracer.tcn import calculate_steady_state_concentration
from .base import wasserstein2_distance, draw_composites

from scipy.optimize import differential_evolution
from scipy.stats import linregress
from scipy.interpolate import (
    UnivariateSpline,
    RegularGridInterpolator
)

from joblib import Parallel, delayed

import gc

def _grid_worker(args):
    
    """
    Internal worker to calculate Wasserstein-2 distances for some draw of
    composite samples.
    
    Parameters
    ----------
    args : tuple
        Tuple of i, j, C given some n, K, E given some n, K, true concs,
        number of grains per sample, number of random draws from the synthetic
        concs.
        
    Returns
    -------
    i : int
        Index for K
    j : int 
        Index for n
    W : array
        1-D np.array contain the W2 distances of each synthetic draw compared to
        the actual distribution of the samples.
    """
    

    i, j, csyn, esyn, C_true, Mcomp, n_repeats = args

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
            v_values=C_true
        )
    
    return i, j, W



class TCNDistributionFit:
    """
    Fit stream-power model parameters to cosmogenic nuclide concentration
    distributions.

    This class provides tools to infer landscape erosion parameters by
    comparing measured cosmogenic nuclide concentrations with synthetic
    concentrations generated from a landscape erosion model. Synthetic
    concentrations are computed from steepness indices using the stream-power
    relation

        E = K * k_s^n

    where ``E`` is erosion rate, ``k_s`` is channel steepness, and ``K`` and
    ``n`` are stream-power parameters. 

    Given a set of steepness indices and production parameters, the class
    generates synthetic nuclide concentrations for some n, K.
    Synthetic samples are then drawn from this population and grouped into
    composite samples (representing multiple grains per measurement, analogous
    to the actually measured samples). The model tries to minimise the 
    misfit in synthetic and true distribution by optimising n, and K. The 
    metric that is used is a Wasserstein-2 metric (Earth Mover Distance).

    This framework can also be used for exploration of how sampling strategy 
    (number of samples, grains per composite) affects the ability to recover 
    underlying landscape parameters.

    Parameters
    ----------
    measured_concentrations : ndarray
        Measured cosmogenic nuclide concentrations for each sample.
        These may represent either individual grains or composite samples.

    grains_per_composite : int or ndarray
        Number of grains contributing to each composite sample. If an integer,
        all samples are assumed to contain the same number of grains. If an
        array, it must have the same length as ``measured_concentrations``,
        allowing variable composite sizes.

    steepness_indices : ndarray
        Channel steepness indices (k_s) for each landscape node used to
        generate the synthetic concentration population.

    halflife : float, optional
        Half-life of the cosmogenic nuclide (years). Defaults to ``np.inf``,
        corresponding to a stable nuclide.

    density : float, optional
        Rock density in g cm⁻³ used in the production–erosion equation.
        Default is 2.7 g cm⁻³.

    Notes
    -----
    The class typically follows this workflow:

    1. Initialize with measured concentrations and landscape steepness indices.
    2. Add production pathways (spallation, muons, etc.).
    3. Generate synthetic concentrations for candidate stream-power parameters.
    4. Draw synthetic composite samples from the landscape distribution.
    5. Compare measured and synthetic distributions using a Wasserstein metric.

    This approach can be used to evaluate parameter identifiability and
    sampling strategies for cosmogenic nuclide studies.
    """
    
    def __init__(
        self,
        measured_concentrations: np.ndarray,
        grains_per_composite: int|np.ndarray,
        steepness_indices: np.ndarray,
        halflife: float = np.inf,
        density: float = 2.7
    ):
        
        """
        Initialize a concentration distribution fitting instance.

        Parameters
        ----------
        measured_concentrations : ndarray
            Measured cosmogenic nuclide concentrations for each sample. These
            values define the target distribution that synthetic samples will be
            compared against.

        grains_per_composite : int or ndarray
            Number of grains contributing to each composite sample.

            - If an integer, all samples are assumed to contain the same number
            of grains.
            - If an array, it must have the same length as
            ``measured_concentrations`` and specifies the number of grains
            in each composite sample individually.

        steepness_indices : ndarray
            Channel steepness indices (k_s) describing the landscape nodes from
            which synthetic grains are drawn.

        halflife : float, optional
            Half-life of the cosmogenic nuclide in years. Use ``np.inf`` for
            stable nuclides (default).

        density : float, optional
            Rock density in g cm⁻³ used in the erosion–production calculation.
            Default is 2.7 g cm⁻³.

        Notes
        -----
        The measured concentrations represent the empirical distribution that
        the model attempts to reproduce. Synthetic concentrations are generated
        from erosion rates derived from the steepness indices via a stream-power
        relation. Composite samples are simulated by randomly drawing grains from
        the synthetic population according to erosion-rate-weighted probabilities.
        """
        
        self._C = np.asarray(measured_concentrations)
        self._ks = np.asarray(steepness_indices)

        if self._C.ndim != 1:
            raise ValueError("measured_concentrations must be 1D")

        if self._ks.ndim != 1:
            raise ValueError("steepness_indices must be 1D")
        
        self._P_dict = {} 
        
        if hasattr(grains_per_composite, "__len__"):
            self._Mcomp = np.asarray(grains_per_composite)

            if len(self._Mcomp) != len(self._C):
                raise ValueError(
                    "grains_per_composite must match number of measured samples"
                )
        else:
            self._Mcomp = np.full(len(self._C), grains_per_composite)
        
        self._t12 = halflife
        self._rho = density
        
    
    def add_production_pathway(
        self,
        scaling_factors: np.ndarray,
        attenuation_length: float = 160.,
        pathway="sp",
        p_slhl=4.03,
    ):
        
        """
        Add a cosmogenic nuclide production pathway to the model.

        Production pathways represent different mechanisms by which
        cosmogenic nuclides are produced in rock (e.g., spallation or
        muon reactions). Each pathway is defined by spatially variable
        production scaling factors, an attenuation length that
        controls how production decreases with depth, and an absolute
        production rate referenced to Sea-Level-High-Latitude.

        Multiple pathways can be added and their contributions are summed
        when calculating steady-state concentrations.

        Parameters
        ----------
        scaling_factors : ndarray
            Spatial scaling factors for surface production rates at each
            landscape node. These typically account for variations in
            elevation, latitude, and shielding. The array must have the
            same length as the steepness index array used to define the
            landscape.

        attenuation_length : float, optional
            Effective attenuation length (g cm⁻²) for this production
            pathway. Default is 160 g cm⁻², typical for spallation.

        pathway : str, optional
            Identifier for the production pathway. 

        p_slhl : float, optional
            Reference sea-level high-latitude (SLHL) production rate
            for the nuclide (atoms g⁻¹ yr⁻¹). Default is 4.03.

        Notes
        -----
        The effective production rate used in concentration calculations is

            P = scaling_factors * p_slhl

        where ``p_slhl`` is the reference production rate and
        ``scaling_factors`` account for site-specific scaling.
        """
        
        self._P_dict[pathway] = {
            "p": scaling_factors*p_slhl,
            "att": attenuation_length,
        }
    
    def _calculate_concs(
        self,
        n: float,
        logK: float
    ):
        
        """
        Compute synthetic steady-state cosmogenic nuclide concentrations.

        This method calculates the surface concentration expected at each
        landscape node for a given set of stream-power parameters. Erosion
        rates are first derived from the steepness indices using the
        stream-power relation

            E = K * k_s^n

        where ``K = 10**logK``. For each node, the steady-state nuclide
        concentration is then computed using the configured production
        pathways. Note that this function assumes steady state exhumation!

        Parameters
        ----------
        n : float
            Stream-power slope exponent.

        logK : float
            Base-10 logarithm of the stream-power coefficient ``K``.

        Returns
        -------
        ndarray
            Synthetic steady-state nuclide concentrations at each
            landscape node. The returned array has the same length as
            ``self._ks``.

        Notes
        -----
        Contributions from all production pathways added via
        ``add_production_pathway`` are summed to obtain the total
        concentration at each node.
        """
        
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
        n_workers=-1,
        smooth_fact=1
    ):
        
        """
        Evaluate model misfit across a grid of stream-power parameters.

        This method performs a grid search over combinations of the
        stream-power parameters ``n`` and ``log10(K)``. For each parameter
        pair, synthetic concentrations are generated for all landscape
        nodes and composite samples are repeatedly drawn from this
        synthetic population. The similarity between the synthetic and
        observed concentration distributions is quantified using the
        Wasserstein-2 distance.

        Because synthetic sampling introduces stochastic variability,
        multiple draws are performed for each parameter combination.
        The resulting distribution of Wasserstein distances can be used
        to assess the robustness of the model fit.

        The best-fit parameters are determined by identifying the
        trade-off valley in the median misfit surface and fitting a
        smooth centerline through this valley. The minimum of the
        smoothed centerline is taken as the preferred parameter
        estimate, which reduces sensitivity to noise in the grid
        search surface.

        Parameters
        ----------
        n_vals : ndarray
            Array of stream-power slope exponent values to evaluate.

        logK_vals : ndarray
            Array of base-10 logarithmic stream-power coefficient values
            to evaluate.

        n_repeats : int, optional
            Number of synthetic composite draws performed for each
            parameter pair. Default is 100.

        parallel : bool, optional
            If True, evaluate grid points in parallel using ``joblib``.
            Default is True.

        n_workers : int, optional
            Number of parallel workers used when ``parallel=True``.
            Default is ``-1`` (use all available cores).

        Returns
        -------
        W_out : ndarray
            Array of Wasserstein distances with shape
            ``(len(logK_vals), len(n_vals), n_repeats)``. Each entry
            contains the misfit for one synthetic draw.

        logK_min : float
            Estimated best-fit value of ``log10(K)``.

        n_min : float
            Estimated best-fit value of ``n``.

        Notes
        -----
        For each parameter pair:

        1. Synthetic concentrations are computed from the landscape
        steepness indices.
        2. Composite samples are drawn according to erosion-rate
        weighting.
        3. The Wasserstein distance between the synthetic and measured
        concentration distributions is calculated.

        The misfit surface typically exhibits a trade-off valley
        between ``n`` and ``K``. Rather than selecting the absolute
        minimum grid cell (which may be sensitive to stochastic noise),
        the algorithm:

        1. Identifies the valley center by locating column-wise minima.
        2. Fits a linear trend through these points.
        3. Interpolates misfit values along this trend.
        4. Applies smoothing to obtain a stable minimum.

        This procedure provides a more robust estimate of the preferred
        stream-power parameters.
        """

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

            results = Parallel(n_jobs=n_workers, backend="loky")(
                delayed(_grid_worker)(task) for task in tasks
            )

            for i, j, W in results:
                W_out[i, j, :] = W

        else:

            for task in tasks:
                i, j, W = _grid_worker(task)
                W_out[i, j, :] = W
        
        out = self._grid_best_fit_smooth_center(
            W=np.median(W_out, axis=-1), 
            n_vals=n_vals,
            logK_vals=logK_vals,
            smoothing_fact=smooth_fact
        )
        
        out["W_raw"] = W_out
        
        # also calculate the raw min of n, K, W
        rc_min = np.unravel_index(
            np.argmin(np.median(W_out, axis=-1).flatten()),
            shape=(len(logK_vals), len(n_vals))
        )
        out["logK_raw"] = logK_vals[rc_min[0]]
        out["n_raw"] = n_vals[rc_min[1]]

        return out

    def _grid_best_fit_smooth_center(
        self,
        W,
        n_vals,
        logK_vals,
        cutoff=np.inf,
        smoothing_fact=1
    ):
        """
        Identify the best-fit parameters from a grid-search misfit surface.

        This method estimates the location of the minimum misfit in a
        two-dimensional ``(n, logK)`` grid by identifying the centerline
        of the trade-off valley in the misfit surface. The approach reduces
        sensitivity to stochastic noise introduced by synthetic sampling.

        Parameters
        ----------
        W : ndarray
            Two-dimensional misfit surface (e.g., median Wasserstein
            distances) with shape ``(len(logK_vals), len(n_vals))``.

        n_vals : ndarray
            Stream-power exponent values corresponding to the second
            dimension of ``W``.

        logK_vals : ndarray
            Logarithmic stream-power coefficient values corresponding
            to the first dimension of ``W``.

        cutoff : float, optional
            Optional misfit threshold used to discard poor-fitting
            grid cells when identifying the trade-off valley.
            Default is ``np.inf``.

        Returns
        -------
        logK_min : float
            Estimated best-fit value of ``log10(K)``.

        n_min : float
            Estimated best-fit value of ``n``.

        Notes
        -----
        The procedure:

        1. Finds the minimum misfit along each ``n`` column of the grid.
        2. Fits a linear regression through these minima to approximate
        the valley centerline.
        3. Interpolates misfit values along this line.
        4. Applies smoothing to determine the minimum of the valley.

        This approach stabilizes parameter estimation when the grid
        search surface contains noise from stochastic sampling.
        """
        
        i_order = []
        j_order = []
        for j in range(W.shape[1]):
            i_min = np.argmin(W[:,j])
            if W[i_min, j] < cutoff:
                i_order.append(i_min)
                j_order.append(j)
                
        # instead of interpolating on W (which pulls non-valley values),
        # convert n, K to s and interpolate the best W line to higher res.
        # Then smooth and return!
        
        logk = logK_vals[i_order]
        n = n_vals[j_order]
        
        # cumulative distance along valley
        dK = np.diff(logk, prepend=logk[0])
        dn = np.diff(n, prepend=n[0])
        s = np.cumsum(np.sqrt(dK**2 + dn**2))
        s = (s - s.min()) / (s.max() - s.min())  # normalize 0..1
        
        # smooth W along s
        spl = UnivariateSpline(s, W[i_order, j_order], s=smoothing_fact)
        s_smooth = np.linspace(0, 1, len(s)*2)
        W_smooth = spl(s_smooth)
        
        # interpolate K and n along s_smooth
        logK_smooth = np.interp(s_smooth, s, logk)
        n_smooth = np.interp(s_smooth, s, n)
        
        # find minimum
        id_min = np.argmin(W_smooth)
        logK_min = logK_smooth[id_min]
        n_min = n_smooth[id_min]
        
        out = {
            "logK": logK_min,
            "n": n_min,
            "logK_line": logK_smooth,
            "n_line": n_smooth,
            "s_line": s_smooth,
            "W_line": W_smooth
        }
        
        return out


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
        
        """
        Estimate stream-power parameters using differential evolution.

        This method searches for the stream-power parameters ``log10(K)``
        and ``n`` that minimize the Wasserstein misfit between observed
        cosmogenic nuclide concentrations and synthetic concentrations
        generated from the landscape model. Optimization is performed
        using the global differential evolution algorithm.

        For each candidate parameter pair, synthetic concentrations are
        computed across the landscape and composite samples are repeatedly
        drawn from this synthetic population. The Wasserstein distance
        between synthetic and measured concentration distributions is
        evaluated for each draw, and the resulting values are aggregated
        to produce the misfit used by the optimizer.

        Parameters
        ----------
        bounds : tuple of tuple, optional
            Parameter bounds for ``(log10(K), n)``. The default bounds are
            ``((-8, -3), (0.25, 1.75))``.

        n_repeats : int, optional
            Number of synthetic composite draws used to evaluate the
            misfit for each parameter pair. Increasing this value reduces
            stochastic noise in the misfit but increases computational
            cost. Default is 100.

        popsize : int, optional
            Population size multiplier used by the differential evolution
            algorithm. Larger values improve exploration of parameter
            space but increase runtime. Default is 12.

        maxiter : int, optional
            Maximum number of optimization generations. Default is 40.

        workers : int or map-like callable, optional
            Number of parallel workers used by the optimizer. If greater
            than 1, candidate parameter evaluations are performed in
            parallel. Default is 1 (no parallelization).

        Returns
        -------
        dict
            Dictionary containing the best-fit parameters:

            - ``"logK"`` : best-fit value of ``log10(K)``
            - ``"n"`` : best-fit value of ``n``
            - ``"misfit"`` : Wasserstein misfit at the optimum

        Notes
        -----
        Differential evolution is a stochastic global optimization method
        that is well suited for noisy objective functions such as those
        produced by repeated synthetic sampling. Compared to the grid
        search approach, this method can locate optimal parameters more
        efficiently when the parameter space is large.
        """
        
        self._de_n_repeats = n_repeats

        result = differential_evolution(
            func=self._misfit_diff_evolve,
            bounds=bounds,
            popsize=popsize,
            maxiter=maxiter,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            workers=workers,
            updating="deferred"
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
    grains_per_composite = np.array([
        32, 36, 38, 30, 31, 33, 31, 
        31, 37, 36, 31, 24, 31, 27, 
        29, 26, 24, 27, 25, 36, 32, 
        38, 36, 29, 39, 40
    ])
    
    # cut number of grains in half...
    grains_per_composite = [g//3 for g in grains_per_composite]
    
    true_comps = draw_composites(
        values=node_concs,
        weights=node_exhum,
        m_grains_per_composite=grains_per_composite,
        n_draws=1
    )[0]
    
    # compare sample and true KDE
    kdeL = ct.statistics.GaussianKDE(
        mu=node_concs, weights=node_exhum
    )
    kdeS = ct.statistics.GaussianKDE(
        mu=true_comps
    )

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
        pathway="sp",
        p_slhl=P_dict["sp"]
    )
    cf.add_production_pathway(
        scaling_factors=totmu_s,
        attenuation_length=att_dict["totmu"],
        pathway="totmu",
        p_slhl=P_dict["totmu"]
    )
    cf.add_production_pathway(
        scaling_factors=nmu_s,
        attenuation_length=att_dict["nmu"],
        pathway="nmu",
        p_slhl=P_dict["nmu"]
    )
    
    res = 200
    logK_vals = np.linspace(-7, -3, res)
    n_vals = np.linspace(0.25, 1.75, res)
    
    n_true_samples = 100
    
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
        out = cf.calculate_W_gridsearch(
            n_vals=n_vals,
            logK_vals=logK_vals,
            n_repeats=100,
            parallel=True,
            n_workers=5,
            smooth_fact=1e4
        )
        end = time.time()
        
        print(
            f"Gridsearch time: {true_comps[0]:.2f}, {i}, "
            f"{end-start:.2f}, {out["logK"]:.2f}, {out["n"]:.2f}"
        )"""
        """       
        fg, ax = plt.subplots(1, 3)
        ax[0].imshow(
            np.log10(np.median(out["W_raw"], axis=-1)),
            extent=[
                n_vals.min(), n_vals.max(),
                logK_vals.min(), logK_vals.max()
            ],
            origin="lower",
            aspect="auto"
        )
        ax[0].plot(out["n_line"], out["logK_line"])
        
        ax[1].plot(out["n_line"], np.log10(out["W_line"]))
        ax[2].plot(out["logK_line"], np.log10(out["W_line"]))
        plt.show()
        """     
        
        # Use diff evolve
        start = time.time()
        out = cf.calculate_W_diff_evolve(
            n_repeats=500,
            workers=5
        )
        end = time.time()
        
        lK_best.append(out["logK"])
        n_best.append(out["n"])
        
        print("Diff evlove time:", end-start, out["logK"], out["n"])

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
        

