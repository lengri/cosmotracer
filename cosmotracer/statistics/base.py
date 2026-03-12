import numpy as np

def wasserstein2_distance(u_values, v_values, u_weights=None, v_weights=None):
    """
    Compute the 1-D Wasserstein-2 distance between two empirical distributions.

    The Wasserstein-2 distance (also called the Earth Mover's Distance in the
    quadratic case) measures the minimal transport cost required to transform
    one distribution into another. In one dimension this can be computed by
    integrating the squared difference between the two quantile functions.

    Parameters
    ----------
    u_values : array_like
        Sample values of the first distribution.
    v_values : array_like
        Sample values of the second distribution.
    u_weights : array_like, optional
        Weights associated with `u_values`. If None, uniform weights are used.
    v_weights : array_like, optional
        Weights associated with `v_values`. If None, uniform weights are used.

    Returns
    -------
    float
        The Wasserstein-2 distance between the two distributions.

    Notes
    -----
    The computation proceeds by:

    1. Sorting both distributions.
    2. Constructing their cumulative distribution functions (CDFs).
    3. Sampling both inverse CDFs (quantile functions) on a shared quantile grid.
    4. Integrating the squared difference between the quantile functions.

    This implementation assumes one-dimensional empirical distributions.
    """

    u_values = np.asarray(u_values)
    v_values = np.asarray(v_values)

    # sort values
    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    u_values = u_values[u_sorter]
    v_values = v_values[v_sorter]

    # weights
    if u_weights is None:
        u_weights = np.full(u_values.size, 1.0 / u_values.size)
    else:
        u_weights = np.asarray(u_weights)[u_sorter]
        u_weights = u_weights / u_weights.sum()

    if v_weights is None:
        v_weights = np.full(v_values.size, 1.0 / v_values.size)
    else:
        v_weights = np.asarray(v_weights)[v_sorter]
        v_weights = v_weights / v_weights.sum()

    # cumulative distribution functions
    u_cdf = np.concatenate(([0.0], np.cumsum(u_weights)))
    v_cdf = np.concatenate(([0.0], np.cumsum(v_weights)))

    # shared quantile grid
    q = np.unique(np.concatenate((u_cdf, v_cdf)))

    # inverse CDFs (quantile functions)
    u_q = np.interp(q, u_cdf, np.concatenate(([u_values[0]], u_values)))
    v_q = np.interp(q, v_cdf, np.concatenate(([v_values[0]], v_values)))

    # integrate squared difference
    dq = np.diff(q)
    diff = u_q[:-1] - v_q[:-1]

    return np.sqrt(np.sum(diff * diff * dq))


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
    values : array
        The array of values to sample from. Usually concentrations.
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