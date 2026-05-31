import numpy as np
from scipy.special import erf

def analytical_solution(x, t, a, k):
    """
    Computes the exact analytical solution for the 1-D Reaction-Diffusion equation.
    
    Parameters:
    -----------
    x : ndarray
        1-D array of spatial coordinates.
    t : float
        The time instance at which to evaluate the solution (t >= 0).
    a : float
        Diffusion coefficient (a > 0).
    k : float
        Reaction/decay coefficient (k >= 0).
        
    Returns:
    --------
    C : ndarray
        1-D array containing the pollutant concentration at each x for time t.
    """
    # Safeguard for initial condition at t = 0 to avoid division by zero
    if np.isclose(t, 0.0):
        C = np.where(np.abs(x) < 1.0, 1.0, 0.0)
        # Handle boundaries exactly matching the discontinuous transition if needed
        C[np.isclose(np.abs(x), 1.0)] = 0.5
        return C
    
    # Compute the core error function terms
    denom = 2.0 * np.sqrt(a * t)
    erf_plus = erf((1.0 + x) / denom)
    erf_minus = erf((1.0 - x) / denom)
    
    # Combine with the exponential decay factor
    C = 0.5 * np.exp(-k * t) * (erf_plus + erf_minus)
    return C
