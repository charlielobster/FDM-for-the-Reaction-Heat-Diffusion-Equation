import numpy as np

def rk4_derivative_evaluation(C, r_factor, k_dt):
    """
    Evaluates the spatial derivative RHS multiplied by dt for the MOL system.
    This combines the diffusion and reaction components cleanly.
    """
    dC_dt = np.zeros_like(C)
    
    # Vectorized central difference for diffusion + linear decay
    dC_dt[1:-1] = r_factor * (C[2:] - 2.0 * C[1:-1] + C[:-2]) - k_dt * C[1:-1]
    
    # Enforce boundary conditions (Dirichlet zero at infinity)
    dC_dt[0] = 0.0
    dC_dt[-1] = 0.0
    return dC_dt

def solver_rk4(x, dx, dt, t_final, a, k):
    """
    Fourth-Order Runge-Kutta (RK4) solver utilizing Method of Lines.
    Provides O(dt^4) temporal accuracy.
    """
    nx = len(x)
    nt = int(round(t_final / dt))
    
    # Initialize concentration array with the corrected top-hat function
    C = np.where(np.abs(x) < 1.0, 1.0, 0.0)
    C[np.isclose(np.abs(x), 1.0)] = 0.5
    
    # Pre-scale parameters to reduce internal loop arithmetic
    r_factor = a * dt / (dx**2)
    k_dt = k * dt
    
    # RK4 Time Marching Loop
    for _ in range(nt):
        # Stage 1: Initial evaluation
        k1 = rk4_derivative_evaluation(C, r_factor, k_dt)
        
        # Stage 2: First midpoint predictor
        k2 = rk4_derivative_evaluation(C + 0.5 * k1, r_factor, k_dt)
        
        # Stage 3: Second midpoint predictor
        k3 = rk4_derivative_evaluation(C + 0.5 * k2, r_factor, k_dt)
        
        # Stage 4: Full-step predictor
        k4 = rk4_derivative_evaluation(C + k3, r_factor, k_dt)
        
        # Weighted combination update step
        C += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        
    return C
