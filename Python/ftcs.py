import numpy as np

def solver_ftcs(x, dx, dt, t_final, a, k):
    """
    Standard Forward-Time Central-Space (FTCS) explicit solver.
    Warning: Subject to strict Courant condition (a * dt / dx^2 <= 0.5)
    """
    nx = len(x)
    nt = int(round(t_final / dt))
    
    # Initialize concentration array with top-hat function
    C = np.where(np.abs(x) < 1.0, 1.0, 0.0)
    C[np.isclose(np.abs(x), 1.0)] = 0.5 # Boundary smoothness
    
    # Precompute the stencil constants
    r = a * dt / (dx**2)
    
    # Time-stepping loop
    for n in range(nt):
        C_next = np.copy(C)
        # Vectorized internal grid update
        C_next[1:-1] = C[1:-1] + r * (C[2:] - 2*C[1:-1] + C[:-2]) - k * dt * C[1:-1]
        
        # Dirichlet Boundary Conditions (infinity approximation)
        C_next[0] = 0.0
        C_next[-1] = 0.0
        
        C = C_next
    return C
