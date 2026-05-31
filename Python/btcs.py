import numpy as np
from scipy.linalg import solve_banded

def solver_btcs(x, dx, dt, t_final, a, k):
    """
    Backward-Time Central-Space (BTCS) Implicit Solver.
    Unconditionally stable for any choices of dt and dx.
    Solves a tridiagonal matrix system at every time step.
    """
    nx = len(x)
    nt = int(round(t_final / dt))
    
    # Initialize concentration array with the top-hat function
    C = np.where(np.abs(x) < 1.0, 1.0, 0.0)
    C[np.isclose(np.abs(x), 1.0)] = 0.5
    
    # Precompute mesh parameter
    r = a * dt / (dx**2)
    
    # Setup the Tridiagonal Matrix components for internal nodes
    # For a node i:  -r*C_{i-1}^{n+1} + (1 + 2r + k*dt)*C_i^{n+1} - r*C_{i+1}^{n+1} = C_i^n
    diag_main = (1.0 + 2.0 * r + k * dt) * np.ones(nx - 2)
    diag_sub  = -r * np.ones(nx - 2)
    diag_super = -r * np.ones(nx - 2)
    
    # Format matrix for scipy's solve_banded (3 rows: upper diag, main diag, lower diag)
    # Row 0: Upper diagonal (shifted right)
    # Row 1: Main diagonal
    # Row 2: Lower diagonal (shifted left)
    A_banded = np.zeros((3, nx - 2))
    A_banded[0, 1:] = diag_super[:-1]
    A_banded[1, :]  = diag_main
    A_banded[2, :-1] = diag_sub[1:]
    
    # Time-marching loop
    for n in range(nt):
        # Establish the Right-Hand Side vector (B) using current time step values
        B = np.copy(C[1:-1])
        
        # Inject Dirichlet Boundary Conditions into the boundary vector endpoints
        # (Since boundaries are 0 at infinity, these remain 0, but good for structural layout)
        B[0] += r * 0.0   # Boundary at x_min
        B[-1] += r * 0.0  # Boundary at x_max
        
        # Solve the simultaneous linear system for the inner grid nodes
        C[1:-1] = solve_banded((1, 1), A_banded, B)
        
        # Enforce exact boundary constants
        C[0] = 0.0
        C[-1] = 0.0
        
    return C
