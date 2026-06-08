import numpy as np
import time
from scipy.special import erf
from scipy.linalg import solve_banded
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class solver_state:
    """
    Container class for the simulation parameters and pre-computed analytical solution 
    for the 1-D Reaction-Diffusion equation.
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
    """
    dx: float
    dt: float
    a: float
    k: float
    t_final: float
    x_range: int
    nx: int
    nt: int
    x: np.ndarray
    t: np.ndarray
    C_exact: np.ndarray

    def __init__(self, dx, dt, a, k, t_final, x_range):
        self.dx = dx
        self.dt = dt
        self.a = a
        self.k = k
        self.t_final = t_final
        self.x_range = x_range
        self.x = np.arange(-self.x_range, self.x_range + self.dx, self.dx)
        self.t = np.arange(0, self.t_final + self.dt, self.dt)
        self.nx = len(self.x)
        self.nt = len(self.t)
        self.C_exact = np.array([self.analytical_solution(xi, self.t_final, self.a, self.k) for xi in self.x])

    def analytical_solution(self, x, t, a, k):
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

class solver_assessment(ABC):
    """
    Abstract base class for solver assessment of the 1-D Reaction-Diffusion equation.
    """
    name: str
    state = solver_state
    C: np.ndarray
    elapsed_time: float
    l2_error: float

    def __init__(self, solver_state):
        self.state = solver_state

    @abstractmethod
    def apply_method(self):
        pass

    def run_with_metrics(self):
        """
        Executes the solver and computes the L2 error against the exact solution.
        Returns the computed concentration profile and the L2 error metric.
        """
        before = time.perf_counter()
        self.apply_method()
        after = time.perf_counter()
        self.elapsed_time = after - before
        self.l2_error = np.sqrt(np.mean((self.C - self.state.C_exact)**2))
        
class ftcs_solver(solver_assessment):

    name = "FTCS"
    def apply_method(self):
        """
        Standard Forward-Time Central-Space (FTCS) explicit solver.
        Warning: Subject to strict Courant condition (a * dt / dx^2 <= 0.5)
        """

        # Initialize concentration array with top-hat function
        self.C = np.where(np.abs(self.state.x) < 1.0, 1.0, 0.0)
        self.C[np.isclose(np.abs(self.state.x), 1.0)] = 0.5 # Boundary smoothness
        
        # Precompute the stencil constants
        r = self.state.a * self.state.dt / (self.state.x[1] - self.state.x[0])**2
        
        # Time-stepping loop
        for n in range(self.state.nt):
            C_next = np.copy(self.C)
            # Vectorized internal grid update
            C_next[1:-1] = self.C[1:-1] + r * (self.C[2:] - 2*self.C[1:-1] + self.C[:-2]) - self.state.k * self.state.dt * self.C[1:-1]
            
            # Dirichlet Boundary Conditions (infinity approximation)
            C_next[0] = 0.0
            C_next[-1] = 0.0
            
            self.C = C_next

class rk4_solver(solver_assessment):

    name = "RK4"
    def rk4_derivative(self, C, r_factor, k_dt):
        """
        Evaluates the spatial derivative RHS multiplied by dt for the MOL system.
        This combines the diffusion and reaction components.
        """
        dC_dt = np.zeros_like(C)
        
        # Vectorized central difference for diffusion + linear decay
        dC_dt[1:-1] = r_factor * (C[2:] - 2.0 * C[1:-1] + C[:-2]) - k_dt * C[1:-1]
        
        # Enforce boundary conditions (Dirichlet zero at infinity)
        dC_dt[0] = 0.0
        dC_dt[-1] = 0.0
        return dC_dt

    def apply_method(self):
        """
        Fourth-Order Runge-Kutta (RK4) solver utilizing Method of Lines.
        Provides O(dt^4) temporal accuracy.
        """
        
        # Initialize concentration array with the corrected top-hat function
        C = np.where(np.abs(self.state.x) < 1.0, 1.0, 0.0)
        C[np.isclose(np.abs(self.state.x), 1.0)] = 0.5
        
        # Pre-scale parameters to reduce internal loop arithmetic
        r_factor = self.state.a * self.state.dt / (self.state.dx**2)
        k_dt = self.state.k * self.state.dt
        
        # RK4 Time Marching Loop
        for _ in range(self.state.nt):
            # Stage 1: Initial evaluation
            k1 = self.rk4_derivative(C, r_factor, k_dt)
            
            # Stage 2: First midpoint predictor
            k2 = self.rk4_derivative(C + 0.5 * k1, r_factor, k_dt)
            
            # Stage 3: Second midpoint predictor
            k3 = self.rk4_derivative(C + 0.5 * k2, r_factor, k_dt)
            
            # Stage 4: Full-step predictor
            k4 = self.rk4_derivative(C + k3, r_factor, k_dt)
            
            # Weighted combination update step
            C += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            
        self.C = C
        

class dufort_frankel_solver(solver_assessment):

    name = "DuFort-Frankel"
    def apply_method(self):
        """
        DuFort-Frankel Explicit Scheme.
        Requires a leap-frog start (uses 3 time-levels).
        """
        
        # Level n-1 (Past) and Level n (Present)
        C_past = np.where(np.abs(self.state.x) < 1.0, 1.0, 0.0)
        
        # Bootstrapping the first step (Level n) using a single stable step of FTCS
        # or the analytical solution to ensure a clean start
        C_curr = np.copy(C_past)
        r = self.state.a * self.state.dt / (self.state.x[1] - self.state.x[0])**2
        
        # Temporary small dt step just to populate the first time level cleanly
        C_curr[1:-1] = C_past[1:-1] + r * (C_past[2:] - 2*C_past[1:-1] + C_past[:-2]) - self.state.k * self.state.dt * C_past[1:-1]
        
        # Pre-calculate DuFort-Frankel scaling denominators
        denom = 1.0 + 2.0 * r + self.state.k * self.state.dt
        alpha = (1.0 - 2.0 * r - self.state.k * self.state.dt) / denom
        beta = (2.0 * r) / denom
        
        # 3-Level Time-stepping loop
        for _ in range(self.state.nt):
            C_next = np.zeros(self.state.nx)
            
            # DuFort-Frankel Leap-Frog Stencil
            C_next[1:-1] = (alpha * C_past[1:-1] + 
                            beta * (C_curr[2:] + C_curr[:-2]))
            
            # Enforce boundaries
            C_next[0] = 0.0
            C_next[-1] = 0.0
            
            # Roll the time vectors forward
            C_past = np.copy(C_curr)
            C_curr = C_next
            
        self.C = C_curr

class btcs_solver(solver_assessment):

    name = "BTCS"
    def apply_method(self):
        """
        Backward-Time Central-Space (BTCS) Implicit Solver.
        Unconditionally stable for any choices of dt and dx.
        Solves a tridiagonal matrix system at every time step.
        """
        
        # Initialize concentration array with the top-hat function
        self.C = np.where(np.abs(self.state.x) < 1.0, 1.0, 0.0)
        self.C[np.isclose(np.abs(self.state.x), 1.0)] = 0.5
        
        # Precompute mesh parameter
        r = self.state.a * self.state.dt / (self.state.x[1] - self.state.x[0])**2
        
        # Setup the Tridiagonal Matrix components for internal nodes
        # For a node i:  -r*C_{i-1}^{n+1} + (1 + 2r + k*dt)*C_i^{n+1} - r*C_{i+1}^{n+1} = C_i^n
        diag_main = (1.0 + 2.0 * r + self.state.k * self.state.dt) * np.ones(self.state.nx - 2)
        diag_sub  = -r * np.ones(self.state.nx - 2)
        diag_super = -r * np.ones(self.state.nx - 2)
        
        # Format matrix for scipy's solve_banded (3 rows: upper diag, main diag, lower diag)
        # Row 0: Upper diagonal (shifted right)
        # Row 1: Main diagonal
        # Row 2: Lower diagonal (shifted left)
        A_banded = np.zeros((3, self.state.nx - 2))
        A_banded[0, 1:] = diag_super[:-1]
        A_banded[1, :]  = diag_main
        A_banded[2, :-1] = diag_sub[1:]
        
        # Time-marching loop
        for _ in range(self.state.nt):
            # Establish the Right-Hand Side vector (B) using current time step values
            B = np.copy(self.C[1:-1])
            
            # Inject Dirichlet Boundary Conditions into the boundary vector endpoints
            # (Since boundaries are 0 at infinity, these remain 0, but good for structural layout)
            B[0] += r * 0.0   # Boundary at x_min
            B[-1] += r * 0.0  # Boundary at x_max
            
            # Solve the simultaneous linear system for the inner grid nodes
            self.C[1:-1] = solve_banded((1, 1), A_banded, B)
            
            # Enforce exact boundary constants
            self.C[0] = 0.0
            self.C[-1] = 0.0
