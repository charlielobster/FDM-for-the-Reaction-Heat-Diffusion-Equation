import numpy as np

from analytic_solution import analytical_solution
from ftcs import solver_ftcs
from dufort_frankel import solver_dufort_frankel
from rk4 import solver_rk4
from make_pngs import make_pngs

def run_project_simulation(dx, dt, t_final=1.0, a=1.0, k=0.1):
    """
    Orchestrates the entire comparison suite for a given grid pairing.
    Calculates spatial domain, runs solvers, and computes L2 error metrics.
    """
    # Define a wide spatial domain to simulate "infinity" boundaries safely
    x_min, x_max = -10.0, 10.0
    x = np.arange(x_min, x_max + dx, dx)
    
    # 1. Compute baseline exact analytical profile
    c_exact = np.array([analytical_solution(xi, t_final, a, k) for xi in x])
    
    # 2. Run numerical solvers
    # --- FTCS ---
    # Note: Will blow up with prompt parameters! We wrap with a try/catch or runtime monitor
    try:
        c_ftcs = solver_ftcs(x, dx, dt, t_final, a, k)
        # Check if numerical overflow happened due to stability failure
        if np.any(np.isnan(c_ftcs)) or np.any(np.abs(c_ftcs) > 1e3):
            ftcs_error = "EXPLODED (Unstable)"
        else:
            ftcs_error = np.sqrt(np.mean((c_ftcs - c_exact)**2)) # RMS / L2 norm proxy
    except Exception:
        ftcs_error = "EXPLODED"
        
    # --- DuFort-Frankel ---
    c_dufort = solver_dufort_frankel(x, dx, dt, t_final, a, k)
    dufort_error = np.sqrt(np.mean((c_dufort - c_exact)**2))
    
    # --- RK4 Method of Lines ---
    c_rk4 = solver_rk4(x, dx, dt, t_final, a, k)
    rk4_error = np.sqrt(np.mean((c_rk4 - c_exact)**2))
    
    return ftcs_error, dufort_error, rk4_error

if __name__ == "__main__":
    
    # Test Run A (Prompt Specs)
    err_ftcs_A, err_df_A, err_rk4_A = run_project_simulation(dx=0.2, dt=0.1)
    
    # Test Run B (Prompt Specs)
    err_ftcs_B, err_df_B, err_rk4_B = run_project_simulation(dx=0.1, dt=0.05)
    
    # Formatting outputs for easy inspection
    print(f"\n[RUN A] Delta X = 0.2, Delta T = 0.1  (Courant r = 2.5)")
    print(f" -> FTCS Error:           {err_ftcs_A}")
    print(f" -> DuFort-Frankel Error: {err_df_A}")
    print(f" -> MOL / RK4 Error:      {err_rk4_A}")
    
    print(f"\n[RUN B] Delta X = 0.1, Delta T = 0.05 (Courant r = 5.0)")
    print(f" -> FTCS Error:           {.2f:err_ftcs_B}")
    print(f" -> DuFort-Frankel Error: {err_df_B}")
    print(f" -> MOL / RK4 Error:      {err_rk4_B}")

    make_pngs()
