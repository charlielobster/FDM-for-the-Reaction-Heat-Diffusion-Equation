import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analytic_solution import analytical_solution
from fd_solvers import ftcs, rk4, dufort_frankel, btcs

# Save results to CSV
def export_to_csv(x, c_exact, c_ftcs, c_dufort, c_rk4, c_btcs, t_final):

    # Bundle all spatial nodes and matching solver outputs into a dictionary
    data_payload = {
        'Spatial_X': x,
        'Analytical_Exact': c_exact,
        'FTCS_Explicit': c_ftcs,
        'DuFort_Frankel': c_dufort,
        'MOL_RK4': c_rk4,
        'BTCS_Implicit': c_btcs
    }

    # Convert directly into a structured dataframe
    df = pd.DataFrame(data_payload)

    csv_filename = f"data\\simulation_data_t_{str(t_final).replace('.', '_')}.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Data successfully checkpointed to '{csv_filename}'.")

# This function generates a high-quality PNG comparing all solvers at the final time step.
def make_png():

    # Define grid parameters for a stable run 
    dx = 0.05
    dt = 0.0001
    t_final = 10.0
    a = 1.0
    k = 0

    x = np.arange(-10.0, 10.0 + dx, dx)

    # Gather Data from All Solvers
    c_exact  = np.array([analytical_solution(xi, t_final, a, k) for xi in x])
    c_ftcs   = ftcs(x, dx, dt, t_final, a, k)
    c_dufort = dufort_frankel(x, dx, dt, t_final, a, k)
    c_rk4    = rk4(x, dx, dt, t_final, a, k)
    c_btcs   = btcs(x, dx, dt, t_final, a, k)

    # Generate and Save the Plot
    plt.figure(figsize=(10, 6), dpi=300) 
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.plot(x, c_exact, 'k-', linewidth=2.5, label='Analytical (Exact)')
    plt.plot(x, c_ftcs, 'ro--', markersize=4, label='FTCS (Explicit)')
    plt.plot(x, c_dufort, 'bs--', markersize=4, label='DuFort-Frankel (Explicit)')
    plt.plot(x, c_rk4, 'g^--', markersize=4, label='MOL / RK4 (Explicit)')
    plt.plot(x, c_btcs, 'm*--', markersize=4, label='BTCS (Implicit)')

    plt.title(f'Concentration Profile Comparison at $t = {t_final}$', fontsize=14, fontweight='bold')
    plt.xlabel('Spatial Coordinate ($x$)', fontsize=12)
    plt.ylabel('Concentration ($C$)', fontsize=12)
    plt.xlim([-4.0, 4.0]) # Zoom in on the active diffusion region
    plt.ylim([-0.1, 1.1])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)

    output_filename = 'data\\solver_comparison_profile.png'
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()

    print(f"Success! Graph generated and saved as '{output_filename}' in your directory.")


def run_project_simulation(dx, dt, t_final=1.0, a=1.0, k=0):
    """
    Orchestrates the entire comparison suite for a given grid pairing.
    Calculates spatial domain, runs solvers, and computes L2 error metrics.
    """

    # Define a wide spatial domain to simulate "infinity" boundaries safely
    x_min, x_max = -10.0, 10.0
    x = np.arange(x_min, x_max + dx, dx)
    
    # 1. Compute baseline exact analytical profile
    c_exact = np.array([analytical_solution(xi, t_final, a, k) for xi in x])
    
    # --- FTCS ---
    c_ftcs = ftcs(x, dx, dt, t_final, a, k)
    ftcs_error = np.sqrt(np.mean((c_ftcs - c_exact)**2)) # RMS / L2 norm proxy
        
    # --- DuFort-Frankel ---
    c_dufort = dufort_frankel(x, dx, dt, t_final, a, k)
    dufort_error = np.sqrt(np.mean((c_dufort - c_exact)**2))
    
    # --- RK4 Method of Lines ---
    c_rk4 = rk4(x, dx, dt, t_final, a, k)
    rk4_error = np.sqrt(np.mean((c_rk4 - c_exact)**2))

    # --- BTCS ---    
    c_btcs = btcs(x, dx, dt, t_final, a, k)
    btcs_error = np.sqrt(np.mean((c_btcs - c_exact)**2))

  #  export_to_csv(x, c_exact, c_ftcs, c_dufort, c_rk4, c_btcs, t_final)

    return ftcs_error, dufort_error, rk4_error, btcs_error

if __name__ == "__main__":
    
    # Test Run A (Prompt Specs)
    err_ftcs_A, err_df_A, err_rk4_A, err_btcs_A = run_project_simulation(dx=0.2, dt=0.1)
    
    # Test Run B (Prompt Specs)
    err_ftcs_B, err_df_B, err_rk4_B, err_btcs_B = run_project_simulation(dx=0.1, dt=0.05)

    # Test Run C (Prompt Specs)
    err_ftcs_C, err_df_C, err_rk4_C, err_btcs_C = run_project_simulation(dx=0.05, dt=0.0005)
     
    # Test Run D (Prompt Specs)
    err_ftcs_D, err_df_D, err_rk4_D, err_btcs_D = run_project_simulation(dx=0.05, dt=0.0001)

    # Test Run E (Prompt Specs)
    err_ftcs_E, err_df_E, err_rk4_E, err_btcs_E = run_project_simulation(dx=0.05, dt=0.00001)

    # Test Run F (Prompt Specs)
    err_ftcs_F, err_df_F, err_rk4_F, err_btcs_F = run_project_simulation(dx=0.1, dt=0.005)

    # Formatting outputs for easy inspection
    print(f"[RUN A] Delta X = 0.2, Delta T = 0.1  (Courant r = 2.5)")
    print(f"FTCS Error:           {err_ftcs_A}")
    print(f"DuFort-Frankel Error: {err_df_A}")
    print(f"MOL / RK4 Error:      {err_rk4_A}")
    print(f"BTCS Error:           {err_btcs_A}")
    
    print(f"[RUN B] Delta X = 0.1, Delta T = 0.05 (Courant r = 5.0)")
    print(f"FTCS Error:           {err_ftcs_B}")
    print(f"DuFort-Frankel Error: {err_df_B}")
    print(f"MOL / RK4 Error:      {err_rk4_B}")
    print(f"BTCS Error:           {err_btcs_B}")

    print(f"[RUN C] Delta X = 0.05, Delta T = 0.0005 (Courant r = 0.2)")
    print(f"FTCS Error:           {err_ftcs_C}")
    print(f"DuFort-Frankel Error: {err_df_C}")
    print(f"MOL / RK4 Error:      {err_rk4_C}")
    print(f"BTCS Error:           {err_btcs_C}")

    print(f"[RUN D] Delta X = 0.05, Delta T = 0.0001 (Courant r = 0.04)")
    print(f"FTCS Error:           {err_ftcs_D}")
    print(f"DuFort-Frankel Error: {err_df_D}")
    print(f"MOL / RK4 Error:      {err_rk4_D}")
    print(f"BTCS Error:           {err_btcs_D}")

    print(f"[RUN E] Delta X = 0.05, Delta T = 0.00001 (Courant r = 0.004)")
    print(f"FTCS Error:           {err_ftcs_E}")
    print(f"DuFort-Frankel Error: {err_df_E}")
    print(f"MOL / RK4 Error:      {err_rk4_E}")
    print(f"BTCS Error:           {err_btcs_E}")

    print(f"[RUN F] Delta X = 0.1, Delta T = 0.005 (Courant r = 0.5)")
    print(f"FTCS Error:           {err_ftcs_F}")
    print(f"DuFort-Frankel Error: {err_df_F}")
    print(f"MOL / RK4 Error:      {err_rk4_F}")
    print(f"BTCS Error:           {err_btcs_F}")

  #  make_png()
    
