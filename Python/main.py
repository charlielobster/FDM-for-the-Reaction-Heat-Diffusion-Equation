import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fd_solvers import solver_state, solver_assessment, ftcs_solver, dufort_frankel_solver, rk4_solver, btcs_solver

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
def group_graph_snapshot(state, c_exact, c_ftcs, c_dufort, c_rk4, c_btcs, x):
    """
        Generates a PNG comparing all solvers at the final time step.
    """
    # 2. Define the textbox style properties
    box_properties = dict(
        boxstyle='round,pad=0.5',   # Shapes: 'square', 'round', etc.
        facecolor='wheat',          # Background color
        edgecolor='darkgoldenrod',  # Border color
        alpha=0.85                  # Transparency (0 = clear, 1 = opaque)
    )

    # Generate and Save the Plot
    plt.figure(figsize=(10, 6), dpi=300) 
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]

    plt.plot(x, c_exact, 'k-', linewidth=2.5, label='Analytical (Exact)')
    plt.plot(x, c_ftcs, 'ro--', markersize=4, label='FTCS (Explicit)')
    plt.plot(x, c_dufort, 'bs--', markersize=4, label='DuFort-Frankel (Explicit)')
    plt.plot(x, c_rk4, 'g^--', markersize=4, label='MOL / RK4 (Explicit)')
    plt.plot(x, c_btcs, 'm*--', markersize=4, label='BTCS (Implicit)')

    plt.title(f'Concentration Profile Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Spatial Coordinate ($x$)', fontsize=12)
    plt.ylabel('Concentration ($C$)', fontsize=12)
    plt.xlim([-4.0, 4.0]) # Zoom in on the active diffusion region
    plt.ylim([-0.1, 1.5])
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)

    # 3. Define the textbox string layout (supports multiline and LaTeX)
    text_content = f"dx: {state.dx}\ndt: {state.dt}\na: {state.a}\nk: {state.k}\nt_final: {state.t_final}"

    # 4. Add the textbox overlay to the upper-left corner
    plt.text(
        -0.5, 1.4,                 # X, Y coordinates (relative to axes)
        text_content, 
        fontsize=11, 
        verticalalignment='top',    # Anchors the top of the box to Y=0.95
        horizontalalignment='left', # Anchors the left of the box to X=0.05
        bbox=box_properties         # Applies the text box styling
    )

    filename = f"data\\assessment_dx_{state.dx}_dt_{state.dt}_a_{state.a}_k_{state.k}_tfinal_{state.t_final}.png"

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    print(f"Graph generated and saved as '{filename}'.")


def run_group_assessment(dx, dt, a, k, t_final, x_range):
    """
    Orchestrates the entire comparison suite for a given grid pairing.
    Calculates spatial domain, runs solvers, and computes L2 error metrics.
    """    
    state = solver_state(dx=dx, dt=dt, a=a, k=k, t_final=t_final, x_range=x_range)
    solvers = [ftcs_solver(state), dufort_frankel_solver(state), rk4_solver(state), btcs_solver(state)]
    for solver in solvers:
        solver.run_with_metrics()
        print(f"{solver.name} - L2 Error: {solver.l2_error:.6e}")
        print(f"{solver.name} - Elapsed Time: {solver.elapsed_time:.4f} seconds")
    
    group_graph_snapshot(state, state.C_exact, solvers[0].C, solvers[1].C, solvers[2].C, solvers[3].C, state.x)
    return solvers

if __name__ == "__main__":

    a_s = [0.1, 1.0, 1.01]
    k_s = [0.0, 0.1, 0.25, 0.5]
    dx_s = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]
    dt_s = [0.1, 0.05, 0.025, 0.021, 0.05, 0.025, 0.0125, 0.0051]
    t_finals = [1.0, 2.5, 5.0, 10.0]
    x_range = 10.0

    for i in [0,1,2,3,4,5,6,7]:         
        print(f"Running Simulation with dx = {dx_s[i]}, dt = {dt_s[i]}, a = 1.0, k = 0.1, t_final = 1.0, x_range = {x_range}")
        solvers = run_group_assessment(dx=dx_s[i], dt=dt_s[i], a=1.0, k=0.1, t_final=1.0, x_range=x_range)
        print(f"Errors - FTCS: {solvers[0].l2_error:.6e}, DuFort-Frankel: {solvers[1].l2_error:.6e}, RK4: {solvers[2].l2_error:.6e}, BTCS: {solvers[3].l2_error:.6e}\n")   

    dx = 0.1
    dt = 0.005

    for a in a_s:
        for k in k_s:
            for t_final in t_finals:
                print(f"Running Simulation with dx = {dx}, dt = {dt}, a = {a}, k = {k}, t_final = {t_final}, x_range = {x_range}")
                solvers = run_group_assessment(dx=dx, dt=dt, a=a, k=k, t_final=t_final, x_range=x_range)
                print(f"Errors - FTCS: {solvers[0].l2_error:.6e}, DuFort-Frankel: {solvers[1].l2_error:.6e}, RK4: {solvers[2].l2_error:.6e}, BTCS: {solvers[3].l2_error:.6e}\n")

