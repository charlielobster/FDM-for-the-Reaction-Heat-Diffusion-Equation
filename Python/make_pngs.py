import os
import numpy as np
import matplotlib.pyplot as plt

# Define your grid parameters for a stable run to show the curves working
# (e.g., dx = 0.2, dt = 0.005 to keep FTCS stable at r = 0.125)
dx = 0.2
dt = 0.005
t_final = 1.0
a = 1.0
k = 0.1

x = np.arange(-10.0, 10.0 + dx, dx)

# -------------------------------------------------------------------------
# 1. Gather Data from All Solvers
# -------------------------------------------------------------------------
# (Assuming your functions: analytical_solution, solver_ftcs, solver_dufort_frankel, solver_rk4)
c_exact  = np.array([analytical_solution(xi, t_final, a, k) for xi in x])
c_ftcs   = solver_ftcs(x, dx, dt, t_final, a, k)
c_dufort = solver_dufort_frankel(x, dx, dt, t_final, a, k)
c_rk4    = solver_rk4(x, dx, dt, t_final, a, k)
c_btcs   = solver_btcs(x, dx, dt, t_final, a, k)

# -------------------------------------------------------------------------
# 2. Generate and Save the Plot
# -------------------------------------------------------------------------
# Set up a professional, clean plot frame
plt.figure(figsize=(10, 6), dpi=300) 

# Plot the exact analytical baseline as a solid background line
plt.plot(x, c_exact, 'k-', linewidth=2.5, label='Analytical (Exact)')

# Overlay numerical solution markers/dashed lines
plt.plot(x, c_ftcs, 'ro--', markersize=4, label='FTCS (Explicit)')
plt.plot(x, c_dufort, 'bs--', markersize=4, label='DuFort-Frankel (Explicit)')
plt.plot(x, c_rk4, 'g^--', markersize=4, label='MOL / RK4 (Explicit)')
plt.plot(x, c_btcs, 'm*--', markersize=4, label='BTCS (Implicit)')

# Format labels and aesthetics using standard LaTeX math rendering
plt.title(f'Concentration Profile Comparison at $t = {t_final}$', fontsize=14, fontweight='bold')
plt.xlabel('Spatial Coordinate ($x$)', fontsize=12)
plt.ylabel('Concentration ($C$)', fontsize=12)
plt.xlim([-4.0, 4.0]) # Zoom in on the active diffusion region
plt.ylim([-0.1, 1.1])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)

# Save directly to a high-resolution PNG image file
output_filename = 'solver_comparison_profile.png'
plt.savefig(output_filename, bbox_inches='tight')
plt.close()

print(f"Success! Graph generated and saved as '{output_filename}' in your directory.")
