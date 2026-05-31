import pandas as pd

# -------------------------------------------------------------------------
# Save Simulation Results to a Universal CSV Checkpoint
# -------------------------------------------------------------------------

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

# Write out to a clean CSV file (dropping the index row for clean reading)
csv_filename = f"simulation_data_t_{str(t_final).replace('.', '_')}.csv"
df.to_csv(csv_filename, index=False)

print(f"Data successfully checkpointed to '{csv_filename}'.")
