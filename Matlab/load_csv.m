% MATLAB script to import and instantly inspect the python data payload
data = readtable('simulation_data_t_1_0.csv');

% Access any column instantly using standard dot notation
plot(data.Spatial_X, data.Analytical_Exact, 'k-', 'LineWidth', 2);
hold on;
plot(data.Spatial_X, data.MOL_RK4, 'g--');
xlabel('Spatial Coordinate (x)');
ylabel('Concentration (C)');
grid on;
