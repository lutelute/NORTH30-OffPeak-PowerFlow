% Complete Inverse Problem Validation for NORTH30 Load Allocation
% This script demonstrates true inverse problem solving with clear validation

clear all;
clc;

%% STEP 1: Problem Setup and Data Preparation
fprintf('=== NORTH30 INVERSE PROBLEM: LOAD ALLOCATION ===\n');
fprintf('Objective: Estimate load distribution from partial measurements\n\n');

% Load original case (this represents the "true" system state)
mpc_true = north30_matpower();
bus_data = readtable('NORTH30_OffPeak_Bus.csv');
branch_data = readtable('NORTH30_OffPeak_Branch.csv');

% Extract true load distribution (what we want to recover)
true_loads = mpc_true.bus(:, 3);  % True Pd values
load_buses = find(true_loads ~= 0);
n_load_buses = length(load_buses);

fprintf('True System State:\n');
fprintf('- Total number of buses: %d\n', size(mpc_true.bus, 1));
fprintf('- Buses with loads: %d\n', n_load_buses);
fprintf('- Total true load: %.1f MW\n', sum(true_loads));
fprintf('- Total generation: %.1f MW\n', sum(mpc_true.gen(:,2)));

%% STEP 2: Create Realistic Measurement Scenario
% Simulate what would be available in practice

% Available measurements (realistic scenario):
% 1. Voltage measurements at ALL buses (common with PMUs)
observed_voltages = bus_data.Vm;
observed_angles = bus_data.Va;

% 2. Power flow measurements on CRITICAL transmission lines only
critical_line_indices = [1, 5, 8, 12, 15, 25, 32, 44, 50, 55]; % Major transmission paths
critical_flows = branch_data.P_f(critical_line_indices);
critical_from = branch_data.fbus(critical_line_indices);
critical_to = branch_data.tbus(critical_line_indices);

% 3. Generation dispatch (assumed known from energy management system)
known_generation = mpc_true.gen(:,2);

% 4. Add realistic measurement noise
rng(42); % For reproducible results
voltage_noise = 0.002; % ±0.2% voltage measurement error
flow_noise = 2.0;      % ±2 MW flow measurement error

observed_voltages_noisy = observed_voltages + voltage_noise * randn(size(observed_voltages));
observed_flows_noisy = critical_flows + flow_noise * randn(size(critical_flows));

fprintf('\nMeasurement Setup (Realistic Scenario):\n');
fprintf('- Voltage measurements: %d buses (±%.1f%% noise)\n', length(observed_voltages), voltage_noise*100);
fprintf('- Flow measurements: %d critical lines (±%.1f MW noise)\n', length(critical_flows), flow_noise);
fprintf('- Known generation: %d units\n', size(mpc_true.gen,1));

%% STEP 3: Formulate Inverse Problem
% Unknown: Load values at load buses
% Known: Generation, partial voltage/flow measurements

fprintf('\nInverse Problem Formulation:\n');
fprintf('Unknowns: Load values at %d buses\n', n_load_buses);
fprintf('Constraints: Power balance + physical limits\n');
fprintf('Observations: %d voltage + %d flow measurements\n', ...
        length(observed_voltages), length(critical_flows));

% Create "unknown load" case (starting point for inverse problem)
mpc_unknown = mpc_true;
mpc_unknown.bus(:, 3) = 0; % Set all loads to zero (unknown)

%% STEP 4: Solve Inverse Problem using Optimization
fprintf('\nSolving inverse problem...\n');

% Decision variables: loads at load buses
x0 = 50 * ones(n_load_buses, 1); % Initial guess: 50 MW at each load bus

% Bounds: reasonable load limits
lb = zeros(n_load_buses, 1);      % Non-negative loads
ub = 500 * ones(n_load_buses, 1); % Maximum 500 MW per bus

% Power balance constraint
total_generation = sum(known_generation);
estimated_losses = 50; % Typical loss estimate
target_total_load = total_generation - estimated_losses;

Aeq = ones(1, n_load_buses);  % Sum of all loads
beq = target_total_load;      % Should equal generation - losses

% Objective function: minimize measurement residuals
objective = @(loads) inverse_objective_function(loads, load_buses, mpc_unknown, ...
                                               observed_voltages_noisy, observed_angles, ...
                                               observed_flows_noisy, critical_from, critical_to);

% Solve optimization problem
options = optimoptions('fmincon', 'Display', 'iter-detailed', 'MaxIterations', 200, ...
                      'OptimalityTolerance', 1e-6, 'StepTolerance', 1e-10);

[estimated_loads, fval, exitflag, output] = fmincon(objective, x0, [], [], ...
                                                   Aeq, beq, lb, ub, [], options);

%% STEP 5: Validation and Comparison
fprintf('\n=== VALIDATION RESULTS ===\n');

if exitflag > 0
    fprintf('✓ Optimization converged successfully\n');
    fprintf('Final objective value: %.6f\n', fval);
    fprintf('Iterations: %d\n', output.iterations);
else
    fprintf('⚠ Optimization warning (exitflag: %d)\n', exitflag);
end

% Create estimated system case
mpc_estimated = mpc_unknown;
for i = 1:n_load_buses
    bus_idx = find(mpc_estimated.bus(:,1) == load_buses(i));
    mpc_estimated.bus(bus_idx, 3) = estimated_loads(i);
end

% Run power flow with estimated loads
results_estimated = rundcpf(mpc_estimated);
results_true = rundcpf(mpc_true);

%% STEP 6: Detailed Comparison and Error Analysis
fprintf('\n=== LOAD ESTIMATION ACCURACY ===\n');
fprintf('Bus | True Load | Estimated | Absolute Error | Relative Error\n');
fprintf('----|-----------|-----------|----------------|---------------\n');

load_errors = [];
for i = 1:n_load_buses
    bus_num = load_buses(i);
    bus_idx = find(mpc_true.bus(:,1) == bus_num);
    
    true_load = mpc_true.bus(bus_idx, 3);
    estimated_load = estimated_loads(i);
    abs_error = abs(estimated_load - true_load);
    
    if true_load > 0.1  % Avoid division by very small numbers
        rel_error = (abs_error / true_load) * 100;
        load_errors = [load_errors; abs_error];
    else
        rel_error = 0;
    end
    
    fprintf('%3d | %9.1f | %9.1f | %14.1f | %13.1f%%\n', ...
            bus_num, true_load, estimated_load, abs_error, rel_error);
end

% Load estimation statistics
load_rmse = sqrt(mean(load_errors.^2));
load_mae = mean(load_errors);
max_load_error = max(load_errors);

fprintf('\nLoad Estimation Statistics:\n');
fprintf('RMSE: %.2f MW\n', load_rmse);
fprintf('MAE:  %.2f MW\n', load_mae);
fprintf('Max Error: %.2f MW\n', max_load_error);
fprintf('Total Load - True: %.1f MW, Estimated: %.1f MW\n', ...
        sum(true_loads), sum(estimated_loads));

%% STEP 7: Power Flow Validation
fprintf('\n=== POWER FLOW VALIDATION ===\n');

if results_estimated.success && results_true.success
    fprintf('✓ Both power flows converged\n');
    
    % Compare critical line flows
    fprintf('\nCritical Line Flow Comparison:\n');
    fprintf('Line   | True Flow | Estimated | Error   | Error%%\n');
    fprintf('-------|-----------|-----------|---------|--------\n');
    
    flow_errors = [];
    for i = 1:length(critical_line_indices)
        line_idx = critical_line_indices(i);
        from_bus = critical_from(i);
        to_bus = critical_to(i);
        
        % Find flows in both cases
        true_idx = find(results_true.branch(:,1) == from_bus & results_true.branch(:,2) == to_bus);
        est_idx = find(results_estimated.branch(:,1) == from_bus & results_estimated.branch(:,2) == to_bus);
        
        if ~isempty(true_idx) && ~isempty(est_idx)
            true_flow = results_true.branch(true_idx(1), 14);
            est_flow = results_estimated.branch(est_idx(1), 14);
            error = est_flow - true_flow;
            
            if abs(true_flow) > 0.1
                error_pct = (error / true_flow) * 100;
            else
                error_pct = 0;
            end
            
            flow_errors = [flow_errors; error];
            
            fprintf('%2d→%-2d  | %9.1f | %9.1f | %7.1f | %6.1f%%\n', ...
                    from_bus, to_bus, true_flow, est_flow, error, error_pct);
        end
    end
    
    % Flow estimation statistics
    flow_rmse = sqrt(mean(flow_errors.^2));
    flow_mae = mean(abs(flow_errors));
    
    fprintf('\nFlow Estimation Statistics:\n');
    fprintf('RMSE: %.2f MW\n', flow_rmse);
    fprintf('MAE:  %.2f MW\n', flow_mae);
    
else
    fprintf('✗ Power flow convergence issues\n');
    flow_rmse = NaN;
    flow_mae = NaN;
end

%% STEP 8: Visualization
figure('Position', [100, 100, 1400, 1000]);

% Subplot 1: Load comparison
subplot(2,3,1);
true_load_values = [];
est_load_values = [];
bus_numbers = [];

for i = 1:n_load_buses
    bus_num = load_buses(i);
    bus_idx = find(mpc_true.bus(:,1) == bus_num);
    true_load_values = [true_load_values; mpc_true.bus(bus_idx, 3)];
    est_load_values = [est_load_values; estimated_loads(i)];
    bus_numbers = [bus_numbers; bus_num];
end

bar([true_load_values, est_load_values]);
xlabel('Load Bus Index');
ylabel('Load (MW)');
title('Load Comparison: True vs Estimated');
legend('True Load', 'Estimated Load', 'Location', 'best');
grid on;

% Subplot 2: Load scatter plot
subplot(2,3,2);
scatter(true_load_values, est_load_values, 60, 'filled');
hold on;
min_val = min([true_load_values; est_load_values]);
max_val = max([true_load_values; est_load_values]);
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
xlabel('True Load (MW)');
ylabel('Estimated Load (MW)');
title('Load Estimation Accuracy');
grid on;
axis equal;

% Add correlation info
corr_coef = corr(true_load_values, est_load_values);
text(0.05, 0.95, sprintf('R² = %.3f', corr_coef^2), 'Units', 'normalized', ...
     'BackgroundColor', 'white');

% Subplot 3: Load estimation errors
subplot(2,3,3);
load_error_by_bus = est_load_values - true_load_values;
bar(load_error_by_bus);
xlabel('Load Bus Index');
ylabel('Error (MW)');
title('Load Estimation Errors');
grid on;
yline(0, 'k--');

% Subplot 4: Flow comparison
subplot(2,3,4);
if exist('flow_errors', 'var') && ~isempty(flow_errors)
    bar(flow_errors);
    xlabel('Critical Line Index');
    ylabel('Flow Error (MW)');
    title('Critical Line Flow Errors');
    grid on;
    yline(0, 'k--');
end

% Subplot 5: Error distribution
subplot(2,3,5);
histogram(load_error_by_bus, 10);
xlabel('Load Error (MW)');
ylabel('Frequency');
title('Load Error Distribution');
grid on;

% Subplot 6: Summary statistics
subplot(2,3,6);
axis off;
summary_text = {
    'INVERSE PROBLEM RESULTS';
    '========================';
    sprintf('Load RMSE: %.2f MW', load_rmse);
    sprintf('Load MAE: %.2f MW', load_mae);
    sprintf('Load Correlation: %.3f', corr_coef);
    '';
    sprintf('Flow RMSE: %.2f MW', flow_rmse);
    sprintf('Flow MAE: %.2f MW', flow_mae);
    '';
    sprintf('Total Load Error: %.1f MW', sum(est_load_values) - sum(true_load_values));
    sprintf('Relative Total Error: %.2f%%', ...
            abs(sum(est_load_values) - sum(true_load_values))/sum(true_load_values)*100);
};

text(0.1, 0.9, summary_text, 'FontSize', 12, 'FontName', 'Courier', ...
     'VerticalAlignment', 'top');

sgtitle('NORTH30 Inverse Problem: Load Allocation Validation', 'FontSize', 14);

%% STEP 9: Save Results
save('inverse_problem_validation_results.mat', 'true_load_values', 'est_load_values', ...
     'load_rmse', 'load_mae', 'flow_rmse', 'flow_mae', 'corr_coef', ...
     'mpc_true', 'mpc_estimated', 'results_true', 'results_estimated');

% Export figure
print(gcf, 'inverse_problem_validation.png', '-dpng', '-r300');

fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('Inverse problem type: Load allocation from partial measurements\n');
fprintf('Method: Constrained nonlinear optimization\n');
fprintf('Success: %s\n', exitflag > 0 ? 'YES' : 'NO');
fprintf('Load estimation RMSE: %.2f MW (%.1f%% of average load)\n', ...
        load_rmse, (load_rmse/mean(true_load_values))*100);
fprintf('Flow prediction RMSE: %.2f MW\n', flow_rmse);
fprintf('Total load recovery accuracy: %.2f%%\n', ...
        (1 - abs(sum(est_load_values) - sum(true_load_values))/sum(true_load_values))*100);

fprintf('\nFiles saved:\n');
fprintf('- inverse_problem_validation_results.mat\n');
fprintf('- inverse_problem_validation.png\n');

end

%% Objective Function for Inverse Problem
function obj_val = inverse_objective_function(loads, load_buses, mpc_base, ...
                                             obs_voltages, obs_angles, ...
                                             obs_flows, flow_from, flow_to)
    
    % Update case with current load estimates
    mpc_temp = mpc_base;
    for i = 1:length(load_buses)
        bus_idx = find(mpc_temp.bus(:,1) == load_buses(i));
        if ~isempty(bus_idx)
            mpc_temp.bus(bus_idx, 3) = loads(i);
        end
    end
    
    % Run DC power flow
    results = rundcpf(mpc_temp);
    
    if ~results.success
        obj_val = 1e8; % Large penalty for non-convergent solutions
        return;
    end
    
    % Calculate weighted sum of squared residuals
    obj_val = 0;
    
    % Voltage angle residuals (weight = 1)
    calc_angles = results.bus(:, 9);
    angle_residuals = calc_angles - obs_angles;
    obj_val = obj_val + sum(angle_residuals.^2);
    
    % Flow residuals (weight = 100, more important for matching)
    for i = 1:length(obs_flows)
        from_bus = flow_from(i);
        to_bus = flow_to(i);
        
        line_idx = find(results.branch(:,1) == from_bus & results.branch(:,2) == to_bus);
        
        if ~isempty(line_idx)
            calc_flow = results.branch(line_idx(1), 14);
            flow_residual = calc_flow - obs_flows(i);
            obj_val = obj_val + 100 * flow_residual^2;
        end
    end
end