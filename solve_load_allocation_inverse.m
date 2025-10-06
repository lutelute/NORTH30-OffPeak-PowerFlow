% Solve inverse problem: Load allocation from observed voltage and partial flow data
% This demonstrates true inverse problem for load distribution

clear all;
clc;

% Load the base case
mpc = north30_matpower();

% Read original data for "observations"
branch_data = readtable('NORTH30_OffPeak_Branch.csv');
bus_data = readtable('NORTH30_OffPeak_Bus.csv');

fprintf('NORTH30 Inverse Problem: Load Allocation\n');
fprintf('========================================\n');

%% Step 1: Create "observed" data (simulate partial measurements)
% In real applications, these would be actual measurements

% Observed voltages (assume we have measurements at all buses)
observed_voltages = bus_data.Vm;
observed_angles = bus_data.Va;

% Observed flows (assume we have measurements on critical lines only)
critical_lines = [1, 5, 8, 13, 15, 25, 32, 44]; % Key transmission lines
observed_flows = branch_data.P_f(critical_lines);
observed_from_bus = branch_data.fbus(critical_lines);
observed_to_bus = branch_data.tbus(critical_lines);

% Known generation (assume generation dispatch is known)
known_gen_buses = mpc.gen(:,1);
known_gen_values = mpc.gen(:,2);

fprintf('Setup:\n');
fprintf('- Observed voltages at %d buses\n', length(observed_voltages));
fprintf('- Observed flows on %d critical lines\n', length(observed_flows));
fprintf('- Known generation at %d buses\n', length(known_gen_values));

%% Step 2: Define inverse problem
% Unknowns: Load values at load buses
load_buses = find(mpc.bus(:,3) ~= 0 | mpc.bus(:,4) ~= 0);
n_loads = length(load_buses);

fprintf('- Unknown loads at %d buses\n', n_loads);

% Initial guess for loads (start with zero)
x0 = zeros(n_loads, 1);

% Define objective function: minimize difference between calculated and observed
objective_function = @(loads) calculate_objective(loads, load_buses, mpc, ...
                                                 observed_voltages, observed_angles, ...
                                                 observed_flows, observed_from_bus, observed_to_bus);

%% Step 3: Set up constraints
% Load constraints (reasonable bounds)
lb = -500 * ones(n_loads, 1);  % Lower bound: -500 MW (generation)
ub = 500 * ones(n_loads, 1);   % Upper bound: 500 MW (load)

% Power balance constraint (total load ≈ total generation - losses)
total_gen = sum(known_gen_values);
estimated_losses = 50; % Estimate 50 MW losses
target_total_load = total_gen - estimated_losses;

% Linear constraint: sum of loads = target total load
Aeq = ones(1, n_loads);
beq = target_total_load;

fprintf('- Total generation: %.1f MW\n', total_gen);
fprintf('- Target total load: %.1f MW\n', target_total_load);

%% Step 4: Solve inverse problem
fprintf('\nSolving inverse problem...\n');

options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 100);

try
    [optimal_loads, fval, exitflag] = fmincon(objective_function, x0, [], [], ...
                                              Aeq, beq, lb, ub, [], options);
    
    if exitflag > 0
        fprintf('Inverse problem solved successfully!\n');
        fprintf('Final objective value: %.6f\n', fval);
    else
        fprintf('Warning: Optimization may not have converged properly.\n');
    end
    
catch ME
    fprintf('Error in optimization: %s\n', ME.message);
    fprintf('Falling back to least squares solution...\n');
    
    % Simple least squares solution for power balance
    optimal_loads = (target_total_load / n_loads) * ones(n_loads, 1);
end

%% Step 5: Validate solution
fprintf('\nValidating solution...\n');

% Update case with estimated loads
mpc_estimated = mpc;
for i = 1:n_loads
    bus_idx = find(mpc_estimated.bus(:,1) == load_buses(i));
    if ~isempty(bus_idx)
        mpc_estimated.bus(bus_idx, 3) = optimal_loads(i); % Set Pd
    end
end

% Run power flow with estimated loads
results_estimated = rundcpf(mpc_estimated);

if results_estimated.success
    fprintf('Power flow with estimated loads converged!\n');
    
    % Compare results
    fprintf('\nComparison of Load Estimates:\n');
    fprintf('Bus  | True Load | Estimated | Error   | Error%%\n');
    fprintf('-----|-----------|-----------|---------|--------\n');
    
    total_error = 0;
    valid_comparisons = 0;
    
    for i = 1:n_loads
        bus_num = load_buses(i);
        bus_idx = find(mpc.bus(:,1) == bus_num);
        
        if ~isempty(bus_idx)
            true_load = mpc.bus(bus_idx, 3);
            estimated_load = optimal_loads(i);
            error = estimated_load - true_load;
            
            if true_load ~= 0
                error_percent = (error / true_load) * 100;
                total_error = total_error + error^2;
                valid_comparisons = valid_comparisons + 1;
            else
                error_percent = 0;
            end
            
            fprintf('%4d | %9.1f | %9.1f | %7.1f | %6.1f%%\n', ...
                    bus_num, true_load, estimated_load, error, error_percent);
        end
    end
    
    % Calculate RMSE for load estimation
    if valid_comparisons > 0
        rmse_loads = sqrt(total_error / valid_comparisons);
        fprintf('\nLoad Estimation RMSE: %.2f MW\n', rmse_loads);
    end
    
    % Compare flow estimations
    fprintf('\nFlow Comparison on Critical Lines:\n');
    fprintf('Line    | Observed | Calculated | Error\n');
    fprintf('--------|----------|------------|------\n');
    
    flow_errors = [];
    for i = 1:length(critical_lines)
        line_idx = critical_lines(i);
        from_bus = observed_from_bus(i);
        to_bus = observed_to_bus(i);
        
        % Find corresponding line in results
        result_idx = find(results_estimated.branch(:,1) == from_bus & ...
                         results_estimated.branch(:,2) == to_bus);
        
        if ~isempty(result_idx)
            observed_flow = observed_flows(i);
            calculated_flow = results_estimated.branch(result_idx(1), 14);
            error = calculated_flow - observed_flow;
            flow_errors = [flow_errors; error];
            
            fprintf('%2d→%-2d   | %8.1f | %10.1f | %5.1f\n', ...
                    from_bus, to_bus, observed_flow, calculated_flow, error);
        end
    end
    
    if ~isempty(flow_errors)
        rmse_flows = sqrt(mean(flow_errors.^2));
        fprintf('\nFlow Estimation RMSE: %.2f MW\n', rmse_flows);
    end
    
else
    fprintf('Power flow with estimated loads did not converge.\n');
end

%% Step 6: Summary
fprintf('\n=== INVERSE PROBLEM SOLUTION SUMMARY ===\n');
fprintf('Problem type: Load allocation from partial measurements\n');
fprintf('Method: Constrained optimization (fmincon)\n');
fprintf('Unknowns: %d load values\n', n_loads);
fprintf('Constraints: Power balance + physical bounds\n');
fprintf('Observations: Voltages + critical line flows\n');

if exist('rmse_loads', 'var')
    fprintf('Load estimation accuracy: RMSE = %.2f MW\n', rmse_loads);
end
if exist('rmse_flows', 'var')
    fprintf('Flow prediction accuracy: RMSE = %.2f MW\n', rmse_flows);
end

% Save results
save('inverse_problem_results.mat', 'optimal_loads', 'load_buses', ...
     'observed_flows', 'results_estimated');

fprintf('\nResults saved to: inverse_problem_results.mat\n');

%% Objective function definition
function obj = calculate_objective(loads, load_buses, mpc, obs_V, obs_angles, ...
                                  obs_flows, obs_from, obs_to)
    % Update load values in the case
    mpc_temp = mpc;
    for i = 1:length(load_buses)
        bus_idx = find(mpc_temp.bus(:,1) == load_buses(i));
        if ~isempty(bus_idx)
            mpc_temp.bus(bus_idx, 3) = loads(i);
        end
    end
    
    % Run DC power flow
    results = rundcpf(mpc_temp);
    
    if ~results.success
        obj = 1e6; % Large penalty for non-convergent cases
        return;
    end
    
    % Calculate objective: weighted sum of squared errors
    obj = 0;
    
    % Voltage angle errors (weight = 1)
    for i = 1:length(obs_angles)
        calc_angle = results.bus(i, 9);
        obs_angle = obs_angles(i);
        obj = obj + (calc_angle - obs_angle)^2;
    end
    
    % Flow errors (weight = 10, more important)
    for i = 1:length(obs_flows)
        from_bus = obs_from(i);
        to_bus = obs_to(i);
        
        line_idx = find(results.branch(:,1) == from_bus & ...
                       results.branch(:,2) == to_bus);
        
        if ~isempty(line_idx)
            calc_flow = results.branch(line_idx(1), 14);
            obs_flow = obs_flows(i);
            obj = obj + 10 * (calc_flow - obs_flow)^2;
        end
    end
end