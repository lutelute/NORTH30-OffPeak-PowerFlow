% Realistic Load Allocation Inverse Problem
% Only use branch flow measurements, no direct bus voltage data
% Objective: Allocate loads to match observed branch flows via DC power flow

clear all;
clc;

fprintf('=== REALISTIC LOAD ALLOCATION INVERSE PROBLEM ===\n');
fprintf('Constraint: Only branch flow measurements available\n');
fprintf('Method: DC power flow with load allocation optimization\n\n');

%% STEP 1: Load True System Data
mpc_true = north30_matpower();
branch_data = readtable('NORTH30_OffPeak_Branch.csv');

% Extract true system state
true_loads = mpc_true.bus(:, 3);
load_buses = find(true_loads ~= 0);
n_load_buses = length(load_buses);

fprintf('System Information:\n');
fprintf('- Total buses: %d\n', size(mpc_true.bus, 1));
fprintf('- Load buses: %d\n', n_load_buses);
fprintf('- True total load: %.1f MW\n', sum(true_loads));
fprintf('- Total generation: %.1f MW\n\n', sum(mpc_true.gen(:,2)));

%% STEP 2: Define Observable Branch Flows (Realistic Scenario)
% In practice, we only measure flows on major transmission lines
% Select representative branches that carry significant power

% Get all branch flows from CSV
observed_branch_flows = branch_data.P_f;
observed_from_buses = branch_data.fbus;
observed_to_buses = branch_data.tbus;

% Remove very small flows (measurement noise or unimportant lines)
significant_flow_threshold = 5; % MW
significant_indices = find(abs(observed_branch_flows) > significant_flow_threshold);

% Use only significant flows for optimization
target_flows = observed_branch_flows(significant_indices);
target_from = observed_from_buses(significant_indices);
target_to = observed_to_buses(significant_indices);

fprintf('Measurement Setup:\n');
fprintf('- Total branches in system: %d\n', length(observed_branch_flows));
fprintf('- Significant flow measurements: %d (> %.1f MW)\n', ...
        length(target_flows), significant_flow_threshold);
fprintf('- Flow range: %.1f to %.1f MW\n', min(abs(target_flows)), max(abs(target_flows)));

%% STEP 3: Set Up Load Allocation Problem
% Unknown: Load distribution among load buses
% Known: Total generation, branch topology
% Objective: Match observed branch flows

% Create base case with zero loads (unknown distribution)
mpc_unknown = mpc_true;
mpc_unknown.bus(:, 3) = 0; % All loads unknown

% Initial guess: uniform load distribution
total_generation = sum(mpc_true.gen(:,2));
estimated_losses = 50; % Estimate transmission losses
target_total_load = total_generation - estimated_losses;

x0 = (target_total_load / n_load_buses) * ones(n_load_buses, 1);

fprintf('\nOptimization Setup:\n');
fprintf('- Decision variables: %d load values\n', n_load_buses);
fprintf('- Target total load: %.1f MW\n', target_total_load);
fprintf('- Initial guess: %.1f MW per bus\n', target_total_load / n_load_buses);

%% STEP 4: Define Optimization Problem
% Objective: Minimize squared error between calculated and observed flows
objective_function = @(loads) flow_matching_objective(loads, load_buses, mpc_unknown, ...
                                                     target_flows, target_from, target_to);

% Constraints (Allow distributed generation)
lb = -150 * ones(n_load_buses, 1);    % Allow negative loads (distributed generation)
ub = 350 * ones(n_load_buses, 1);     % Upper bound for loads

% Power balance constraint: total load = generation - losses
Aeq = ones(1, n_load_buses);
beq = target_total_load;

fprintf('- Load bounds: %.0f to %.0f MW per bus (negative = distributed generation)\n', lb(1), ub(1));
fprintf('- Power balance constraint: Σ loads = %.1f MW\n', target_total_load);

%% STEP 5: Solve Optimization Problem
fprintf('\nSolving load allocation problem...\n');

options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'MaxIterations', 500, ...
    'OptimalityTolerance', 1e-6, ...
    'StepTolerance', 1e-8, ...
    'ConstraintTolerance', 1e-6);

[optimal_loads, fval, exitflag, output] = fmincon(objective_function, x0, [], [], ...
                                                 Aeq, beq, lb, ub, [], options);

%% STEP 6: Validation and Results
fprintf('\n=== OPTIMIZATION RESULTS ===\n');

if exitflag > 0
    fprintf('✓ Optimization converged successfully\n');
    fprintf('Final objective value: %.6f\n', fval);
    fprintf('Iterations: %d\n', output.iterations);
    
    % Calculate RMSE from final objective
    rmse_flows = sqrt(fval / length(target_flows));
    fprintf('Branch flow RMSE: %.2f MW\n', rmse_flows);
else
    fprintf('⚠ Optimization failed (exitflag: %d)\n', exitflag);
end

% Create estimated case and run power flow
mpc_estimated = mpc_unknown;
for i = 1:n_load_buses
    bus_idx = find(mpc_estimated.bus(:,1) == load_buses(i));
    mpc_estimated.bus(bus_idx, 3) = optimal_loads(i);
end

% Run DC power flow with estimated loads
results_estimated = rundcpf(mpc_estimated);
results_true = rundcpf(mpc_true);

%% STEP 7: Detailed Comparison
fprintf('\n=== LOAD ALLOCATION RESULTS ===\n');
fprintf('Bus | True Load | Estimated | Error   | Error%%\n');
fprintf('----|-----------|-----------|---------|--------\n');

load_errors = [];
for i = 1:n_load_buses
    bus_num = load_buses(i);
    bus_idx = find(mpc_true.bus(:,1) == bus_num);
    
    true_load = mpc_true.bus(bus_idx, 3);
    estimated_load = optimal_loads(i);
    error = estimated_load - true_load;
    
    if true_load > 0.1
        error_pct = (error / true_load) * 100;
        load_errors = [load_errors; abs(error)];
    else
        error_pct = NaN;
    end
    
    fprintf('%3d | %9.1f | %9.1f | %7.1f | %6.1f%%\n', ...
            bus_num, true_load, estimated_load, error, error_pct);
end

% Load allocation statistics
load_rmse = sqrt(mean(load_errors.^2));
load_mae = mean(load_errors);
total_load_error = abs(sum(optimal_loads) - sum(true_loads));

fprintf('\nLoad Allocation Statistics:\n');
fprintf('RMSE: %.2f MW\n', load_rmse);
fprintf('MAE:  %.2f MW\n', load_mae);
fprintf('Total load error: %.2f MW (%.1f%%)\n', ...
        total_load_error, (total_load_error/sum(true_loads))*100);

%% STEP 8: Branch Flow Validation
fprintf('\n=== BRANCH FLOW MATCHING ===\n');

if results_estimated.success && results_true.success
    fprintf('✓ Power flows converged for both cases\n\n');
    
    fprintf('Branch Flow Comparison (Significant Flows Only):\n');
    fprintf('From-To | True Flow | Estimated | Error   | Error%%\n');
    fprintf('--------|-----------|-----------|---------|--------\n');
    
    flow_errors = [];
    for i = 1:length(target_flows)
        from_bus = target_from(i);
        to_bus = target_to(i);
        true_flow = target_flows(i);
        
        % Find estimated flow
        est_idx = find(results_estimated.branch(:,1) == from_bus & ...
                      results_estimated.branch(:,2) == to_bus);
        
        if ~isempty(est_idx)
            estimated_flow = results_estimated.branch(est_idx(1), 14);
            error = estimated_flow - true_flow;
            
            if abs(true_flow) > 0.1
                error_pct = (error / true_flow) * 100;
            else
                error_pct = 0;
            end
            
            flow_errors = [flow_errors; abs(error)];
            
            fprintf(' %2d→%-2d  | %9.1f | %9.1f | %7.1f | %6.1f%%\n', ...
                    from_bus, to_bus, true_flow, estimated_flow, error, error_pct);
        end
    end
    
    % Flow matching statistics
    if ~isempty(flow_errors)
        flow_rmse_final = sqrt(mean(flow_errors.^2));
        flow_mae = mean(flow_errors);
        max_flow_error = max(flow_errors);
    else
        flow_rmse_final = NaN;
        flow_mae = NaN;
        max_flow_error = NaN;
    end
    
    fprintf('\nBranch Flow Statistics:\n');
    fprintf('RMSE: %.2f MW\n', flow_rmse_final);
    fprintf('MAE:  %.2f MW\n', flow_mae);
    fprintf('Max error: %.2f MW\n', max_flow_error);
    fprintf('Mean target flow: %.2f MW\n', mean(abs(target_flows)));
    fprintf('Relative RMSE: %.1f%%\n', (flow_rmse_final/mean(abs(target_flows)))*100);
    
else
    fprintf('✗ Power flow convergence issues\n');
end

%% STEP 9: Visualization
figure('Position', [100, 100, 1400, 800]);

% Subplot 1: Load comparison
subplot(2,3,1);
true_load_vals = [];
est_load_vals = [];
for i = 1:n_load_buses
    bus_idx = find(mpc_true.bus(:,1) == load_buses(i));
    true_load_vals = [true_load_vals; mpc_true.bus(bus_idx, 3)];
    est_load_vals = [est_load_vals; optimal_loads(i)];
end

bar([true_load_vals, est_load_vals]);
xlabel('Load Bus Index');
ylabel('Load (MW)');
title('Load Allocation: True vs Estimated');
legend('True', 'Estimated', 'Location', 'best');
grid on;

% Subplot 2: Load correlation
subplot(2,3,2);
scatter(true_load_vals, est_load_vals, 60, 'filled');
hold on;
min_val = min([true_load_vals; est_load_vals]);
max_val = max([true_load_vals; est_load_vals]);
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
xlabel('True Load (MW)');
ylabel('Estimated Load (MW)');
title('Load Correlation');
grid on;
axis equal;

% Correlation coefficient
load_corr = corr(true_load_vals, est_load_vals);
text(0.05, 0.95, sprintf('R² = %.3f', load_corr^2), 'Units', 'normalized', ...
     'BackgroundColor', 'white');

% Subplot 3: Flow comparison
subplot(2,3,3);
if exist('flow_errors', 'var') && ~isempty(flow_errors)
    % Ensure same length for plotting
    n_points = min(length(target_flows), length(flow_errors));
    if n_points > 0
        scatter(abs(target_flows(1:n_points)), flow_errors(1:n_points), 60, 'filled');
        xlabel('Target Flow Magnitude (MW)');
        ylabel('Flow Error (MW)');
        title('Flow Matching Accuracy');
        grid on;
    else
        text(0.5, 0.5, 'No flow data available', 'HorizontalAlignment', 'center');
    end
else
    text(0.5, 0.5, 'No flow errors calculated', 'HorizontalAlignment', 'center');
end

% Subplot 4: Load errors
subplot(2,3,4);
load_error_vals = est_load_vals - true_load_vals;
bar(load_error_vals);
xlabel('Load Bus Index');
ylabel('Load Error (MW)');
title('Load Allocation Errors');
grid on;
yline(0, 'k--');

% Subplot 5: Flow error distribution
subplot(2,3,5);
if exist('flow_errors', 'var') && ~isempty(flow_errors)
    histogram(flow_errors, min(15, length(flow_errors)));
    xlabel('Flow Error (MW)');
    ylabel('Frequency');
    title('Flow Error Distribution');
    grid on;
else
    text(0.5, 0.5, 'No flow error data', 'HorizontalAlignment', 'center');
end

% Subplot 6: Summary
subplot(2,3,6);
axis off;
if exitflag > 0
    converged_str = 'YES';
else
    converged_str = 'NO';
end

summary_text = {
    'OPTIMIZATION SUMMARY';
    '==================';
    sprintf('Converged: %s', converged_str);
    sprintf('Load RMSE: %.1f MW', load_rmse);
    sprintf('Flow RMSE: %.1f MW', flow_rmse_final);
    sprintf('Load Correlation: %.3f', load_corr);
    '';
    sprintf('Total Load Recovery:');
    sprintf('True: %.0f MW', sum(true_load_vals));
    sprintf('Est:  %.0f MW', sum(est_load_vals));
    sprintf('Error: %.1f%%', total_load_error/sum(true_load_vals)*100);
};

text(0.1, 0.9, summary_text, 'FontSize', 11, 'FontName', 'Courier', ...
     'VerticalAlignment', 'top');

sgtitle('Realistic Load Allocation: Flow-Based Inverse Problem', 'FontSize', 14);

%% STEP 10: Save Results
save('realistic_load_allocation_results.mat', 'optimal_loads', 'load_buses', ...
     'true_load_vals', 'est_load_vals', 'target_flows', 'flow_errors', ...
     'load_rmse', 'flow_rmse_final', 'load_corr');

print(gcf, 'realistic_load_allocation.png', '-dpng', '-r300');

fprintf('\n=== FINAL SUMMARY ===\n');
fprintf('Problem: Load allocation from branch flow measurements only\n');
fprintf('Method: DC power flow with constrained optimization\n');
if exitflag > 0
    fprintf('Success: YES\n');
else
    fprintf('Success: NO\n');
end
fprintf('Load allocation RMSE: %.2f MW\n', load_rmse);
fprintf('Flow matching RMSE: %.2f MW\n', flow_rmse_final);
fprintf('Load correlation: R² = %.3f\n', load_corr^2);

fprintf('\nFiles saved:\n');
fprintf('- realistic_load_allocation_results.mat\n');
fprintf('- realistic_load_allocation.png\n');

%% Objective Function: Flow Matching Only
function obj_val = flow_matching_objective(loads, load_buses, mpc_base, ...
                                          target_flows, target_from, target_to)
    
    % Update case with current load allocation
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
        obj_val = 1e6; % Large penalty for non-convergent solutions
        return;
    end
    
    % Calculate sum of squared flow errors
    obj_val = 0;
    
    for i = 1:length(target_flows)
        from_bus = target_from(i);
        to_bus = target_to(i);
        target_flow = target_flows(i);
        
        % Find corresponding branch in results
        branch_idx = find(results.branch(:,1) == from_bus & ...
                         results.branch(:,2) == to_bus);
        
        if ~isempty(branch_idx)
            calculated_flow = results.branch(branch_idx(1), 14);
            flow_error = calculated_flow - target_flow;
            obj_val = obj_val + flow_error^2;
        else
            % Large penalty if branch not found
            obj_val = obj_val + (target_flow)^2 * 100;
        end
    end
end