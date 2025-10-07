% Load Allocation Inverse Problem Demonstration
% Clear demonstration of inverse problem for load allocation
% Shows the process: Unknown loads → Observed flows → Estimated loads

clear all;
clc;

fprintf('=== LOAD ALLOCATION INVERSE PROBLEM DEMONSTRATION ===\n');
fprintf('Scenario: Unknown load distribution, known branch flows\n');
fprintf('Goal: Estimate load allocation from flow measurements\n\n');

%% STEP 1: Setup True System (What we want to recover)
mpc_true = north30_matpower();
branch_data = readtable('NORTH30_OffPeak_Branch.csv');

% Extract true load distribution (our "unknown" target)
true_load_distribution = mpc_true.bus(:, 3);
load_buses = find(true_load_distribution > 0);
true_loads_only = true_load_distribution(load_buses);

% Run true power flow to get "observed" flows
results_true = rundcpf(mpc_true);

fprintf('TRUE SYSTEM STATE (Unknown to inverse problem):\n');
fprintf('- Load buses: %d\n', length(load_buses));
fprintf('- Total load: %.1f MW\n', sum(true_loads_only));
fprintf('- Load range: %.1f to %.1f MW\n', min(true_loads_only), max(true_loads_only));

%% STEP 2: Create "Observed" Flow Measurements (What we can measure)
% Select key transmission lines for flow observation
key_lines = [1, 5, 8, 12, 15, 20, 25, 30, 35, 40, 45, 50]; % Major lines
observed_flows = [];
observed_from = [];
observed_to = [];

for i = 1:length(key_lines)
    if key_lines(i) <= length(branch_data.P_f)
        flow_val = branch_data.P_f(key_lines(i));
        if abs(flow_val) > 10  % Only significant flows
            observed_flows = [observed_flows; flow_val];
            observed_from = [observed_from; branch_data.fbus(key_lines(i))];
            observed_to = [observed_to; branch_data.tbus(key_lines(i))];
        end
    end
end

fprintf('\nOBSERVED MEASUREMENTS (Available to inverse problem):\n');
fprintf('- Flow measurements: %d lines\n', length(observed_flows));
fprintf('- Flow range: %.1f to %.1f MW\n', min(abs(observed_flows)), max(abs(observed_flows)));
fprintf('- Total generation: %.1f MW (known)\n', sum(mpc_true.gen(:,2)));

%% STEP 3: Setup Inverse Problem (Start with unknown loads)
% Create case with UNKNOWN load distribution
mpc_unknown = mpc_true;
mpc_unknown.bus(:, 3) = 0;  % All loads set to ZERO (unknown)

% Decision variables: load values at load buses
n_vars = length(load_buses);
total_generation = sum(mpc_true.gen(:,2));
estimated_losses = 45;
target_total = total_generation - estimated_losses;

% Initial guess: uniform distribution
x0 = (target_total / n_vars) * ones(n_vars, 1);

fprintf('\nINVERSE PROBLEM SETUP:\n');
fprintf('- Unknown variables: %d load values\n', n_vars);
fprintf('- Known constraints: Total load = %.1f MW\n', target_total);
fprintf('- Observations: %d flow measurements\n', length(observed_flows));

%% STEP 4: Solve Load Allocation Problem
fprintf('\nSOLVING LOAD ALLOCATION...\n');

% Objective: match observed flows
objective = @(loads) load_allocation_objective(loads, load_buses, mpc_unknown, ...
                                              observed_flows, observed_from, observed_to);

% Constraints
lb = zeros(n_vars, 1);           % Non-negative loads
ub = 400 * ones(n_vars, 1);     % Reasonable upper bounds
Aeq = ones(1, n_vars);           % Total load constraint
beq = target_total;

% Solve optimization
options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 300);
[estimated_loads, fval, exitflag] = fmincon(objective, x0, [], [], Aeq, beq, lb, ub, [], options);

%% STEP 5: Validation of Load Allocation
fprintf('\n=== LOAD ALLOCATION RESULTS ===\n');

if exitflag > 0
    fprintf('✓ Load allocation converged successfully\n');
    fprintf('Final objective value: %.6f\n', fval);
    
    % Create case with estimated loads
    mpc_estimated = mpc_unknown;
    for i = 1:length(load_buses)
        bus_idx = find(mpc_estimated.bus(:,1) == load_buses(i));
        mpc_estimated.bus(bus_idx, 3) = estimated_loads(i);
    end
    
    % Run power flow with estimated loads
    results_estimated = rundcpf(mpc_estimated);
    
    if results_estimated.success
        fprintf('✓ Power flow with estimated loads converged\n');
    end
else
    fprintf('✗ Load allocation failed to converge\n');
end

%% STEP 6: Detailed Load Allocation Analysis
fprintf('\n=== LOAD ALLOCATION ACCURACY ===\n');
fprintf('Bus | True Load | Estimated | Error   | Error%% | Status\n');
fprintf('----|-----------|-----------|---------|--------|--------\n');

allocation_errors = [];
for i = 1:length(load_buses)
    bus_num = load_buses(i);
    true_load = true_loads_only(i);
    estimated_load = estimated_loads(i);
    error = estimated_load - true_load;
    
    if true_load > 0.1
        error_pct = (error / true_load) * 100;
        allocation_errors = [allocation_errors; abs(error)];
        
        if abs(error_pct) < 10
            status = 'Good';
        elseif abs(error_pct) < 25
            status = 'Fair';
        else
            status = 'Poor';
        end
    else
        error_pct = 0;
        status = 'N/A';
    end
    
    fprintf('%3d | %9.1f | %9.1f | %7.1f | %6.1f%% | %s\n', ...
            bus_num, true_load, estimated_load, error, error_pct, status);
end

% Load allocation statistics
allocation_rmse = sqrt(mean(allocation_errors.^2));
allocation_mae = mean(allocation_errors);
total_error = abs(sum(estimated_loads) - sum(true_loads_only));

fprintf('\nLoad Allocation Performance:\n');
fprintf('RMSE: %.2f MW\n', allocation_rmse);
fprintf('MAE:  %.2f MW\n', allocation_mae);
fprintf('Total load error: %.2f MW (%.2f%%)\n', ...
        total_error, (total_error/sum(true_loads_only))*100);

%% STEP 7: Flow Matching Verification
fprintf('\n=== FLOW MATCHING VERIFICATION ===\n');
fprintf('Line   | Observed | Calculated | Error   | Error%%\n');
fprintf('-------|----------|------------|---------|--------\n');

flow_match_errors = [];
for i = 1:length(observed_flows)
    from_bus = observed_from(i);
    to_bus = observed_to(i);
    observed_flow = observed_flows(i);
    
    % Find calculated flow
    calc_idx = find(results_estimated.branch(:,1) == from_bus & ...
                   results_estimated.branch(:,2) == to_bus);
    
    if ~isempty(calc_idx)
        calculated_flow = results_estimated.branch(calc_idx(1), 14);
        error = calculated_flow - observed_flow;
        
        if abs(observed_flow) > 0.1
            error_pct = (error / observed_flow) * 100;
        else
            error_pct = 0;
        end
        
        flow_match_errors = [flow_match_errors; abs(error)];
        
        fprintf('%2d→%-2d  | %8.1f | %10.1f | %7.1f | %6.1f%%\n', ...
                from_bus, to_bus, observed_flow, calculated_flow, error, error_pct);
    end
end

flow_rmse = sqrt(mean(flow_match_errors.^2));
fprintf('\nFlow Matching RMSE: %.2f MW\n', flow_rmse);

%% STEP 8: Comprehensive Visualization
figure('Position', [50, 50, 1600, 1000]);

% Subplot 1: Load allocation comparison
subplot(2,4,1);
bar([true_loads_only, estimated_loads]);
xlabel('Load Bus Index');
ylabel('Load (MW)');
title('Load Allocation: True vs Estimated');
legend('True Load', 'Estimated Load', 'Location', 'best');
grid on;

% Subplot 2: Load allocation scatter
subplot(2,4,2);
scatter(true_loads_only, estimated_loads, 80, 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
min_val = min([true_loads_only; estimated_loads]);
max_val = max([true_loads_only; estimated_loads]);
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
xlabel('True Load (MW)');
ylabel('Estimated Load (MW)');
title('Load Allocation Accuracy');
grid on;
axis equal;

% Correlation coefficient
load_corr = corr(true_loads_only, estimated_loads);
text(0.05, 0.95, sprintf('R² = %.3f', load_corr^2), 'Units', 'normalized', ...
     'BackgroundColor', 'white', 'FontSize', 12);

% Subplot 3: Load allocation errors
subplot(2,4,3);
load_errors_plot = estimated_loads - true_loads_only;
bar(load_errors_plot, 'FaceColor', [0.8 0.4 0.4]);
xlabel('Load Bus Index');
ylabel('Allocation Error (MW)');
title('Load Allocation Errors');
grid on;
yline(0, 'k--', 'LineWidth', 1);

% Subplot 4: Flow matching accuracy
subplot(2,4,4);
if ~isempty(flow_match_errors)
    bar(flow_match_errors, 'FaceColor', [0.4 0.6 0.8]);
    xlabel('Measurement Index');
    ylabel('Flow Error (MW)');
    title('Flow Matching Errors');
    grid on;
end

% Subplot 5: Load distribution pie chart (True)
subplot(2,4,5);
if length(true_loads_only) <= 8
    pie(true_loads_only, cellstr(num2str(load_buses)));
    title('True Load Distribution');
else
    % Group smaller loads
    [sorted_loads, sort_idx] = sort(true_loads_only, 'descend');
    top_loads = sorted_loads(1:min(6, length(sorted_loads)));
    top_buses = load_buses(sort_idx(1:length(top_loads)));
    other_load = sum(sorted_loads(length(top_loads)+1:end));
    
    pie_data = [top_loads; other_load];
    pie_labels = [cellstr(num2str(top_buses)); {'Others'}];
    pie(pie_data, pie_labels);
    title('True Load Distribution');
end

% Subplot 6: Load distribution pie chart (Estimated)
subplot(2,4,6);
if length(estimated_loads) <= 8
    pie(estimated_loads, cellstr(num2str(load_buses)));
    title('Estimated Load Distribution');
else
    [sorted_est, sort_idx] = sort(estimated_loads, 'descend');
    top_est = sorted_est(1:min(6, length(sorted_est)));
    top_buses_est = load_buses(sort_idx(1:length(top_est)));
    other_est = sum(sorted_est(length(top_est)+1:end));
    
    pie_data_est = [top_est; other_est];
    pie_labels_est = [cellstr(num2str(top_buses_est)); {'Others'}];
    pie(pie_data_est, pie_labels_est);
    title('Estimated Load Distribution');
end

% Subplot 7: Error histogram
subplot(2,4,7);
histogram(load_errors_plot, min(12, length(load_errors_plot)));
xlabel('Load Allocation Error (MW)');
ylabel('Frequency');
title('Error Distribution');
grid on;

% Add statistics
mean_error = mean(load_errors_plot);
std_error = std(load_errors_plot);
xline(mean_error, 'r--', 'LineWidth', 2, 'Label', sprintf('Mean: %.1f', mean_error));
xline(mean_error + std_error, 'g--', 'LineWidth', 1);
xline(mean_error - std_error, 'g--', 'LineWidth', 1);

% Subplot 8: Summary statistics
subplot(2,4,8);
axis off;

% Calculate success rate outside the cell array
good_estimates = sum(abs(load_errors_plot./true_loads_only) < 0.25);

summary_stats = {
    'LOAD ALLOCATION SUMMARY';
    '======================';
    sprintf('Load RMSE: %.1f MW', allocation_rmse);
    sprintf('Load MAE: %.1f MW', allocation_mae);
    sprintf('Load Correlation: R^2 = %.3f', load_corr^2);
    '';
    sprintf('Flow RMSE: %.1f MW', flow_rmse);
    sprintf('Total Load Recovery:');
    sprintf('  True: %.0f MW', sum(true_loads_only));
    sprintf('  Est:  %.0f MW', sum(estimated_loads));
    sprintf('  Error: %.1f%%', total_error/sum(true_loads_only)*100);
    '';
    sprintf('Success Rate:');
    sprintf('  Good estimates: %d/%d', good_estimates, length(true_loads_only));
};

text(0.05, 0.95, summary_stats, 'FontSize', 10, 'FontName', 'Courier', ...
     'VerticalAlignment', 'top');

sgtitle('Load Allocation Inverse Problem: From Flow Measurements to Load Distribution', ...
        'FontSize', 16, 'FontWeight', 'bold');

%% STEP 9: Save Results
save('load_allocation_demo_results.mat', 'true_loads_only', 'estimated_loads', ...
     'load_buses', 'allocation_rmse', 'flow_rmse', 'load_corr');

print(gcf, 'load_allocation_inverse_problem.png', '-dpng', '-r300');

fprintf('\n=== DEMONSTRATION SUMMARY ===\n');
fprintf('Inverse Problem: Load allocation from flow measurements\n');
fprintf('Input: %d flow observations from key transmission lines\n', length(observed_flows));
fprintf('Output: Load distribution across %d buses\n', length(load_buses));
fprintf('Accuracy: Load RMSE = %.1f MW, Flow RMSE = %.1f MW\n', allocation_rmse, flow_rmse);
fprintf('Correlation: R² = %.3f\n', load_corr^2);
fprintf('Total load recovery: %.1f%%\n', (1-total_error/sum(true_loads_only))*100);

fprintf('\nFiles generated:\n');
fprintf('- load_allocation_inverse_problem.png\n');
fprintf('- load_allocation_demo_results.mat\n');

%% Objective Function for Load Allocation
function obj = load_allocation_objective(loads, load_buses, mpc_base, target_flows, flow_from, flow_to)
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
        obj = 1e8;  % Large penalty for non-convergent solutions
        return;
    end
    
    % Calculate flow matching objective
    obj = 0;
    for i = 1:length(target_flows)
        from_bus = flow_from(i);
        to_bus = flow_to(i);
        target = target_flows(i);
        
        % Find calculated flow
        calc_idx = find(results.branch(:,1) == from_bus & results.branch(:,2) == to_bus);
        
        if ~isempty(calc_idx)
            calculated = results.branch(calc_idx(1), 14);
            obj = obj + (calculated - target)^2;
        else
            obj = obj + target^2 * 1000;  % Large penalty if branch not found
        end
    end
end