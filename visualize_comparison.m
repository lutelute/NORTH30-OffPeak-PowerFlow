% Visualize comparison between CSV data and DC power flow results
% Plot branch flows with branch number on x-axis and power flow (MW) on y-axis

clear all;
clc;

% Load DC power flow results
if exist('north30_dc_results.mat', 'file')
    load('north30_dc_results.mat');
else
    fprintf('Error: DC power flow results not found. Please run run_dc_powerflow.m first.\n');
    return;
end

% Read original branch data from CSV
try
    branch_data = readtable('NORTH30_OffPeak_Branch.csv');
catch
    fprintf('Error: Cannot read CSV file. Please check file path.\n');
    return;
end

% Extract data from CSV
csv_from = branch_data.fbus;
csv_to = branch_data.tbus;
csv_pf = branch_data.P_f;  % Power flow from -> to (MW)
csv_pt = branch_data.P_t;  % Power flow to -> from (MW)

% Extract calculated results
calc_from = results.branch(:,1);
calc_to = results.branch(:,2);
calc_pf = results.branch(:,14);  % DC power flow (MW)

% Create branch labels for x-axis
n_branches = min(length(csv_from), size(results.branch, 1));
branch_numbers = 1:n_branches;
branch_labels = cell(n_branches, 1);

% Match CSV data with calculated results
csv_matched = nan(n_branches, 1);
calc_matched = nan(n_branches, 1);
csv_pt_matched = nan(n_branches, 1);
calc_pt_matched = nan(n_branches, 1);

fprintf('Matching branches between CSV and calculated results...\n');

for i = 1:n_branches
    if i <= length(csv_from) && i <= size(results.branch, 1)
        % Create branch label
        branch_labels{i} = sprintf('%d-%d', csv_from(i), csv_to(i));
        
        % Store CSV values
        csv_matched(i) = csv_pf(i);
        csv_pt_matched(i) = csv_pt(i);
        
        % Store calculated values
        calc_matched(i) = calc_pf(i);
        calc_pt_matched(i) = -calc_pf(i);  % DC assumes P_t = -P_f
    end
end

% Remove NaN values for plotting
valid_idx = ~isnan(csv_matched) & ~isnan(calc_matched);
branch_numbers_valid = branch_numbers(valid_idx);
csv_valid = csv_matched(valid_idx);
calc_valid = calc_matched(valid_idx);
csv_pt_valid = csv_pt_matched(valid_idx);
calc_pt_valid = calc_pt_matched(valid_idx);
labels_valid = branch_labels(valid_idx);

% Create visualization
figure('Position', [100, 100, 1400, 800]);

% Subplot 1: P_from comparison
subplot(2,2,1);
hold on;
plot(branch_numbers_valid, csv_valid, 'bo-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'CSV Data (P\_from)');
plot(branch_numbers_valid, calc_valid, 'rs--', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'DC Calculation (P\_from)');
xlabel('Branch Number');
ylabel('Power Flow (MW)');
title('Branch Power Flow Comparison: From Bus → To Bus');
legend('Location', 'best');
grid on;
hold off;

% Subplot 2: P_to comparison
subplot(2,2,2);
hold on;
plot(branch_numbers_valid, csv_pt_valid, 'go-', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'CSV Data (P\_to)');
plot(branch_numbers_valid, calc_pt_valid, 'ms--', 'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', 'DC Calculation (P\_to)');
xlabel('Branch Number');
ylabel('Power Flow (MW)');
title('Branch Power Flow Comparison: To Bus ← From Bus');
legend('Location', 'best');
grid on;
hold off;

% Subplot 3: Error analysis
subplot(2,2,3);
errors = calc_valid - csv_valid;
bar(branch_numbers_valid, errors, 'FaceColor', [0.8 0.4 0.4]);
xlabel('Branch Number');
ylabel('Error (MW)');
title('Power Flow Error (DC Calculation - CSV Data)');
grid on;

% Add zero line
hold on;
plot([min(branch_numbers_valid), max(branch_numbers_valid)], [0, 0], 'k--', 'LineWidth', 1);
hold off;

% Subplot 4: Scatter plot for correlation
subplot(2,2,4);
scatter(csv_valid, calc_valid, 60, 'filled', 'MarkerFaceAlpha', 0.7);
hold on;

% Perfect correlation line
min_val = min([csv_valid; calc_valid]);
max_val = max([csv_valid; calc_valid]);
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2, 'DisplayName', 'Perfect Match');

% Linear fit
p = polyfit(csv_valid, calc_valid, 1);
fit_line = polyval(p, [min_val, max_val]);
plot([min_val, max_val], fit_line, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('Linear Fit (R² = %.3f)', corr(csv_valid, calc_valid)^2));

xlabel('CSV Data (MW)');
ylabel('DC Calculation (MW)');
title('Correlation: CSV vs DC Calculation');
legend('Location', 'best');
grid on;
axis equal;
hold off;

% Calculate and display statistics
rmse = sqrt(mean(errors.^2));
mae = mean(abs(errors));
max_error = max(abs(errors));
correlation = corr(csv_valid, calc_valid);

% Add overall statistics as text
sgtitle(sprintf('NORTH30 Branch Flow Comparison\nRMSE: %.2f MW | MAE: %.2f MW | Max Error: %.2f MW | Correlation: %.3f', ...
                rmse, mae, max_error, correlation), 'FontSize', 14);

% Print detailed statistics
fprintf('\n=== DETAILED COMPARISON STATISTICS ===\n');
fprintf('Number of branches compared: %d\n', length(branch_numbers_valid));
fprintf('RMSE (Root Mean Square Error): %.2f MW\n', rmse);
fprintf('MAE (Mean Absolute Error): %.2f MW\n', mae);
fprintf('Maximum Absolute Error: %.2f MW\n', max_error);
fprintf('Correlation Coefficient: %.3f\n', correlation);
fprintf('R-squared: %.3f\n', correlation^2);
fprintf('Mean CSV Power Flow: %.2f MW\n', mean(abs(csv_valid)));
fprintf('Mean DC Calculated Power Flow: %.2f MW\n', mean(abs(calc_valid)));
fprintf('Relative RMSE: %.2f%%\n', (rmse / mean(abs(csv_valid))) * 100);

% Identify branches with largest errors
[~, error_idx] = sort(abs(errors), 'descend');
fprintf('\nBranches with largest errors:\n');
fprintf('Rank | Branch | CSV (MW) | DC (MW) | Error (MW)\n');
fprintf('-----|--------|----------|---------|----------\n');
for i = 1:min(10, length(error_idx))
    idx = error_idx(i);
    fprintf('%4d | %6s | %8.1f | %7.1f | %9.1f\n', ...
            i, labels_valid{idx}, csv_valid(idx), calc_valid(idx), errors(idx));
end

% Create error distribution histogram
figure('Position', [200, 200, 800, 600]);
subplot(2,1,1);
histogram(errors, 20, 'FaceColor', [0.7 0.7 0.9], 'EdgeColor', 'black');
xlabel('Error (MW)');
ylabel('Frequency');
title('Distribution of Power Flow Errors');
grid on;

% Add statistics to histogram
mean_error = mean(errors);
std_error = std(errors);
xline(mean_error, 'r--', 'LineWidth', 2, 'Label', sprintf('Mean: %.2f', mean_error));
xline(mean_error + std_error, 'g--', 'LineWidth', 1, 'Label', sprintf('+1σ: %.2f', mean_error + std_error));
xline(mean_error - std_error, 'g--', 'LineWidth', 1, 'Label', sprintf('-1σ: %.2f', mean_error - std_error));

% Q-Q plot for normality check
subplot(2,1,2);
qqplot(errors);
title('Q-Q Plot: Error Distribution vs Normal Distribution');
grid on;

% Save results
fprintf('\nSaving visualization results...\n');
save('branch_flow_comparison.mat', 'csv_valid', 'calc_valid', 'errors', 'branch_numbers_valid', 'labels_valid');

% Export high-quality figures
print(gcf, 'error_distribution.png', '-dpng', '-r300');
figure(1);
print(gcf, 'branch_flow_comparison.png', '-dpng', '-r300');

fprintf('Figures saved as:\n');
fprintf('- branch_flow_comparison.png\n');
fprintf('- error_distribution.png\n');
fprintf('Data saved as: branch_flow_comparison.mat\n');