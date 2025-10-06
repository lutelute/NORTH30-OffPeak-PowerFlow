% Compare DC power flow results with original CSV data
% Calculate RMSE between calculated and actual branch flows

clear all;
clc;

% Load DC power flow results
load('north30_dc_results.mat');

% Read original branch data from CSV
branch_data = readtable('NORTH30_OffPeak_Branch.csv');

% Extract relevant columns from CSV
csv_from = branch_data.fbus;
csv_to = branch_data.tbus;
csv_pf = branch_data.P_f;  % Power flow from -> to (MW)
csv_pt = branch_data.P_t;  % Power flow to -> from (MW)

% Extract calculated results
calc_from = results.branch(:,1);
calc_to = results.branch(:,2);
calc_pf = results.branch(:,14);  % DC power flow (MW)

fprintf('Branch Flow Comparison: CSV vs DC Power Flow\n');
fprintf('==========================================\n');
fprintf('Branch   CSV_P_f   DC_P_f   Error   CSV_P_t   DC_P_t   Error_t\n');
fprintf('-------|---------|--------|-------|---------|--------|--------\n');

% Initialize arrays for RMSE calculation
errors_pf = [];
errors_pt = [];
valid_comparisons = 0;

% Compare each branch
for i = 1:length(csv_from)
    if i > size(results.branch, 1)
        break;  % No more calculated results
    end
    
    % Find matching branch in calculated results
    from_bus = csv_from(i);
    to_bus = csv_to(i);
    
    % Find corresponding calculated branch
    calc_idx = find(calc_from == from_bus & calc_to == to_bus);
    
    if ~isempty(calc_idx)
        calc_idx = calc_idx(1);  % Take first match if multiple
        
        csv_pf_val = csv_pf(i);
        calc_pf_val = calc_pf(calc_idx);
        csv_pt_val = csv_pt(i);
        calc_pt_val = -calc_pf_val;  % DC assumes P_t = -P_f
        
        error_pf = calc_pf_val - csv_pf_val;
        error_pt = calc_pt_val - csv_pt_val;
        
        % Store errors for RMSE calculation
        if ~isnan(csv_pf_val) && ~isnan(calc_pf_val)
            errors_pf = [errors_pf; error_pf];
            valid_comparisons = valid_comparisons + 1;
        end
        
        if ~isnan(csv_pt_val) && ~isnan(calc_pt_val)
            errors_pt = [errors_pt; error_pt];
        end
        
        fprintf('%2d-%2d  |%8.1f |%7.1f |%6.1f |%8.1f |%7.1f |%7.1f\n', ...
                from_bus, to_bus, csv_pf_val, calc_pf_val, error_pf, ...
                csv_pt_val, calc_pt_val, error_pt);
    else
        fprintf('%2d-%2d  | No matching calculated branch found\n', from_bus, to_bus);
    end
end

% Calculate RMSE
if ~isempty(errors_pf)
    rmse_pf = sqrt(mean(errors_pf.^2));
    mae_pf = mean(abs(errors_pf));
    max_error_pf = max(abs(errors_pf));
    
    fprintf('\nStatistical Analysis for P_from:\n');
    fprintf('================================\n');
    fprintf('Number of valid comparisons: %d\n', length(errors_pf));
    fprintf('RMSE (Root Mean Square Error): %.2f MW\n', rmse_pf);
    fprintf('MAE (Mean Absolute Error): %.2f MW\n', mae_pf);
    fprintf('Maximum Absolute Error: %.2f MW\n', max_error_pf);
    fprintf('Standard Deviation of Errors: %.2f MW\n', std(errors_pf));
    fprintf('Mean Error (Bias): %.2f MW\n', mean(errors_pf));
else
    fprintf('\nNo valid comparisons found for P_from\n');
end

if ~isempty(errors_pt)
    rmse_pt = sqrt(mean(errors_pt.^2));
    mae_pt = mean(abs(errors_pt));
    max_error_pt = max(abs(errors_pt));
    
    fprintf('\nStatistical Analysis for P_to:\n');
    fprintf('==============================\n');
    fprintf('Number of valid comparisons: %d\n', length(errors_pt));
    fprintf('RMSE (Root Mean Square Error): %.2f MW\n', rmse_pt);
    fprintf('MAE (Mean Absolute Error): %.2f MW\n', mae_pt);
    fprintf('Maximum Absolute Error: %.2f MW\n', max_error_pt);
    fprintf('Standard Deviation of Errors: %.2f MW\n', std(errors_pt));
    fprintf('Mean Error (Bias): %.2f MW\n', mean(errors_pt));
else
    fprintf('\nNo valid comparisons found for P_to\n');
end

% Overall assessment
if ~isempty(errors_pf)
    total_csv_power = sum(abs(csv_pf), 'omitnan');
    relative_rmse = (rmse_pf / total_csv_power) * 100;
    
    fprintf('\nOverall Assessment:\n');
    fprintf('==================\n');
    fprintf('Relative RMSE: %.2f%%\n', relative_rmse);
    
    if relative_rmse < 5
        fprintf('Excellent agreement (RMSE < 5%%)\n');
    elseif relative_rmse < 10
        fprintf('Good agreement (RMSE < 10%%)\n');
    elseif relative_rmse < 20
        fprintf('Acceptable agreement (RMSE < 20%%)\n');
    else
        fprintf('Poor agreement (RMSE > 20%%)\n');
        fprintf('Consider checking:\n');
        fprintf('- Generator dispatch differences\n');
        fprintf('- Load model differences\n');
        fprintf('- Network topology\n');
        fprintf('- AC vs DC approximation limitations\n');
    end
end

% Create error histogram
if ~isempty(errors_pf)
    figure;
    histogram(errors_pf, 20);
    xlabel('Error (MW)');
    ylabel('Frequency');
    title('Distribution of Power Flow Errors (CSV vs DC Calculation)');
    grid on;
    
    % Add statistics to plot
    text(0.7, 0.8, sprintf('RMSE: %.1f MW\nMAE: %.1f MW', rmse_pf, mae_pf), ...
         'Units', 'normalized', 'BackgroundColor', 'white');
end

fprintf('\nNote: DC power flow assumes:\n');
fprintf('- Constant voltage magnitudes (1.0 pu)\n');
fprintf('- No reactive power flows\n');
fprintf('- No losses\n');
fprintf('- Linear relationship between angle and power\n');