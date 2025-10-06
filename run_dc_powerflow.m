% NORTH30 Off-Peak DC Power Flow Analysis
% This script runs DC power flow calculation for the NORTH30 system

clear all;
clc;

% Load the MATPOWER case
mpc = north30_matpower();

% Display system information
fprintf('NORTH30 Off-Peak System - DC Power Flow Analysis\n');
fprintf('===============================================\n');
fprintf('Number of buses: %d\n', size(mpc.bus, 1));
fprintf('Number of generators: %d\n', size(mpc.gen, 1));
fprintf('Number of branches: %d\n', size(mpc.branch, 1));
fprintf('System MVA base: %.1f MVA\n\n', mpc.baseMVA);

% Check system balance
total_gen_p = sum(mpc.gen(:,2));
total_load_p = sum(mpc.bus(:,3));
fprintf('Total Generation: %.1f MW\n', total_gen_p);
fprintf('Total Load: %.1f MW\n', total_load_p);
fprintf('Power Balance: %.1f MW\n\n', total_gen_p - total_load_p);

% Run DC power flow calculation
fprintf('Running DC power flow calculation...\n');
results = rundcpf(mpc);

% Check convergence
if results.success
    fprintf('DC power flow converged successfully!\n\n');
    
    % Display bus results
    fprintf('Bus Results (DC Power Flow):\n');
    fprintf('Bus#  Voltage Angle(deg) Pg(MW) Pd(MW)  P_net(MW)\n');
    fprintf('------|-------|----------|------|-------|----------\n');
    
    for i = 1:size(results.bus, 1)
        bus_num = results.bus(i, 1);
        va = results.bus(i, 9);  % Voltage angle
        pd = results.bus(i, 3);  % Load
        
        % Find generator at this bus
        gen_idx = find(results.gen(:, 1) == bus_num);
        if ~isempty(gen_idx)
            pg = sum(results.gen(gen_idx, 2));
        else
            pg = 0;
        end
        
        p_net = pg - pd;  % Net power injection
        
        fprintf('%5d |%6.3f |%9.2f |%5.1f |%6.1f |%9.1f\n', ...
                bus_num, 1.0, va, pg, pd, p_net);
    end
    
    % Display branch flows
    fprintf('\nBranch Flow Results (DC Power Flow):\n');
    fprintf('From  To   P_flow(MW)  X(pu)    Angle_diff(deg)\n');
    fprintf('------|----|-----------|---------|--------------\n');
    
    for i = 1:size(results.branch, 1)
        from_bus = results.branch(i, 1);
        to_bus = results.branch(i, 2);
        x = results.branch(i, 4);  % Reactance
        pf = results.branch(i, 14); % Power flow from -> to
        
        % Calculate angle difference
        from_idx = find(results.bus(:,1) == from_bus);
        to_idx = find(results.bus(:,1) == to_bus);
        angle_diff = results.bus(from_idx, 9) - results.bus(to_idx, 9);
        
        if abs(pf) > 10  % Show flows > 10 MW
            fprintf('%5d %4d %10.1f %8.4f %13.2f\n', ...
                    from_bus, to_bus, pf, x, angle_diff);
        end
    end
    
    % System summary
    fprintf('\nSystem Summary (DC Power Flow):\n');
    fprintf('Total Generation: %.1f MW\n', sum(results.gen(:, 2)));
    fprintf('Total Load: %.1f MW\n', sum(results.bus(:, 3)));
    fprintf('Total Losses: %.1f MW (DC approximation: 0 MW)\n', ...
            sum(results.gen(:, 2)) - sum(results.bus(:, 3)));
    
    % Find heavily loaded lines
    fprintf('\nHeavily Loaded Lines (>100 MW):\n');
    fprintf('From  To   P_flow(MW)  Loading(%%)\n');
    fprintf('------|----|-----------|---------\n');
    
    for i = 1:size(results.branch, 1)
        pf = abs(results.branch(i, 14));
        rating = results.branch(i, 6);  % rateA
        if pf > 100 && rating > 0
            loading = (pf / rating) * 100;
            fprintf('%5d %4d %10.1f %8.1f\n', ...
                    results.branch(i, 1), results.branch(i, 2), ...
                    results.branch(i, 14), loading);
        end
    end
    
    % Voltage angle analysis
    fprintf('\nVoltage Angle Analysis:\n');
    fprintf('Max angle difference: %.2f degrees\n', ...
            max(results.bus(:,9)) - min(results.bus(:,9)));
    fprintf('Reference bus (angle = 0): Bus %d\n', ...
            results.bus(results.bus(:,9) == 0, 1));
    
else
    fprintf('DC power flow did not converge!\n');
    fprintf('This indicates a fundamental problem with the network connectivity.\n');
end

% Save results
save('north30_dc_results.mat', 'results');
fprintf('\nResults saved to north30_dc_results.mat\n');