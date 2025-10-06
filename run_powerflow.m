% NORTH30 Off-Peak Power Flow Analysis
% This script runs power flow calculation for the NORTH30 system

clear all;
clc;

% Load the MATPOWER case
mpc = north30_matpower();

% Display system information
fprintf('NORTH30 Off-Peak System Analysis\n');
fprintf('================================\n');
fprintf('Number of buses: %d\n', size(mpc.bus, 1));
fprintf('Number of generators: %d\n', size(mpc.gen, 1));
fprintf('Number of branches: %d\n', size(mpc.branch, 1));
fprintf('System MVA base: %.1f MVA\n\n', mpc.baseMVA);

% Run power flow calculation
fprintf('Running power flow calculation...\n');
results = runpf(mpc);

% Check convergence
if results.success
    fprintf('Power flow converged successfully!\n\n');
    
    % Display bus results
    fprintf('Bus Voltage Results:\n');
    fprintf('Bus#  |Voltage| Angle(deg) Pg(MW) Qg(Mvar) Pd(MW) Qd(Mvar)\n');
    fprintf('------|-------|----------|-------|--------|-------|--------\n');
    
    for i = 1:size(results.bus, 1)
        bus_num = results.bus(i, 1);
        vm = results.bus(i, 8);
        va = results.bus(i, 9);
        pd = results.bus(i, 3);
        qd = results.bus(i, 4);
        
        % Find generator at this bus
        gen_idx = find(results.gen(:, 1) == bus_num);
        if ~isempty(gen_idx)
            pg = sum(results.gen(gen_idx, 2));
            qg = sum(results.gen(gen_idx, 3));
        else
            pg = 0;
            qg = 0;
        end
        
        fprintf('%5d |%6.3f |%9.2f |%6.1f |%7.1f |%6.1f |%7.1f\n', ...
                bus_num, vm, va, pg, qg, pd, qd);
    end
    
    % Display total generation and load
    fprintf('\n');
    fprintf('System Summary:\n');
    fprintf('Total Generation: %.1f MW, %.1f Mvar\n', ...
            sum(results.gen(:, 2)), sum(results.gen(:, 3)));
    fprintf('Total Load: %.1f MW, %.1f Mvar\n', ...
            sum(results.bus(:, 3)), sum(results.bus(:, 4)));
    fprintf('Total Losses: %.1f MW, %.1f Mvar\n', ...
            sum(results.gen(:, 2)) - sum(results.bus(:, 3)), ...
            sum(results.gen(:, 3)) - sum(results.bus(:, 4)));
    
    % Branch flow analysis
    fprintf('\nBranch Flow Results (showing flows > 50 MW):\n');
    fprintf('From  To   P_from   Q_from   P_to     Q_to     Losses\n');
    fprintf('------|----|---------|---------|---------|---------|---------\n');
    
    for i = 1:size(results.branch, 1)
        pf = results.branch(i, 14);
        qf = results.branch(i, 15);
        pt = results.branch(i, 16);
        qt = results.branch(i, 17);
        
        if abs(pf) > 50  % Show only significant flows
            fprintf('%5d %4d %8.1f %8.1f %8.1f %8.1f %8.1f\n', ...
                    results.branch(i, 1), results.branch(i, 2), ...
                    pf, qf, pt, qt, pf + pt);
        end
    end
    
else
    fprintf('Power flow did not converge!\n');
    fprintf('Please check the system data.\n');
end

% Save results
save('north30_results.mat', 'results');
fprintf('\nResults saved to north30_results.mat\n');