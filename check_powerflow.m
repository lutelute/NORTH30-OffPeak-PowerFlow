% Check power flow convergence and system balance
clear all;
clc;

% Load the case
mpc = north30_matpower();

% Check system balance
total_gen_p = sum(mpc.gen(:,2));
total_load_p = sum(mpc.bus(:,3));
total_gen_q = sum(mpc.gen(:,3));
total_load_q = sum(mpc.bus(:,4));

fprintf('System Balance Check:\n');
fprintf('Total Generation P: %.1f MW\n', total_gen_p);
fprintf('Total Load P: %.1f MW\n', total_load_p);
fprintf('P Balance: %.1f MW\n', total_gen_p - total_load_p);
fprintf('Total Generation Q: %.1f Mvar\n', total_gen_q);
fprintf('Total Load Q: %.1f Mvar\n', total_load_q);
fprintf('Q Balance: %.1f Mvar\n\n', total_gen_q - total_load_q);

% Check for problematic data
fprintf('Data Quality Check:\n');

% Check for buses with no voltage specified
zero_voltage = find(mpc.bus(:,8) == 0);
if ~isempty(zero_voltage)
    fprintf('WARNING: Buses with zero voltage magnitude: %s\n', num2str(zero_voltage'));
end

% Check for branches with zero impedance
zero_impedance = find(mpc.branch(:,3) == 0 & mpc.branch(:,4) == 0);
if ~isempty(zero_impedance)
    fprintf('WARNING: Branches with zero impedance: %s\n', num2str(zero_impedance'));
end

% Check for isolated buses
connected_buses = unique([mpc.branch(:,1); mpc.branch(:,2)]);
all_buses = mpc.bus(:,1);
isolated_buses = setdiff(all_buses, connected_buses);
if ~isempty(isolated_buses)
    fprintf('WARNING: Isolated buses: %s\n', num2str(isolated_buses'));
end

% Run power flow with verbose output
fprintf('\nRunning power flow with verbose output:\n');
mpopt = mpoption('verbose', 2, 'pf.nr.max_it', 20);
results = runpf(mpc, mpopt);

if results.success
    fprintf('\nPower flow converged!\n');
else
    fprintf('\nPower flow failed to converge. Trying DC power flow...\n');
    dc_results = rundcpf(mpc);
    if dc_results.success
        fprintf('DC power flow converged. AC convergence issue likely due to:\n');
        fprintf('- Voltage magnitude constraints\n');
        fprintf('- Reactive power limits\n');
        fprintf('- Initial voltage settings\n');
    end
end