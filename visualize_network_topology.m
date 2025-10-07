% NORTH30 Network Topology Visualization
% 2D visualization of bus connections with branch numbers

clear all;
clc;

fprintf('=== NORTH30 NETWORK TOPOLOGY VISUALIZATION ===\n');

%% Load System Data
mpc = north30_matpower();
branch_data = readtable('NORTH30_OffPeak_Branch.csv');
bus_data = readtable('NORTH30_OffPeak_Bus.csv');

% Extract network topology
buses = mpc.bus(:, 1);              % Bus numbers
from_buses = mpc.branch(:, 1);      % From bus
to_buses = mpc.branch(:, 2);        % To bus
n_buses = length(buses);
n_branches = length(from_buses);

fprintf('System Information:\n');
fprintf('- Buses: %d\n', n_buses);
fprintf('- Branches: %d\n', n_branches);

%% Create Graph Object for Layout
% Create adjacency matrix
adj_matrix = zeros(n_buses, n_buses);
for i = 1:n_branches
    from_idx = find(buses == from_buses(i));
    to_idx = find(buses == to_buses(i));
    if ~isempty(from_idx) && ~isempty(to_idx)
        adj_matrix(from_idx, to_idx) = 1;
        adj_matrix(to_idx, from_idx) = 1;  % Undirected graph
    end
end

% Create graph object for automatic layout
G = graph(adj_matrix);

%% Generate Node Positions
% Use force-directed layout for better visualization
fprintf('Generating network layout...\n');

% Try different layout algorithms
layout_method = 'force';  % Options: 'force', 'circle', 'layered'

switch layout_method
    case 'force'
        % Force-directed layout (good for general networks)
        [x_pos, y_pos] = layout(G, 'force', 'UseGravity', true);
    case 'circle'
        % Circular layout
        [x_pos, y_pos] = layout(G, 'circle');
    case 'layered'
        % Layered layout (good for hierarchical networks)
        [x_pos, y_pos] = layout(G, 'layered');
end

% Scale and adjust positions for better visualization
x_pos = x_pos * 10;  % Scale up
y_pos = y_pos * 10;

%% Classify Buses by Function
load_buses = find(mpc.bus(:, 3) > 0);           % Buses with load
gen_buses = mpc.gen(:, 1);                      % Generator buses
slack_bus = find(mpc.bus(:, 2) == 3);           % Slack bus

fprintf('Bus Classification:\n');
fprintf('- Load buses: %d\n', length(load_buses));
fprintf('- Generator buses: %d\n', length(gen_buses));
fprintf('- Slack bus: %d\n', length(slack_bus));

%% Create Comprehensive Network Visualization
figure('Position', [50, 50, 1600, 1200]);

% Main network plot
subplot(2, 2, [1, 2]);
hold on;

% Plot branches first (so they appear behind nodes)
for i = 1:n_branches
    from_idx = find(buses == from_buses(i));
    to_idx = find(buses == to_buses(i));
    
    if ~isempty(from_idx) && ~isempty(to_idx)
        % Get branch power flow for line thickness
        branch_power = abs(branch_data.P_f(i));
        if isnan(branch_power)
            branch_power = 0;
        end
        
        % Line thickness based on power flow (1-5 range)
        line_width = max(1, min(5, branch_power / 50));
        
        % Color based on power level
        if branch_power > 100
            line_color = [1, 0, 0];      % Red for heavy flow
        elseif branch_power > 50
            line_color = [1, 0.5, 0];    % Orange for medium flow
        else
            line_color = [0, 0, 1];      % Blue for light flow
        end
        
        % Draw branch line
        plot([x_pos(from_idx), x_pos(to_idx)], [y_pos(from_idx), y_pos(to_idx)], ...
             'Color', line_color, 'LineWidth', line_width);
        
        % Add branch number at midpoint
        mid_x = (x_pos(from_idx) + x_pos(to_idx)) / 2;
        mid_y = (y_pos(from_idx) + y_pos(to_idx)) / 2;
        
        % Add small offset to avoid overlap
        offset_x = 0.2 * (rand - 0.5);
        offset_y = 0.2 * (rand - 0.5);
        
        text(mid_x + offset_x, mid_y + offset_y, sprintf('%d', i), ...
             'FontSize', 8, 'BackgroundColor', 'white', 'EdgeColor', 'black', ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
end

% Plot buses with different symbols and colors
for i = 1:n_buses
    bus_num = buses(i);
    
    % Determine bus type and corresponding visual style
    if any(slack_bus == bus_num)
        % Slack bus - large red square
        scatter(x_pos(i), y_pos(i), 200, 'rs', 'filled', 'MarkerEdgeColor', 'black', 'LineWidth', 2);
        bus_label_color = 'red';
    elseif any(gen_buses == bus_num)
        % Generator bus - green circle
        scatter(x_pos(i), y_pos(i), 150, 'go', 'filled', 'MarkerEdgeColor', 'black', 'LineWidth', 1.5);
        bus_label_color = 'green';
    elseif any(load_buses == bus_num)
        % Load bus - blue triangle
        scatter(x_pos(i), y_pos(i), 120, 'b^', 'filled', 'MarkerEdgeColor', 'black', 'LineWidth', 1);
        bus_label_color = 'blue';
    else
        % Transit bus - gray circle
        scatter(x_pos(i), y_pos(i), 100, 'ko', 'filled', 'MarkerFaceColor', [0.7, 0.7, 0.7], 'MarkerEdgeColor', 'black');
        bus_label_color = 'black';
    end
    
    % Add bus number label
    text(x_pos(i), y_pos(i) + 0.8, sprintf('B%d', bus_num), ...
         'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', ...
         'Color', bus_label_color, 'BackgroundColor', 'white', 'EdgeColor', 'gray');
end

% Formatting
grid on;
axis equal;
title('NORTH30 Network Topology with Branch Numbers', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X Position');
ylabel('Y Position');

% Legend
legend_entries = {
    'Slack Bus (Red Square)', 
    'Generator Bus (Green Circle)', 
    'Load Bus (Blue Triangle)', 
    'Transit Bus (Gray Circle)'
};
legend(legend_entries, 'Location', 'best', 'FontSize', 10);

hold off;

%% Bus Details Subplot
subplot(2, 2, 3);
axis off;

% Create bus information table
bus_info_text = {
    'BUS INFORMATION';
    '===============';
    '';
    'BUS TYPES:';
    sprintf('Slack Bus: %d', slack_bus);
    sprintf('Generator Buses: %s', num2str(gen_buses'));
    sprintf('Load Buses: %s', num2str(load_buses'));
    '';
    'POWER DATA:';
    sprintf('Total Generation: %.1f MW', sum(mpc.gen(:,2)));
    sprintf('Total Load: %.1f MW', sum(mpc.bus(:,3)));
    '';
    'VOLTAGE LEVELS:';
    sprintf('275 kV buses: %d', sum(mpc.bus(:,10) == 275));
    sprintf('187 kV buses: %d', sum(mpc.bus(:,10) == 187));
    sprintf('16.5 kV buses: %d', sum(mpc.bus(:,10) == 16.5));
};

text(0.05, 0.95, bus_info_text, 'FontSize', 10, 'FontName', 'Courier', ...
     'VerticalAlignment', 'top', 'Units', 'normalized');

%% Branch Details Subplot
subplot(2, 2, 4);
axis off;

% Branch statistics
high_flow_branches = sum(abs(branch_data.P_f) > 100);
medium_flow_branches = sum(abs(branch_data.P_f) > 50 & abs(branch_data.P_f) <= 100);
low_flow_branches = sum(abs(branch_data.P_f) <= 50);

branch_info_text = {
    'BRANCH INFORMATION';
    '==================';
    '';
    'FLOW CLASSIFICATION:';
    sprintf('Heavy Flow (>100MW): %d branches', high_flow_branches);
    sprintf('Medium Flow (50-100MW): %d branches', medium_flow_branches);
    sprintf('Light Flow (<50MW): %d branches', low_flow_branches);
    '';
    'LINE COLORS:';
    'Red: Heavy flow (>100 MW)';
    'Orange: Medium flow (50-100 MW)';
    'Blue: Light flow (<50 MW)';
    '';
    'LINE THICKNESS:';
    'Proportional to power flow';
    'Thicker = Higher power flow';
    '';
    'BRANCH NUMBERS:';
    'Displayed at line midpoints';
    'White background for visibility';
};

text(0.05, 0.95, branch_info_text, 'FontSize', 10, 'FontName', 'Courier', ...
     'VerticalAlignment', 'top', 'Units', 'normalized');

%% Overall Title
sgtitle('NORTH30 System: Network Topology and Connection Analysis', ...
        'FontSize', 16, 'FontWeight', 'bold');

%% Create Simplified Connection Matrix Visualization
figure('Position', [100, 100, 1000, 800]);

% Create connection matrix for visualization
connection_matrix = zeros(max(buses), max(buses));
for i = 1:n_branches
    from_bus = from_buses(i);
    to_bus = to_buses(i);
    power_flow = abs(branch_data.P_f(i));
    if isnan(power_flow)
        power_flow = 0;
    end
    connection_matrix(from_bus, to_bus) = power_flow;
    connection_matrix(to_bus, from_bus) = power_flow;  % Symmetric
end

% Plot connection matrix
imagesc(connection_matrix);
colorbar;
colormap('hot');
title('Bus Connection Matrix (Power Flow Magnitude)', 'FontSize', 14);
xlabel('To Bus');
ylabel('From Bus');

% Add grid
grid on;
set(gca, 'GridColor', 'white', 'GridAlpha', 0.5);

% Add text annotations for non-zero connections
[row_idx, col_idx] = find(connection_matrix > 0);
for i = 1:length(row_idx)
    if row_idx(i) < col_idx(i)  % Avoid duplicate labels
        power_val = connection_matrix(row_idx(i), col_idx(i));
        if power_val > 20  % Only label significant flows
            text(col_idx(i), row_idx(i), sprintf('%.0f', power_val), ...
                 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 8);
        end
    end
end

%% Save Results
print(gcf, 'north30_network_topology.png', '-dpng', '-r300');

% Save the first figure too
figure(1);
print(gcf, 'north30_network_detailed.png', '-dpng', '-r300');

% Save data for further analysis
save('network_topology_data.mat', 'x_pos', 'y_pos', 'buses', 'from_buses', 'to_buses', ...
     'load_buses', 'gen_buses', 'slack_bus', 'connection_matrix');

fprintf('\n=== VISUALIZATION COMPLETED ===\n');
fprintf('Files generated:\n');
fprintf('- north30_network_detailed.png (Detailed topology with branch numbers)\n');
fprintf('- north30_network_topology.png (Connection matrix)\n');
fprintf('- network_topology_data.mat (Position and connection data)\n');

fprintf('\nNetwork Characteristics:\n');
fprintf('- Largest connected component: %d buses\n', n_buses);
fprintf('- Network density: %.3f\n', 2*n_branches/(n_buses*(n_buses-1)));
fprintf('- Average node degree: %.1f\n', 2*n_branches/n_buses);
fprintf('- Heavily loaded branches: %d\n', high_flow_branches);