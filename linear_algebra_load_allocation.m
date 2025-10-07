% Linear Algebra Based Load Allocation (Direct Solution)
% 線形代数による負荷配分の直接解法 - 最適化不要
% 単純な四則演算と行列計算のみで逆問題を解く

clear all;
clc;

fprintf('=== 線形代数による負荷配分直接解法 ===\n');
fprintf('Method: Direct matrix solution (no optimization)\n');
fprintf('Theory: P = H * L (Power flow = Sensitivity * Load)\n\n');

%% STEP 1: システムデータの読み込み
mpc = north30_matpower();
branch_data = readtable('NORTH30_OffPeak_Branch.csv');

% システム情報
n_buses = size(mpc.bus, 1);
n_branches = size(mpc.branch, 1);
load_buses = find(mpc.bus(:, 3) ~= 0);
n_load_buses = length(load_buses);

fprintf('System Information:\n');
fprintf('- Total buses: %d\n', n_buses);
fprintf('- Load buses: %d\n', n_load_buses);
fprintf('- Branches: %d\n', n_branches);

%% STEP 2: 感度行列の計算 (DC Power Flow Sensitivity Matrix)
fprintf('\nSTEP 2: 感度行列計算中...\n');

% DC潮流のための基本行列を作成
B_bus = zeros(n_buses, n_buses);
B_branch = zeros(n_branches, n_buses);

% バス・ブランチ接続行列
A = zeros(n_branches, n_buses);
for i = 1:n_branches
    from_bus = mpc.branch(i, 1);
    to_bus = mpc.branch(i, 2);
    from_idx = find(mpc.bus(:,1) == from_bus);
    to_idx = find(mpc.bus(:,1) == to_bus);
    
    if ~isempty(from_idx) && ~isempty(to_idx)
        A(i, from_idx) = 1;
        A(i, to_idx) = -1;
    end
end

% サセプタンス行列の構築
for i = 1:n_branches
    from_bus = mpc.branch(i, 1);
    to_bus = mpc.branch(i, 2);
    reactance = mpc.branch(i, 4);  % X値
    
    from_idx = find(mpc.bus(:,1) == from_bus);
    to_idx = find(mpc.bus(:,1) == to_bus);
    
    if ~isempty(from_idx) && ~isempty(to_idx) && reactance > 0
        susceptance = 1 / reactance;
        
        % B行列の構築
        B_bus(from_idx, from_idx) = B_bus(from_idx, from_idx) + susceptance;
        B_bus(to_idx, to_idx) = B_bus(to_idx, to_idx) + susceptance;
        B_bus(from_idx, to_idx) = B_bus(from_idx, to_idx) - susceptance;
        B_bus(to_idx, from_idx) = B_bus(to_idx, from_idx) - susceptance;
        
        % ブランチ・バス感度行列
        B_branch(i, from_idx) = susceptance;
        B_branch(i, to_idx) = -susceptance;
    end
end

% スラックバスを除去したB行列
slack_bus_idx = find(mpc.bus(:, 2) == 3);
if isempty(slack_bus_idx)
    slack_bus_idx = 1;  % デフォルトでバス1をスラック
end

non_slack_buses = setdiff(1:n_buses, slack_bus_idx);
B_reduced = B_bus(non_slack_buses, non_slack_buses);

fprintf('✓ 感度行列計算完了\n');
fprintf('- B行列サイズ: %dx%d\n', size(B_reduced,1), size(B_reduced,2));
fprintf('- 最小固有値: %.6f\n', min(real(eig(B_reduced))));

%% STEP 3: 潮流・負荷感度行列の構築
fprintf('\nSTEP 3: 潮流-負荷感度行列構築中...\n');

% 負荷バスのみに対応するマッピング行列
L_map = zeros(length(non_slack_buses), n_load_buses);
for i = 1:n_load_buses
    load_bus_num = load_buses(i);
    bus_idx = find(mpc.bus(:,1) == load_bus_num);
    reduced_idx = find(non_slack_buses == bus_idx);
    
    if ~isempty(reduced_idx)
        L_map(reduced_idx, i) = 1;
    end
end

% 感度行列: H = B_branch * inv(B_reduced) * L_map
% P_flow = H * P_load の関係
try
    B_inv = inv(B_reduced);
    fprintf('✓ B行列の逆行列計算成功\n');
catch
    fprintf('⚠ B行列が特異行列のため擬似逆行列を使用\n');
    B_inv = pinv(B_reduced);
end

% 感度行列H: [n_branches x n_load_buses]
% H(i,j) = ブランチiの潮流に対する負荷バスjの負荷の感度
H_full = zeros(n_branches, n_load_buses);

for i = 1:n_branches
    % ブランチiの潮流感度
    branch_sens = B_branch(i, non_slack_buses) * B_inv * L_map;
    H_full(i, :) = branch_sens;
end

fprintf('✓ 感度行列H構築完了: %dx%d\n', size(H_full,1), size(H_full,2));

%% STEP 4: 観測可能な潮流データの選択とマッピング
fprintf('\nSTEP 4: 観測データ選択中...\n');

% データの整合性チェック
fprintf('データ整合性チェック:\n');
fprintf('- MATLABケース ブランチ数: %d\n', n_branches);
fprintf('- CSVデータ ブランチ数: %d\n', length(branch_data.P_f));

% MATLABケースとCSVデータのブランチ対応付け
branch_mapping = [];
observed_flows_mapped = [];
H_mapped = [];

for i = 1:length(branch_data.P_f)
    if isnan(branch_data.P_f(i))
        continue;
    end
    
    % CSVのfrom-to バス番号
    csv_from = branch_data.fbus(i);
    csv_to = branch_data.tbus(i);
    
    % MATLABケースで対応するブランチを探す
    matlab_idx = [];
    for j = 1:n_branches
        matlab_from = mpc.branch(j, 1);
        matlab_to = mpc.branch(j, 2);
        
        if (matlab_from == csv_from && matlab_to == csv_to) || ...
           (matlab_from == csv_to && matlab_to == csv_from)
            matlab_idx = j;
            break;
        end
    end
    
    % 対応が見つかった場合
    if ~isempty(matlab_idx) && abs(branch_data.P_f(i)) > 5
        branch_mapping = [branch_mapping; matlab_idx];
        observed_flows_mapped = [observed_flows_mapped; branch_data.P_f(i)];
        H_mapped = [H_mapped; H_full(matlab_idx, :)];
    end
end

fprintf('✓ ブランチマッピング完了\n');
fprintf('- 対応付けられたブランチ数: %d\n', length(branch_mapping));
fprintf('- 有意な潮流を持つブランチ: %d\n', length(observed_flows_mapped));

% マッピングされたデータを使用
P_observed = observed_flows_mapped;
H_selected = H_mapped;

m = length(P_observed);  % 観測方程式数
n = n_load_buses;        % 未知負荷数

fprintf('✓ 観測データ選択完了\n');
fprintf('- 観測方程式数 (m): %d\n', m);
fprintf('- 未知数 (n): %d\n', n);

% データが十分にあるかチェック
if m == 0
    error('観測データが不足しています。ブランチデータを確認してください。');
end

if isempty(H_selected)
    error('感度行列が空です。データマッピングを確認してください。');
end

fprintf('- 感度行列サイズ: %dx%d\n', size(H_selected,1), size(H_selected,2));
fprintf('- 条件数: %.2e\n', cond(H_selected));

%% STEP 5: 線形代数による直接解法
fprintf('\nSTEP 5: 線形代数による解法実行中...\n');

% 問題: P_observed = H_selected * L_estimated
% 解: L_estimated = (H_selected)^+ * P_observed

if m >= n
    fprintf('過決定系 (m≥n): 最小二乗解を計算\n');
    % 最小二乗解: L = (H^T * H)^(-1) * H^T * P
    try
        L_estimated = (H_selected' * H_selected) \ (H_selected' * P_observed);
        solution_method = 'Normal Equations';
    catch
        L_estimated = pinv(H_selected) * P_observed;
        solution_method = 'Pseudo-inverse';
    end
elseif m < n
    fprintf('劣決定系 (m<n): 最小ノルム解を計算\n');
    % 最小ノルム解
    L_estimated = pinv(H_selected) * P_observed;
    solution_method = 'Minimum Norm Solution';
else
    fprintf('丁度決定系 (m=n): 直接解を計算\n');
    try
        L_estimated = H_selected \ P_observed;
        solution_method = 'Direct Solution';
    catch
        L_estimated = pinv(H_selected) * P_observed;
        solution_method = 'Pseudo-inverse (singular)';
    end
end

fprintf('✓ 解法完了: %s\n', solution_method);

%% STEP 6: 結果の検証
fprintf('\nSTEP 6: 結果検証中...\n');

% 推定潮流の計算
P_estimated = H_selected * L_estimated;

% 誤差の計算
flow_errors = P_estimated - P_observed;
rmse_flow = sqrt(mean(flow_errors.^2));
mae_flow = mean(abs(flow_errors));
max_error = max(abs(flow_errors));

% 真値との比較（負荷バス）
true_loads = zeros(n_load_buses, 1);
for i = 1:n_load_buses
    bus_idx = find(mpc.bus(:,1) == load_buses(i));
    true_loads(i) = mpc.bus(bus_idx, 3);
end

load_errors = L_estimated - true_loads;
rmse_load = sqrt(mean(load_errors.^2));
mae_load = mean(abs(load_errors));

% 制約チェック: 負荷バランス
total_estimated_load = sum(L_estimated);
total_true_load = sum(true_loads);
total_generation = sum(mpc.gen(:,2));
balance_error = abs(total_estimated_load - total_true_load);

fprintf('✓ 検証完了\n');

%% STEP 7: 結果表示
fprintf('\n=== 線形代数解法結果 ===\n');
fprintf('解法: %s\n', solution_method);
if m > n
    system_type = '過決定';
elseif m < n
    system_type = '劣決定';
else
    system_type = '丁度決定';
end
fprintf('システム: %s (%d x %d)\n', system_type, m, n);

fprintf('\n潮流マッチング精度:\n');
fprintf('RMSE: %.2f MW\n', rmse_flow);
fprintf('MAE:  %.2f MW\n', mae_flow);
fprintf('最大誤差: %.2f MW\n', max_error);
fprintf('相対RMSE: %.1f%%\n', (rmse_flow/mean(abs(P_observed)))*100);

fprintf('\n負荷推定精度:\n');
fprintf('RMSE: %.2f MW\n', rmse_load);
fprintf('MAE:  %.2f MW\n', mae_load);
fprintf('負荷バランス誤差: %.2f MW\n', balance_error);

fprintf('\n負荷推定詳細:\n');
fprintf('Bus | True Load | Estimated | Error   | Error%%\n');
fprintf('----|-----------|-----------|---------|--------\n');

for i = 1:min(10, n_load_buses)  % 最初の10バスを表示
    bus_num = load_buses(i);
    true_load = true_loads(i);
    est_load = L_estimated(i);
    error = est_load - true_load;
    
    if abs(true_load) > 0.1
        error_pct = (error / true_load) * 100;
    else
        error_pct = 0;
    end
    
    fprintf('%3d | %9.1f | %9.1f | %7.1f | %6.1f%%\n', ...
            bus_num, true_load, est_load, error, error_pct);
end

if n_load_buses > 10
    fprintf('... (%d buses total)\n', n_load_buses);
end

%% STEP 8: 可視化
fprintf('\n可視化作成中...\n');

figure('Position', [100, 100, 1600, 1000]);

% サブプロット1: 負荷比較
subplot(2,3,1);
bar([true_loads(1:min(15,n_load_buses)), L_estimated(1:min(15,n_load_buses))]);
xlabel('Load Bus Index');
ylabel('Load (MW)');
title('Load Allocation: True vs Estimated (Linear Algebra)');
legend('True', 'Estimated', 'Location', 'best');
grid on;

% サブプロット2: 負荷相関
subplot(2,3,2);
scatter(true_loads, L_estimated, 60, 'filled');
hold on;
min_val = min([true_loads; L_estimated]);
max_val = max([true_loads; L_estimated]);
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
xlabel('True Load (MW)');
ylabel('Estimated Load (MW)');
title('Load Correlation');
grid on;
axis equal;

% 相関係数
R = corrcoef(true_loads, L_estimated);
r_squared = R(1,2)^2;
text(0.05, 0.95, sprintf('R² = %.3f', r_squared), 'Units', 'normalized', ...
     'BackgroundColor', 'white');

% サブプロット3: 潮流マッチング
subplot(2,3,3);
scatter(P_observed, P_estimated, 60, 'filled');
hold on;
min_val = min([P_observed; P_estimated]);
max_val = max([P_observed; P_estimated]);
plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 2);
xlabel('Observed Flow (MW)');
ylabel('Estimated Flow (MW)');
title('Flow Matching');
grid on;
axis equal;

% サブプロット4: 誤差分布
subplot(2,3,4);
histogram(flow_errors, min(15, length(flow_errors)));
xlabel('Flow Error (MW)');
ylabel('Frequency');
title('Flow Error Distribution');
grid on;

% サブプロット5: 感度行列の可視化
subplot(2,3,5);
imagesc(H_selected);
colorbar;
xlabel('Load Bus Index');
ylabel('Branch Index');
title('Sensitivity Matrix H');

% サブプロット6: 統計情報
subplot(2,3,6);
axis off;

stats_text = {
    'LINEAR ALGEBRA SOLUTION';
    '====================';
    sprintf('Method: %s', solution_method);
    sprintf('System: %dx%d (%s)', m, n, system_type);
    '';
    'ACCURACY:';
    sprintf('Flow RMSE: %.1f MW', rmse_flow);
    sprintf('Load RMSE: %.1f MW', rmse_load);
    sprintf('Load R²: %.3f', r_squared);
    '';
    'MATRIX PROPERTIES:';
    sprintf('Condition Number: %.1e', cond(H_selected));
    sprintf('Rank: %d', rank(H_selected));
    '';
    'COMPUTATION:';
    'No optimization required';
    'Direct matrix solution';
    'Fast and deterministic';
};

text(0.1, 0.9, stats_text, 'FontSize', 11, 'FontName', 'Courier', ...
     'VerticalAlignment', 'top');

sgtitle('Linear Algebra Based Load Allocation (Direct Solution)', 'FontSize', 14);

%% STEP 9: 結果の保存
save('linear_algebra_results.mat', 'L_estimated', 'true_loads', 'H_selected', ...
     'P_observed', 'P_estimated', 'rmse_flow', 'rmse_load', 'r_squared', 'solution_method');

print(gcf, 'linear_algebra_load_allocation.png', '-dpng', '-r300');

fprintf('✓ 可視化完了\n');
fprintf('\n=== 解法比較 ===\n');
fprintf('線形代数解法の特徴:\n');
fprintf('✓ 最適化不要 - 直接的な行列計算\n');
fprintf('✓ 高速 - O(n³)の計算量\n');
fprintf('✓ 決定論的 - 常に同じ解\n');
fprintf('✓ 理論的根拠 - 線形システム理論\n');
fprintf('✓ 実装簡単 - 基本的な線形代数のみ\n');

fprintf('\n制約:\n');
fprintf('- 線形近似(DC潮流)が前提\n');
fprintf('- 観測点数と未知数の関係に依存\n');
fprintf('- 不等式制約は直接扱えない\n');

fprintf('\nファイル出力:\n');
fprintf('- linear_algebra_results.mat\n');
fprintf('- linear_algebra_load_allocation.png\n');

fprintf('\n=== 完了 ===\n');