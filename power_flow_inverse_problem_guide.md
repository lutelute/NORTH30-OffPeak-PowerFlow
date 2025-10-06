# 電力系統における潮流計算の逆問題解説

## 1. はじめに

電力系統の潮流計算は、一般的に「順問題」として扱われますが、実際の系統運用では「逆問題」として取り組むことが多くあります。本解説書では、NORTH30軽負荷断面データを例に、潮流計算における逆問題の概念と解決手法について説明します。

## 2. 順問題と逆問題の定義

### 2.1 順問題（Forward Problem）
- **入力**: 系統構成、発電機出力、負荷
- **出力**: 各母線の電圧、各線路の潮流
- **目的**: 与えられた運転状態での系統状態を計算

### 2.2 逆問題（Inverse Problem）
- **入力**: 観測された電圧、潮流データ
- **出力**: 発電機出力、負荷、系統パラメータ
- **目的**: 観測データから系統の運転状態や特性を推定

## 3. 電力系統における逆問題の種類

### 3.1 状態推定問題
```
観測データ → 系統の真の状態
- 電圧振幅・位相角の推定
- 負荷の推定
- 発電機出力の推定
```

### 3.2 パラメータ推定問題
```
運転データ → 系統パラメータ
- 線路インピーダンスの推定
- 変圧器タップ比の推定
- 負荷特性の推定
```

### 3.3 トポロジ推定問題
```
観測データ → 系統構成
- 開閉器状態の推定
- 線路の接続状態の推定
```

## 4. NORTH30データでの逆問題解析

### 4.1 問題設定
NORTH30軽負荷断面では以下のデータが与えられています：
- 母線電圧（振幅・位相角）
- 発電機出力（有効・無効電力）
- 負荷（有効・無効電力）
- 線路潮流（有効・無効電力）

これらのデータから、**系統の整合性検証**と**パラメータ検証**を行う逆問題として捉えることができます。

### 4.2 数学的定式化

#### 潮流方程式（順問題）
```
P_i = V_i Σ V_j (G_ij cos θ_ij + B_ij sin θ_ij)
Q_i = V_i Σ V_j (G_ij sin θ_ij - B_ij cos θ_ij)
```

#### 逆問題（最小二乗問題）
```
minimize: Σ (P_calc - P_obs)² + Σ (Q_calc - Q_obs)²
subject to: 潮流方程式制約
```

### 4.3 実装例

```matlab
% 逆問題による系統パラメータ検証
function [estimated_params, rmse] = solve_inverse_problem(observed_data)
    % 初期パラメータ設定
    x0 = initial_parameters();
    
    % 目的関数：観測値と計算値の差の二乗和
    objective = @(x) calculate_residual(x, observed_data);
    
    % 制約条件
    constraints = setup_constraints();
    
    % 最適化実行
    [estimated_params, fval] = fmincon(objective, x0, [], [], [], [], ...
                                      lb, ub, constraints);
    
    % RMSE計算
    rmse = sqrt(fval / length(observed_data));
end
```

## 5. 逆問題の解法手法

### 5.1 最小二乗法
- **特徴**: 線形近似による高速解法
- **適用**: パラメータ推定、状態推定
- **利点**: 計算が高速、理論が確立
- **欠点**: 非線形性の取り扱いが困難

### 5.2 加重最小二乗法（WLS）
```matlab
J = Σ w_i (z_i - h_i(x))²
```
- 観測精度に応じた重み付け
- 測定誤差の分散を考慮

### 5.3 非線形最適化
- **手法**: Newton-Raphson法、Gauss-Newton法
- **特徴**: 非線形潮流方程式を直接扱える
- **課題**: 初期値依存性、収束性

### 5.4 確率的手法
- **手法**: カルマンフィルタ、粒子フィルタ
- **特徴**: 動的推定、不確実性の定量化
- **応用**: リアルタイム状態推定

## 6. NORTH30での逆問題解析手順

### 6.1 データ前処理
```matlab
% 1. 観測データの整理
observed_voltages = extract_bus_voltages(csv_data);
observed_flows = extract_branch_flows(csv_data);
observed_injections = extract_power_injections(csv_data);

% 2. データ品質チェック
quality_flags = check_data_quality(observed_data);
```

### 6.2 問題定式化
```matlab
% 3. 推定対象の設定
estimation_variables = {
    'generator_outputs',    % 発電機出力
    'load_values',         % 負荷値  
    'line_parameters'      % 線路パラメータ
};

% 4. 制約条件の設定
constraints = {
    'power_balance',       % 電力バランス
    'voltage_limits',      % 電圧制約
    'generation_limits'    % 発電制約
};
```

### 6.3 解法実行
```matlab
% 5. 最適化問題の解法
[solution, diagnostics] = solve_estimation_problem(observed_data, ...
                                                   estimation_variables, ...
                                                   constraints);
```

## 7. 解の評価と検証

### 7.1 適合度評価
```matlab
% RMSE計算
rmse_voltage = sqrt(mean((V_calc - V_obs).^2));
rmse_power = sqrt(mean((P_calc - P_obs).^2));

% 相対誤差
relative_error = abs(calc_values - obs_values) ./ abs(obs_values) * 100;
```

### 7.2 統計的検定
- **カイ二乗検定**: 残差の統計的有意性
- **正規性検定**: 誤差分布の正規性
- **外れ値検出**: 不良データの特定

### 7.3 感度解析
```matlab
% パラメータ感度の評価
sensitivity_matrix = calculate_sensitivity(solution, parameters);
condition_number = cond(sensitivity_matrix);
```

## 8. 逆問題の課題と対策

### 8.1 非一意性問題
- **課題**: 複数の解が存在する可能性
- **対策**: 
  - 事前情報の活用
  - 正則化項の追加
  - 物理制約の強化

### 8.2 不適切性（Ill-posedness）
- **課題**: 小さな観測誤差が大きな推定誤差を引き起こす
- **対策**:
  - Tikhonov正則化
  - L1正則化（スパース推定）
  - ベイズ推定

### 8.3 計算複雑性
- **課題**: 大規模系統での計算時間
- **対策**:
  - 分散最適化
  - 近似手法の活用
  - 並列計算

## 9. NORTH30での具体的応用例

### 9.1 負荷推定
```matlab
% 観測された電圧・潮流から負荷を推定
estimated_loads = estimate_loads_from_measurements(voltage_data, flow_data);
```

### 9.2 発電機出力検証
```matlab
% 発電機出力の妥当性検証
[is_feasible, violations] = verify_generation_dispatch(gen_outputs, system_data);
```

### 9.3 線路パラメータ同定
```matlab
% 観測データから線路インピーダンスを推定
estimated_impedances = identify_line_parameters(measurement_data);
```

## 10. 実装上の注意点

### 10.1 数値安定性
- 行列の条件数管理
- スケーリングの適切な実施
- 収束判定基準の設定

### 10.2 初期値設定
- 物理的に妥当な初期値
- 複数の初期値からの試行
- 事前情報の活用

### 10.3 制約条件
- 等式制約（電力バランス）
- 不等式制約（運転制約）
- 整数制約（トポロジ）

## 11. まとめ

電力系統における逆問題は、実際の系統運用において重要な役割を果たします。NORTH30軽負荷断面データを用いた解析では、以下の点が重要です：

1. **問題の適切な定式化**
2. **観測データの品質管理**
3. **解法の選択と実装**
4. **結果の妥当性検証**
5. **不確実性の定量化**

本解説書で示した手法を参考に、実際の系統データに対する逆問題解析を実施することで、系統の理解を深め、運用の最適化に貢献できます。

## 参考文献

1. Abur, A. and Expósito, A.G., "Power System State Estimation: Theory and Implementation"
2. Monticelli, A., "State Estimation in Electric Power Systems"
3. IEEE Tutorial on Power System State Estimation
4. MATPOWER Documentation and User's Manual

---
*本解説書は、NORTH30軽負荷断面データを用いた潮流計算の逆問題解析のために作成されました。*