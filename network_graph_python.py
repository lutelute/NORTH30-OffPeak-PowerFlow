#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NORTH30 Network Graph Visualization using Python
ネットワークグラフの可視化 - NetworkX + Matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

print("=== NORTH30 ネットワークグラフ可視化 (Python) ===")

# データの読み込み
print("データ読み込み中...")
try:
    # エンコーディングを指定してCSVを読み込み
    data_path = '/Users/shigenoburyuto/Documents/GitHub/North_sys_/NORTH30_OffPeak_データ一式/'
    bus_data = pd.read_csv(data_path + 'NORTH30_OffPeak_Bus.csv', encoding='shift_jis')
    branch_data = pd.read_csv(data_path + 'NORTH30_OffPeak_Branch.csv', encoding='shift_jis')
    gen_data = pd.read_csv(data_path + 'NORTH30_OffPeak_Gen.csv', encoding='shift_jis')
    print(f"✓ バスデータ: {len(bus_data)} buses")
    print(f"✓ ブランチデータ: {len(branch_data)} branches")
    print(f"✓ 発電機データ: {len(gen_data)} generators")
except Exception as e:
    print(f"✗ データ読み込みエラー: {e}")
    exit(1)

# NetworkXグラフの作成
print("\nNetworkXグラフ構築中...")
G = nx.Graph()

# ノード(バス)の追加
for _, bus in bus_data.iterrows():
    if pd.isna(bus['bus_i']):
        continue
    bus_num = int(bus['bus_i'])
    bus_type = int(bus['type'])
    load = bus['Pd'] if not pd.isna(bus['Pd']) else 0
    voltage = bus['baseKV'] if not pd.isna(bus['baseKV']) else 275
    
    # バスタイプの判定
    if bus_type == 3:
        node_type = 'slack'
    elif bus_num in gen_data['bus'].values:
        node_type = 'generator'
    elif load > 0:
        node_type = 'load'
    else:
        node_type = 'transit'
    
    G.add_node(bus_num, 
               type=node_type,
               load=load,
               voltage=voltage,
               bus_type=bus_type)

print(f"✓ ノード追加完了: {G.number_of_nodes()} nodes")

# エッジ(ブランチ)の追加
for _, branch in branch_data.iterrows():
    if pd.isna(branch['fbus']) or pd.isna(branch['tbus']):
        continue
    from_bus = int(branch['fbus'])
    to_bus = int(branch['tbus'])
    power_flow = abs(branch['P_f']) if not pd.isna(branch['P_f']) else 0
    
    # 抵抗とリアクタンス
    resistance = branch['r'] if 'r' in branch and not pd.isna(branch['r']) else 0.01
    reactance = branch['x'] if 'x' in branch and not pd.isna(branch['x']) else 0.1
    
    G.add_edge(from_bus, to_bus,
               power_flow=power_flow,
               resistance=resistance,
               reactance=reactance)

print(f"✓ エッジ追加完了: {G.number_of_edges()} edges")

# レイアウトの計算
print("\nレイアウト計算中...")
# 複数のレイアウトアルゴリズムを試す
try:
    # Spring layoutを基本として使用
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    layout_name = "Spring Layout"
except:
    # フォールバック: circular layout
    pos = nx.circular_layout(G)
    layout_name = "Circular Layout"

print(f"✓ レイアウト完了: {layout_name}")

# 可視化の設定
print("\n可視化準備中...")
plt.style.use('default')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('NORTH30 電力系統ネットワーク可視化 (Python NetworkX)', fontsize=16, fontweight='bold')

# === サブプロット1: 基本ネットワーク ===
ax1.set_title('基本ネットワーク構成', fontsize=14, fontweight='bold')

# エッジ(送電線)の描画 - 電力フローに応じて太さと色を変更
edge_weights = []
edge_colors = []
for (u, v, d) in G.edges(data=True):
    power = d['power_flow']
    edge_weights.append(max(0.5, min(8, power/25)))  # 線の太さ
    
    if power > 100:
        edge_colors.append('red')      # 重負荷
    elif power > 50:
        edge_colors.append('orange')   # 中負荷
    else:
        edge_colors.append('blue')     # 軽負荷

nx.draw_networkx_edges(G, pos, ax=ax1, 
                      width=edge_weights, 
                      edge_color=edge_colors, 
                      alpha=0.7)

# ノード(バス)の描画 - タイプ別に色分け
node_colors = []
node_sizes = []
for node in G.nodes():
    node_type = G.nodes[node]['type']
    if node_type == 'slack':
        node_colors.append('red')
        node_sizes.append(800)
    elif node_type == 'generator':
        node_colors.append('green')
        node_sizes.append(600)
    elif node_type == 'load':
        node_colors.append('blue')
        node_sizes.append(400)
    else:
        node_colors.append('gray')
        node_sizes.append(300)

nx.draw_networkx_nodes(G, pos, ax=ax1,
                      node_color=node_colors,
                      node_size=node_sizes,
                      alpha=0.8)

# ノードラベル
labels = {node: f'B{node}' for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=8, font_weight='bold')

ax1.set_aspect('equal')
ax1.axis('off')

# 凡例
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=15, label='スラックバス'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12, label='発電バス'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='負荷バス'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='中継バス'),
    plt.Line2D([0], [0], color='red', linewidth=3, label='重負荷線路(>100MW)'),
    plt.Line2D([0], [0], color='orange', linewidth=3, label='中負荷線路(50-100MW)'),
    plt.Line2D([0], [0], color='blue', linewidth=3, label='軽負荷線路(<50MW)')
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)

# === サブプロット2: 電力フロー強調表示 ===
ax2.set_title('電力フロー強調表示', fontsize=14, fontweight='bold')

# 高負荷ブランチのみを強調
high_flow_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['power_flow'] > 50]
medium_flow_edges = [(u, v) for (u, v, d) in G.edges(data=True) if 20 <= d['power_flow'] <= 50]
low_flow_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['power_flow'] < 20]

# エッジを段階的に描画
nx.draw_networkx_edges(G, pos, edgelist=low_flow_edges, ax=ax2, 
                      width=1, edge_color='lightgray', alpha=0.3)
nx.draw_networkx_edges(G, pos, edgelist=medium_flow_edges, ax=ax2, 
                      width=3, edge_color='orange', alpha=0.7)
nx.draw_networkx_edges(G, pos, edgelist=high_flow_edges, ax=ax2, 
                      width=5, edge_color='red', alpha=0.9)

nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=node_colors, node_size=node_sizes, alpha=0.8)
nx.draw_networkx_labels(G, pos, labels, ax=ax2, font_size=8, font_weight='bold')

ax2.set_aspect('equal')
ax2.axis('off')

# === サブプロット3: 電圧レベル別表示 ===
ax3.set_title('電圧レベル別バス分類', fontsize=14, fontweight='bold')

# 電圧レベル別の色分け
voltage_colors = []
for node in G.nodes():
    voltage = G.nodes[node]['voltage']
    if voltage >= 275:
        voltage_colors.append('red')       # 超高圧
    elif voltage >= 187:
        voltage_colors.append('orange')    # 高圧
    elif voltage >= 100:
        voltage_colors.append('yellow')    # 中圧
    else:
        voltage_colors.append('lightblue') # 低圧

nx.draw_networkx_edges(G, pos, ax=ax3, width=1, edge_color='gray', alpha=0.5)
nx.draw_networkx_nodes(G, pos, ax=ax3, 
                      node_color=voltage_colors, 
                      node_size=500, 
                      alpha=0.8)
nx.draw_networkx_labels(G, pos, labels, ax=ax3, font_size=8, font_weight='bold')

ax3.set_aspect('equal')
ax3.axis('off')

# 電圧レベル凡例
voltage_legend = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, label='275kV'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=12, label='187kV'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=12, label='100kV+'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=12, label='<100kV')
]
ax3.legend(handles=voltage_legend, loc='upper right', fontsize=10)

# === サブプロット4: ネットワーク統計情報 ===
ax4.set_title('ネットワーク統計情報', fontsize=14, fontweight='bold')
ax4.axis('off')

# ネットワーク統計の計算
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
density = nx.density(G)
avg_degree = np.mean([d for n, d in G.degree()])

# バス種類の統計
bus_types = {'slack': 0, 'generator': 0, 'load': 0, 'transit': 0}
voltage_levels = {}
for node in G.nodes():
    bus_types[G.nodes[node]['type']] += 1
    voltage = G.nodes[node]['voltage']
    voltage_levels[voltage] = voltage_levels.get(voltage, 0) + 1

# 電力フロー統計
power_flows = [d['power_flow'] for (u, v, d) in G.edges(data=True)]
high_flow_count = sum(1 for p in power_flows if p > 100)
medium_flow_count = sum(1 for p in power_flows if 50 <= p <= 100)
low_flow_count = sum(1 for p in power_flows if p < 50)

# 統計情報のテキスト表示
stats_text = f"""
ネットワーク基本情報:
• ノード数: {n_nodes}
• エッジ数: {n_edges}
• ネットワーク密度: {density:.3f}
• 平均次数: {avg_degree:.2f}

バス種類別統計:
• スラックバス: {bus_types['slack']}
• 発電バス: {bus_types['generator']}
• 負荷バス: {bus_types['load']}
• 中継バス: {bus_types['transit']}

電力フロー統計:
• 重負荷線路(>100MW): {high_flow_count}
• 中負荷線路(50-100MW): {medium_flow_count}
• 軽負荷線路(<50MW): {low_flow_count}

電圧レベル統計:
"""

for voltage, count in sorted(voltage_levels.items(), reverse=True):
    stats_text += f"• {voltage}kV: {count}バス\n"

stats_text += f"""
ネットワーク特性:
• 接続性: {"連結" if nx.is_connected(G) else "非連結"}
• 最大次数: {max(dict(G.degree()).values())}
• 最小次数: {min(dict(G.degree()).values())}
• 平均最短路長: {nx.average_shortest_path_length(G):.2f}
"""

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# レイアウト調整
plt.tight_layout()

# 保存
output_file = data_path + 'north30_network_python.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"\n✓ 可視化完了: {output_file}")

# 追加分析: 中心性解析
print("\nネットワーク中心性解析...")
centrality_data = {}

# 次数中心性
degree_centrality = nx.degree_centrality(G)
# 媒介中心性
betweenness_centrality = nx.betweenness_centrality(G)
# 近接中心性
closeness_centrality = nx.closeness_centrality(G)

print("\n=== 中心性ランキング (Top 5) ===")
print("\n1. 次数中心性 (Degree Centrality):")
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
for i, (node, centrality) in enumerate(top_degree, 1):
    print(f"   {i}. バス{node}: {centrality:.3f}")

print("\n2. 媒介中心性 (Betweenness Centrality):")
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
for i, (node, centrality) in enumerate(top_betweenness, 1):
    print(f"   {i}. バス{node}: {centrality:.3f}")

print("\n3. 近接中心性 (Closeness Centrality):")
top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
for i, (node, centrality) in enumerate(top_closeness, 1):
    print(f"   {i}. バス{node}: {centrality:.3f}")

# 中心性可視化用の追加プロット
fig2, ax_centrality = plt.subplots(1, 1, figsize=(12, 10))
ax_centrality.set_title('ネットワーク中心性可視化', fontsize=16, fontweight='bold')

# 媒介中心性に基づいてノードサイズを調整
centrality_sizes = [betweenness_centrality[node] * 5000 + 100 for node in G.nodes()]

# 次数中心性に基づいて色を調整
centrality_colors = [degree_centrality[node] for node in G.nodes()]

nx.draw_networkx_edges(G, pos, ax=ax_centrality, width=1, edge_color='gray', alpha=0.5)
nodes = nx.draw_networkx_nodes(G, pos, ax=ax_centrality,
                              node_size=centrality_sizes,
                              node_color=centrality_colors,
                              cmap='viridis',
                              alpha=0.8)
nx.draw_networkx_labels(G, pos, labels, ax=ax_centrality, font_size=8, font_weight='bold')

ax_centrality.set_aspect('equal')
ax_centrality.axis('off')

# カラーバー
cbar = plt.colorbar(nodes, ax=ax_centrality, shrink=0.8)
cbar.set_label('次数中心性 (Degree Centrality)', fontsize=12)

plt.figtext(0.02, 0.02, 'ノードサイズ: 媒介中心性に比例', fontsize=10, style='italic')

centrality_output = data_path + 'north30_centrality_analysis.png'
plt.savefig(centrality_output, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"✓ 中心性解析完了: {centrality_output}")

# データをCSVで保存
centrality_df = pd.DataFrame({
    'bus': list(G.nodes()),
    'degree_centrality': [degree_centrality[node] for node in G.nodes()],
    'betweenness_centrality': [betweenness_centrality[node] for node in G.nodes()],
    'closeness_centrality': [closeness_centrality[node] for node in G.nodes()],
    'degree': [G.degree(node) for node in G.nodes()]
})
csv_output = data_path + 'north30_centrality_analysis.csv'
centrality_df.to_csv(csv_output, index=False)
print(f"✓ 中心性データ保存: {csv_output}")

# plt.show()  # GUI表示を無効化

print("\n=== 解析完了 ===")
print("生成されたファイル:")
print(f"• {output_file} (基本ネットワーク可視化)")
print(f"• {centrality_output} (中心性解析)")  
print(f"• {csv_output} (中心性データ)")