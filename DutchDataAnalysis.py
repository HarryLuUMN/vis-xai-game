#%%

# method: import dataset
def parse_dat_file(path):
    with open(path, 'r') as file:
        data = file.readlines()

    data = [list(map(int, line.split())) for line in data]
    print(data)
    return data


#%%
import itertools

advice_path = r'DutchSchoolDataset\CompletedDataset\klas12b-advice.dat'

# this dataset contains the grades given to students
advice_data = parse_dat_file(advice_path)
advice_data = list(itertools.chain(*advice_data))
print(advice_data)

# %%
import matplotlib.pyplot as plt

plt.hist(advice_data, edgecolor='black')
plt.title('Histogram of Advice Data')
plt.xlabel('Advice Score')
plt.ylabel('Frequency')
plt.show()
# %%
student_ids = [i for i in range(1, advice_data.__len__() + 1)] 

plt.bar(student_ids, advice_data, color='green', edgecolor='black', width=0.5)
plt.title('Advice Data for each Students')
plt.xlabel('Student ID')
plt.ylabel('Advice Score')

# %%
net1_path = r'DutchSchoolDataset\CompletedDataset\klas12b-net-1.dat'
net2_path = r'DutchSchoolDataset\CompletedDataset\klas12b-net-2.dat'
net3_path = r'DutchSchoolDataset\CompletedDataset\klas12b-net-3.dat'
net4_path = r'DutchSchoolDataset\CompletedDataset\klas12b-net-4.dat'

#%%

net1_data = parse_dat_file(net1_path)
net2_data = parse_dat_file(net2_path)
net3_data = parse_dat_file(net3_path)
net4_data = parse_dat_file(net4_path)

print(net2_data)
# %%
import networkx as nx
import numpy as np
import csv
#%%
def visualize_network(data):
    adj_matrix = np.array(data)

    G = nx.from_numpy_array(adj_matrix)

    pos = nx.spring_layout(G)
    edges = G.edges(data=True)

    # Determine the weights and styles for edges
    weights = [edge[2]['weight'] for edge in edges]
    styles = ['dashed' if edge[2]['weight'] == 9 else 'solid' for edge in edges]

    # Draw the graph with different edge styles and adjust the width for dashed lines
    for edge, style in zip(edges, styles):
        # width = edge[2]['weight'] if style == 'solid' else edge[2]['weight'] * 0.01
        color = 'red' if style == 'dashed' else 'gray'
        color = 'black' if edge[2]['weight'] == 10 else color
        nx.draw_networkx_edges(G, pos, edgelist=[edge], style=style, edge_color=color)

    nx.draw_networkx_nodes(G, pos, node_color='lightblue')
    nx.draw_networkx_labels(G, pos)

    plt.title('Graph Visualization from Adjacency Matrix with Weights')
    plt.show()

# %%
visualize_network(net1_data)
visualize_network(net2_data)
visualize_network(net3_data)
visualize_network(net4_data)
# %%
# loading data from files

demo_path = r'DutchSchoolDataset\CompletedDataset\klas12b-demographics.dat'
demo_data = parse_dat_file(demo_path)

delin_path = r'DutchSchoolDataset\CompletedDataset\klas12b-delinquency.dat'
delin_data = parse_dat_file(delin_path)

alcolhol_path = r'DutchSchoolDataset\CompletedDataset\klas12b-alcohol.dat'
alcohol_data = parse_dat_file(alcolhol_path)

print(demo_data, delin_data)

# %%
import matplotlib.colors as mcolors

def visualize_temporal_attributes_network(adj_matrix, attributes_table):
    # 将邻接矩阵转为 NumPy 数组
    adj_matrix = np.array(adj_matrix)
    
    # 根据邻接矩阵创建图
    G = nx.from_numpy_array(adj_matrix)

    # 计算图节点的布局（保持布局一致性）
    pos = nx.spring_layout(G)

    # 创建颜色映射器，将属性值1-5映射为特定的颜色
    cmap = plt.get_cmap('viridis', 5)  # 使用 'viridis' colormap，并分为5个离散区间
    norm = mcolors.BoundaryNorm(boundaries=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], ncolors=5)  # 设置属性值的区间
    
    # 遍历属性表的每个时间点
    for i, attributes in enumerate(attributes_table):
        plt.figure()
        
        # 根据属性值设置节点颜色
        node_colors = [cmap(norm(attr)) for attr in attributes]
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap)
        
        # 绘制边
        nx.draw_networkx_edges(G, pos)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos)
        
        # 添加标题
        plt.title(f'Network Visualization at Time {i+1}')
        
        # 显示颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 空数组用于颜色条
        plt.colorbar(sm, ticks=[1, 2, 3, 4, 5], label='Node Attributes')
        
        # 显示图
        plt.show()



# %%

delin_data = np.array(delin_data)
delin_data = np.transpose(delin_data)

print(delin_data, delin_data.shape)

visualize_temporal_attributes_network(net1_data, delin_data)

# %%
import csv

def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Example usage:
save_to_csv(delin_data, 'delin_data.csv')
save_to_csv(net1_data, 'net1_data.csv')
save_to_csv(net2_data, 'net2_data.csv')
save_to_csv(net3_data, 'net3_data.csv')
save_to_csv(net4_data, 'net4_data.csv')
save_to_csv(demo_data, 'demo_data.csv')
save_to_csv([advice_data], 'advice_data.csv')
save_to_csv(alcohol_data, 'alcohol_data.csv')


# %%
