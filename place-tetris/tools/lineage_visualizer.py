import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import networkx as nx

from networkx.drawing.nx_agraph import graphviz_layout

# Save the original stderr
original_stderr = sys.stderr
original_stdout = sys.stdout

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import transform_encode as te

import local_params
from local_params import GP, TP

try:
    # Redirect stderr to os.devnull
    sys.stderr = open(os.devnull, 'w')
    sys.stdout = open(os.devnull, 'w')
    import matplotlib.pyplot as plt
    #manager = plt.get_current_fig_manager()
    #manager.window.showMaximized()

finally:
    # Restore the original stderr
    sys.stderr = original_stderr
    sys.stdout = original_stdout

def visualize_lineages(df, N=30, K=10):
    # Ensure K is not greater than N
    K = min(K, N)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Identify the top N ranked models and their descendants
    top_n_set = df.nlargest(N, 'rank')
    top_n_ids = set(top_n_set['id'])
    
    # Identify the top K ranked models specifically for coloring
    top_k_set = df.nlargest(K, 'rank')
    top_k_ids = set(top_k_set['id'])
    
    # Function to recursively add parents to the graph
    def add_parents(child_id):
        parent_id = df[df['id'] == child_id]['parentID'].iloc[0]
        if parent_id is not None and parent_id not in G:
            parent = df[df['id'] == parent_id].iloc[0]
            G.add_node(parent_id, rank=f"{parent['rank']:.0f}", gen=parent['gen'], color='green' if parent_id in top_k_ids else 'skyblue')
            G.add_edge(parent_id, child_id)
            add_parents(parent_id)
        if parent_id is not None:
            G.add_edge(parent_id, child_id)
    
    # Add the top models and their descendants
    for node_id in top_n_ids:
        G.add_node(node_id, rank=f"{df[df['id'] == node_id]['rank'].iloc[0]:.0f}", gen=df[df['id'] == node_id]['gen'].iloc[0], color='green' if node_id in top_k_ids else 'skyblue')
        add_parents(node_id)

    # Assign subset keys based on generations
    generations = sorted(set(df['gen']))  # Get unique generations
    subset_key = {gen: [node for node in G if G.nodes[node]['gen'] == gen] for gen in generations}
    
    # Calculate the hierarchical layout
    #pos = nx.multipartite_layout(G, subset_key=subset_key)
    pos = graphviz_layout(G, prog='dot', args='-Grankdir=LR')

    # Place the child nodes with the highest generation number closer to the center of the graph
    def adjust_verts(parent_id, y_shift):
        children = list(G.successors(parent_id))
        if children:
            # Get list of (index, id, gen) for each child
            c_gen = np.array([(i, child, int(df[df['id'] == child]['gen'].item())) for i, child in enumerate(children)])
            ys = np.array([pos[child][1] + y_shift for child in children])
            
            # Determine the order of c_gen such that highest gen numbers are at the center of the list
            sorted_indices = np.argsort(c_gen[:, 2].astype(int))
            left_slice = sorted_indices[::2]
            right_slice = sorted_indices[1::2][::-1] # Reverse to get descending order

            reordered_ind = np.concatenate((left_slice,right_slice))
            reordered_c_gen = c_gen[reordered_ind]

            # Adjust the y position of the child nodes
            for i, child in enumerate(reordered_c_gen):
                id = child[1]
                #print(f'Parent Rank: {df[df['id'] == parent_id]['rank'].iloc[0]:.0f}, Child Rank: {df[df['id'] == id]['rank'].iloc[0]:.0f}')
                old_y = pos[id][1]
                pos[id] = (pos[id][0], ys[i])
                adjust_verts(id, pos[id][1] - old_y)
                
    # Find all root nodes
    root_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]

    # Adjust the vertical position of the child nodes for each root node
    for root_node in root_nodes:
        adjust_verts(root_node, 0)

    # Alter the horizontal position of the nodes
    for node_id, (_, y_offset) in pos.items():
        gen = G.nodes[node_id]['gen']
        pos[node_id] = (gen*20, y_offset)

    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'rank'), 
            node_size=500, node_color=[data['color'] for _, data in G.nodes(data=True)], 
            font_size=8, arrowsize=10, arrowstyle='-|>', font_weight='bold')
    
    # Add generation labels at the bottom
    # Get the bounds of the window
    x_values, y_values = zip(*pos.values())
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    # Calculate the bottom y position for generation labels
    bottom_y = min_y - 0.1 * (max_y - min_y)  # 10% below the lowest node
    for gen in generations:
        gen_nodes = subset_key[gen]
        gen_positions = [pos[node] for node in gen_nodes]
        x = np.mean([x for x, _ in gen_positions])
        plt.text(x, bottom_y + 0.01, f'{gen}', 
                 horizontalalignment='center', fontsize=12, fontweight='bold', color='red')
    
    # Display the plot
    plt.title('Model Lineages Visualization')
    plt.axis('off')  # Turn off the axis
    # Set the window size to full screen
    plt.show()

# Example DataFrame
'''data = {
    'id': np.arange(1, 51),  # 50 models
    'parentID': [None, None, None, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,
                 9, 10, 10, 11, 11, 12, 12, None, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19,
                 19, None, 20, 21, 21, 22, 22, 23, 23, None],  # Parents, some roots
    'rank': np.random.randint(1, 100, 50),  # Random rankings between 1 and 100
    'gen': [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
            3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
            6, 7, 7, 7, 7, 7, 7, 8, 8, 9]  # Generations
}
df = pd.DataFrame(data)

# Visualize the lineages
visualize_lineages(df)'''

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROF_DIR = os.path.join(CURRENT_DIR, "../profiler/")
MODELS_DIR = os.path.join(CURRENT_DIR, "../models/")

ft = TP["feature_transform"]
file_name = f"models_{GP['rows']}x{GP['cols']}_{te.encode(ft)}.parquet"
models_data_file = os.path.join(MODELS_DIR, file_name)
data = pd.read_parquet(models_data_file)

visualize_lineages(data, N=40, K=10)
