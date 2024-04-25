import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import networkx as nx

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

finally:
    # Restore the original stderr
    sys.stderr = original_stderr
    sys.stdout = original_stdout

# Function to recursively add descendants to the graph
'''def add_descendants(G, df, parent_id, y_offset_dict, pos, top_k_ids, depth=0):
    children = df[df['parentID'] == parent_id]
    
    # We use a depth parameter to keep track of the horizontal positioning.
    for i, child in children.iterrows():
        child_id = child['id']
        if child_id not in G:
            # The generation of the child is based on its 'gen' value in the DataFrame.
            child_gen = child['gen']
            
            # Check the next available vertical position for this generation.
            y_offset = y_offset_dict.get(child_gen, depth)
            
            # Update y_offset_dict for the next sibling or cousin.
            y_offset_dict[child_gen] = y_offset + 1
            
            # Set position for child.
            pos[child_id] = (child_gen, y_offset)
            
            # Color the node if it's in the top K models.
            color = 'green' if child_id in top_k_ids else 'skyblue'
            
            # Add node and edge to the graph.
            G.add_node(child_id, rank=f"{child['rank']:.0f}", gen=child_gen, color=color)
            G.add_edge(parent_id, child_id)
            
            # Recurse to add this child's descendants.
            add_descendants(G, df, child_id, y_offset_dict, pos, top_k_ids, depth + 1)'''

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
    
    # Function to recursively add descendants to the graph
    def add_descendants(parent_id):
        children = df[df['parentID'] == parent_id]
        for _, child in children.iterrows():
            child_id = child['id']
            if child_id not in G:
                G.add_node(child_id, rank=f"{child['rank']:.0f}", gen=child['gen'], color='green' if child_id in top_k_ids else 'skyblue')
                G.add_edge(parent_id, child_id)
                add_descendants(child_id)
    
    # Add the top models and their descendants
    for node_id in top_n_ids:
        G.add_node(node_id, rank=f"{df[df['id'] == node_id]['rank'].iloc[0]:.0f}", gen=df[df['id'] == node_id]['gen'].iloc[0], color='green' if node_id in top_k_ids else 'skyblue')
        add_descendants(node_id)

    # Assign subset keys based on generations
    generations = sorted(set(df['gen']))  # Get unique generations
    subset_key = {gen: [node for node in G if G.nodes[node]['gen'] == gen] for gen in generations}
    
    # Calculate the hierarchical layout
    pos = nx.multipartite_layout(G, subset_key=subset_key)
    
    '''# Get the nodes in the last generation
    last_gen = max(generations)
    last_layer_nodes = subset_key[f"gen_{last_gen}"]
    # Group nodes in the last generation that share the same parent
    parent_child_dict = nx.to_dict_of_lists(G)
    shared_parent_groups = {}
    for parent, children in parent_child_dict.items():
        last_gen_children = [child for child in children if child in last_layer_nodes]
        if last_gen_children:
            shared_parent_groups[parent] = last_gen_children
    '''
    
    # Draw the graph
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'rank'), 
            node_size=600, node_color=[data['color'] for _, data in G.nodes(data=True)], 
            font_size=9, arrowsize=12, arrowstyle='-|>', font_weight='bold')
    
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
        plt.text(x, bottom_y + 0.05, f'{gen}', 
                 horizontalalignment='center', fontsize=12, fontweight='bold', color='red')
    

    # Display the plot
    plt.title('Model Lineages Visualization')
    plt.axis('off')  # Turn off the axis
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
