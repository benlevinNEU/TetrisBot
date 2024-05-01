import numpy as np
import pandas as pd

TEST = False

#from tools.lineage_visualizer import visualize_lineages

def getParents(models_info, pop_size, n_parents, lin_prop_max, p_random):
    # Sort models by descending rank (assuming rank is higher for better models)
    models = models_info.sort_values(by="rank", ascending=False)

    child_set = int(pop_size * (1 - p_random))
    
    # Initialize lineage tracking and parent selection
    lineage_dict = {}
    lineage_counts = {}
    parents = pd.DataFrame(columns=models_info.columns)
    parents['max_children'] = []

    # Assign each model to its own lineage initially
    for index, model in models.iterrows():
        lineage_dict[model['id']] = set([model['id']])
        lineage_counts[model['id']] = 0

    # Merge lineages based on parent-child relationships
    for index, model in models.iterrows():
        child_id = model['id']

        parent_id = model['parentID']
        if parent_id in models['id'].values: # Should always be true unless parrent id is null
            lineage_dict[child_id].update(lineage_dict[parent_id])
            #lineage_dict[parent_id].update(lineage_dict[child_id])
            for key, lineage in lineage_dict.items():
                if parent_id in lineage or child_id in lineage:
                    lineage.update(lineage_dict[child_id])
        else:
            if parent_id is not None:
                raise ValueError(f"Parent ID {parent_id} not found in models for child ID {child_id}")
            
    if TEST:
        return lineage_dict # For testing purposes

    # Select parents ensuring lineage proportions are maintained
    total_selected = 0
    for index, model in models.iterrows():
        if n_parents >= n_parents and total_selected >= child_set:
            break
        
        current_id = model['id']
        # Identify the lineage of the current model
        current_lineage = lineage_dict[current_id]

        # Calculate the total offspring allowed for the lineage
        lineage_limit = int(np.ceil(lin_prop_max * child_set))

        # Sum the current offspring count for all members of this lineage
        current_lineage_offspring = sum([lineage_counts[id] for id in current_lineage])

        # Determine available slots for this lineage
        available_slots = min(lineage_limit - current_lineage_offspring, int(np.ceil(child_set / n_parents)))

        if available_slots == 0:
            # Reduce the 'max_children' value for all models in the current lineage to accomadate the new model
            lineage_parent_indices = parents[parents['id'].isin(current_lineage)].index
            parents.loc[lineage_parent_indices, 'max_children'] = np.ceil((parents.loc[lineage_parent_indices, 'max_children'] * len(parents) / (len(parents) + 1))).astype(int)

            # Increase the number of children in new population
            if current_lineage_offspring >= lineage_limit:
                child_set += 1

            # Enable an extra child to be selected for this lineage if lineage is full
            lineage_limit += 1

            # Determine available slots for this lineage
            available_slots = lineage_limit - current_lineage_offspring

            # Flexible number of parents to accomadate all best preforming parents by reducing number of kids per parent
            n_parents += 1
        
        if available_slots > 0:
            selected_slots = int(min(available_slots, child_set - total_selected))
            # Append to parents DataFrame
            new_model = model.copy()
            new_model['max_children'] = selected_slots
            new_model_df = pd.DataFrame([new_model])
            if parents.empty:
                parents = new_model_df
            else:
                parents = pd.concat([parents, new_model_df])
            
            # Update lineage counts
            lineage_counts[current_id] += selected_slots
            total_selected += selected_slots

    return parents

if __name__ == '__main__':
    # Test the function
    models_info = pd.DataFrame({
        'id': ['0', '1', '02', '03', '14', '15', '026', '027', '038', '039', '03_10', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17', '026_18', '026_19'],
        'parentID': [None, None, '0', '0', '1', '1', '02', '02', '03', '03', '03', '14', '14', '14', '15', '15', '15', '15', '026', '026'],
        'rank': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    })

    sorted_models = models_info.sort_values(by="rank", ascending=False)

    parents = getParents(sorted_models, 10, 4, 0.4, 0.2)

    lineage_dict = {
        '0': {'0', '02', '03', '026', '027', '038', '039', '03_10', '026_18', '026_19'},
        '1': {'1', '14', '15', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17'},
        '02': {'0', '02', '03', '026', '027', '038', '039', '03_10', '026_18', '026_19'},
        '03': {'0', '02', '03', '026', '027', '038', '039', '03_10', '026_18', '026_19'},
        '14': {'1', '14', '15', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17'},
        '15': {'1', '14', '15', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17'},
        '026': {'0', '02', '03', '026', '027', '038', '039', '03_10', '026_18', '026_19'},
        '027': {'0', '02', '03', '026', '027', '038', '039', '03_10', '026_18', '026_19'},
        '038': {'0', '02', '03', '026', '027', '038', '039', '03_10', '026_18', '026_19'},
        '039': {'0', '02', '03', '026', '027', '038', '039', '03_10', '026_18', '026_19'},
        '03_10': {'0', '02', '03', '026', '027', '038', '039', '03_10', '026_18', '026_19'},
        '14_11': {'1', '14', '15', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17'},
        '14_12': {'1', '14', '15', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17'},
        '14_13': {'1', '14', '15', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17'},
        '15_14': {'1', '14', '15', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17'},
        '15_15': {'1', '14', '15', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17'},
        '15_16': {'1', '14', '15', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17'},
        '15_17': {'1', '14', '15', '14_11', '14_12', '14_13', '15_14', '15_15', '15_16', '15_17'},
        '026_18': {'0', '02', '03', '026', '027', '038', '039', '03_10', '026_18', '026_19'},
        '026_19': {'0', '02', '03', '026', '027', '038', '039', '03_10', '026_18', '026_19'}
    }

    # Test the function
    if TEST:
        assert lineage_dict == parents
        print('All tests passed!')

    else: print(parents)

    #visualize_lineages(models_info)