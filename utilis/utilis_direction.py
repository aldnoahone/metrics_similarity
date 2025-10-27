import numpy as np
from skimage.filters import threshold_otsu, threshold_yen, threshold_li, threshold_triangle
from function import direction_judger
from utilis import colors_cover_2
"""
the directional link analysis

"""

from utilis import colors_cover_2
from function import direction_judger
import numpy as np

color_map = {
    1: colors_cover_2[0],    
    2: colors_cover_2[1],   
    3: colors_cover_2[2],
    4: colors_cover_2[3], 
    6: colors_cover_2[3],
    7: colors_cover_2[2],
    8: colors_cover_2[1],
    9: colors_cover_2[0]
}

def link_to_direction(link_matrix, coors, step_x, step_y):
    """
    Assign direction labels to connections in the link matrix.
    
    Args:
        link_matrix: Binary matrix indicating connections between points
        coors: Coordinate array for the points (what was 'S' in original)
        step_x: Step size in x direction (what was '_dist' in original)
        step_y: Step size in y direction (what was '_dist' in original)
    
    Returns:
        arr_notes: Matrix with direction labels for each connection
    """
    arr_notes = np.zeros(shape=link_matrix.shape, dtype=np.int8)
    
    for idx, row in enumerate(link_matrix):
        idx_neighbors = np.where(row == 1)[0]
        
        # Check if there are any neighbors (empty array has length 0)
        if len(idx_neighbors) == 0:
            continue
        
        for idx_neighbor in idx_neighbors:
            arr_notes[idx, idx_neighbor] = direction_judger(
                indice_center=idx,
                indice_neighboor=idx_neighbor,  # Fixed typo
                coors=coors,
                step_x=step_x,
                step_y=step_y
            )
    
    return arr_notes

def extract_direction_info(amp, target_directions, data_raw, S, _dist):
    """
    Extract complete direction information including distances, neighbors, and validity
    
    Returns:
    Dict structure:
    {
        direction: {
            'distances': np.array,
            'neighbors': np.array,
            'has_neighbor': np.array (boolean mask)
        }
    }
    """
    nei_1_list = data_raw[amp]['nei_1']
    dist_matrix = data_raw[amp]['dist_matrix']
    N = len(nei_1_list)
    
    # Handle single direction input
    if isinstance(target_directions, int):
        target_directions = [target_directions]
        return_single = True
    else:
        return_single = False
    
    # Initialize results
    results = {}
    for direction in target_directions:
        results[direction] = {
            'distances': np.full(N, np.nan),
            'neighbors': np.full(N, -1, dtype=int),
            'has_neighbor': np.full(N, False, dtype=bool)
        }
    
    for i, neighbors in enumerate(nei_1_list):
        for j in neighbors:
            if j == i:
                continue
                
            direction = direction_judger(
                indice_center=i,
                indice_neighboor=j,
                coors=S,
                step_x=_dist,
                step_y=_dist
            )
            
            if direction in target_directions:
                results[direction]['distances'][i] = dist_matrix[i, j]
                results[direction]['neighbors'][i] = j
                results[direction]['has_neighbor'][i] = True
    
    return results[list(target_directions)[0]] if return_single else results

def compute_thresholds(values):
    """Compute all threshold methods for given values"""
    if len(values) <= 1:
        return {}
    
    thresholds = {}
    try:
        thresholds['Otsu'] = threshold_otsu(values)
    except:
        pass
    try:
        thresholds['Yen'] = threshold_yen(values)
    except:
        pass
    try:
        thresholds['Li'] = threshold_li(values)
    except:
        pass
    try:
        thresholds['Triangle'] = threshold_triangle(values)
    except:
        pass
    
    return thresholds

def directional_link_matrix(amp, data_raw, S, _dist, directions, threshold_method='Otsu', 
                           return_both_matrices=True):
    """
    Create directional link matrices based on threshold analysis
    
    Parameters:
    amp: amplitude value
    data_raw: data dictionary
    S: coordinate array
    _dist: step distance
    directions: 'all', int, or list of ints
    threshold_method: 'Otsu', 'Yen', 'Li', 'Triangle'
    return_both_matrices: if True, returns (above_threshold, below_threshold)
                         if False, returns only above_threshold
    
    Returns:
    if directions == 'all':
        link_matrix(es) based on threshold from all directions' distances combined
    if directions == int:
        single directional link_matrix(es) based on threshold from single direction
    if directions == list:
        merged directional link_matrix(es) from multiple directions using single merged threshold
    """
    
    nei_1_list = data_raw[amp]['nei_1']
    dist_matrix = data_raw[amp]['dist_matrix']
    N = len(nei_1_list)
    
    # Initialize link matrices
    link_matrix_above = np.zeros((N, N), dtype=np.uint8)
    link_matrix_below = np.zeros((N, N), dtype=np.uint8)
    
    if directions == 'all':
        # Traditional method: use all neighbor distances regardless of direction
        all_distances = []
        for element in data_raw[amp]['dist_2_of_nei_1']:
            for value in element:
                if value != 0:
                    all_distances.append(value)
        
        if len(all_distances) == 0:
            print(f"No distance data available for amp {amp}")
            if return_both_matrices:
                return link_matrix_above, link_matrix_below
            else:
                return link_matrix_above
        
        # Compute threshold from all distances
        all_distances = np.array(all_distances)
        thresholds = compute_thresholds(all_distances)
        
        if threshold_method not in thresholds:
            print(f"Could not compute {threshold_method} threshold for amp {amp}")
            if return_both_matrices:
                return link_matrix_above, link_matrix_below
            else:
                return link_matrix_above
        
        threshold_value = thresholds[threshold_method]
        print(f"For directions {directions} the threshold is {threshold_value}")
        # Apply threshold to all neighbor connections
        for i, neighbors in enumerate(nei_1_list):
            for j in neighbors:
                if j == i:
                    continue
                dist = dist_matrix[i, j]
                if dist > threshold_value:
                    link_matrix_above[i, j] = 1
                else:
                    link_matrix_below[i, j] = 1
    
    else:
        # Directional method
        if isinstance(directions, int):
            target_directions = [directions]
        elif isinstance(directions, list):
            target_directions = directions
        else:
            # Handle string inputs like 'cardinal', 'diagonal'
            direction_groups = {
                'cardinal': [2, 4, 6, 8],
                'diagonal': [1, 3, 7, 9],
                'horizontal': [4, 6],
                'vertical': [2, 8]
            }
            target_directions = direction_groups.get(directions, [directions])
        
        # Extract direction information
        direction_info = extract_direction_info(amp, target_directions, data_raw, S, _dist)
        
        if isinstance(directions, int):
            # Single direction: compute threshold from single direction's distances
            direction = directions
            if direction not in direction_info:
                print(f"No data for direction {direction} in amp {amp}")
                if return_both_matrices:
                    return link_matrix_above, link_matrix_below
                else:
                    return link_matrix_above
            
            # Get distances for this direction (exclude NaN values)
            direction_distances = direction_info[direction]['distances']
            valid_distances = direction_distances[~np.isnan(direction_distances)]
            
            if len(valid_distances) == 0:
                print(f"No valid distances for direction {direction} in amp {amp}")
                if return_both_matrices:
                    return link_matrix_above, link_matrix_below
                else:
                    return link_matrix_above
            
            # Compute threshold for this direction
            thresholds = compute_thresholds(valid_distances)
            if threshold_method not in thresholds:
                print(f"Could not compute {threshold_method} threshold for direction {direction}")
                if return_both_matrices:
                    return link_matrix_above, link_matrix_below
                else:
                    return link_matrix_above
            
            threshold_value = thresholds[threshold_method]
            print(f"For directions {directions} the threshold is {threshold_value}")
            # Apply threshold only to connections in this direction
            for i in range(N):
                if direction_info[direction]['has_neighbor'][i]:
                    j = direction_info[direction]['neighbors'][i]
                    dist = direction_info[direction]['distances'][i]
                    if dist > threshold_value:
                        link_matrix_above[i, j] = 1
                    else:
                        link_matrix_below[i, j] = 1
        
        else:
            # Multiple directions: MERGE all distances first, then compute single threshold
            merged_distances = []
            
            for direction in target_directions:
                if direction not in direction_info:
                    continue
                
                # Get distances for this direction
                direction_distances = direction_info[direction]['distances']
                valid_distances = direction_distances[~np.isnan(direction_distances)]
                
                if len(valid_distances) > 0:
                    merged_distances.extend(valid_distances)
            
            if len(merged_distances) == 0:
                print(f"No valid distances for directions {target_directions} in amp {amp}")
                if return_both_matrices:
                    return link_matrix_above, link_matrix_below
                else:
                    return link_matrix_above
            
            # Compute single threshold from merged distances
            merged_distances = np.array(merged_distances)
            thresholds = compute_thresholds(merged_distances)
            
            if threshold_method not in thresholds:
                print(f"Could not compute {threshold_method} threshold for merged directions")
                if return_both_matrices:
                    return link_matrix_above, link_matrix_below
                else:
                    return link_matrix_above
            
            threshold_value = thresholds[threshold_method]
            print(f"For directions {directions} the threshold is {threshold_value}")
            # Apply the single threshold to all connections in specified directions
            for direction in target_directions:
                if direction not in direction_info:
                    continue
                
                for i in range(N):
                    if direction_info[direction]['has_neighbor'][i]:
                        j = direction_info[direction]['neighbors'][i]
                        dist = direction_info[direction]['distances'][i]
                        if not np.isnan(dist):
                            if dist > threshold_value:
                                link_matrix_above[i, j] = 1
                            else:
                                link_matrix_below[i, j] = 1
    
    if return_both_matrices:
        return link_matrix_above, link_matrix_below
    else:
        return link_matrix_above
    
# Usage examples:
# def demo_usage():
#     """
#     Example usage of the directional_link_matrix function
#     """
#     amp = 5.0
    
#     # Case 1: All directions (traditional method)
#     link_above_all, link_below_all = directional_link_matrix(
#         amp, data_raw, S, _dist, 'all', 'Otsu')
#     print(f"All directions - Above threshold connections: {np.sum(link_above_all)}")
    
#     # Case 2: Single direction
#     link_above_north, link_below_north = directional_link_matrix(
#         amp, data_raw, S, _dist, 2, 'Otsu')  # North direction
#     print(f"North direction - Above threshold connections: {np.sum(link_above_north)}")
    
#     # Case 3: Multiple directions (merged)
#     link_above_cardinal, link_below_cardinal = directional_link_matrix(
#         amp, data_raw, S, _dist, [2, 4, 6, 8], 'Otsu')  # Cardinal directions
#     print(f"Cardinal directions - Above threshold connections: {np.sum(link_above_cardinal)}")
    
#     # Case 4: Using predefined direction groups
#     link_above_diag, link_below_diag = directional_link_matrix(
#         amp, data_raw, S, _dist, 'diagonal', 'Yen')  # Diagonal directions
#     print(f"Diagonal directions - Above threshold connections: {np.sum(link_above_diag)}")
    
#     # Case 5: Only return above-threshold matrix
#     link_above_only = directional_link_matrix(
#         amp, data_raw, S, _dist, [2, 8], 'Otsu', return_both_matrices=False)
#     print(f"Vertical directions - Above threshold connections: {np.sum(link_above_only)}")

# # Integration with your existing plotting workflow:
# def create_directional_plots(amp, data_raw, S, _dist):
#     """
#     Example of how to integrate with your plotting functions
#     """
#     # Get different types of link matrices
#     link_all_above, link_all_below = directional_link_matrix(
#         amp, data_raw, S, _dist, 'all', 'Otsu')
    
#     link_cardinal_above, link_cardinal_below = directional_link_matrix(
#         amp, data_raw, S, _dist, 'cardinal', 'Otsu')
    
#     link_north_above, link_north_below = directional_link_matrix(
#         amp, data_raw, S, _dist, 2, 'Otsu')
    
#     # Now you can use these matrices with your existing plotting functions
#     # For example:
#     # link_plot(ax=axes[0], S=S, link_matrix=link_all_above, ...)
#     # link_plot(ax=axes[1], S=S, link_matrix=link_cardinal_above, ...)
#     # link_plot(ax=axes[2], S=S, link_matrix=link_north_above, ...)
    
#     return {
#         'all_above': link_all_above,
#         'all_below': link_all_below,
#         'cardinal_above': link_cardinal_above,
#         'cardinal_below': link_cardinal_below,
#         'north_above': link_north_above,
#         'north_below': link_north_below
#     }

# # Example: Compare different threshold methods for same direction
# def compare_threshold_methods(amp, data_raw, S, _dist, directions):
#     """
#     Compare different threshold methods for the same directional analysis
#     """
#     methods = ['Otsu', 'Yen', 'Li', 'Triangle']
#     results = {}
    
#     for method in methods:
#         try:
#             link_above, link_below = directional_link_matrix(
#                 amp, data_raw, S, _dist, directions, method)
#             results[method] = {
#                 'above_count': np.sum(link_above),
#                 'below_count': np.sum(link_below),
#                 'total_connections': np.sum(link_above) + np.sum(link_below)
#             }
#         except Exception as e:
#             print(f"Error with {method}: {e}")
#             results[method] = None
    
#     return results


"""
the directional hist plot

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from skimage.filters import threshold_otsu, threshold_yen, threshold_li, threshold_triangle
import matplotlib as mpl

# Set font parameters
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14


def extract_directional_distances(amp, target_directions, data_raw, S, _dist):
    """
    Extract distances for specific directions
    
    Parameters:
    amp: amplitude value
    target_directions: int, list, or string ('all', 'cardinal', 'diagonal', 'horizontal', 'vertical')
    data_raw: data dictionary
    S: coordinate array
    _dist: step distance
    
    Returns:
    dict with direction as key, distances array as value
    """
    # Predefined direction groups
    direction_groups = {
        'cardinal': [2, 4, 6, 8],           # North, West, East, South
        'diagonal': [1, 3, 7, 9],           # NW, NE, SW, SE
        'all': [1, 2, 3, 4, 6, 7, 8, 9],    # All directions
        'horizontal': [4, 6],               # West, East
        'vertical': [2, 8],                 # North, South
    }
    
    # Handle different input types
    if isinstance(target_directions, str):
        if target_directions in direction_groups:
            target_directions = direction_groups[target_directions]
        else:
            raise ValueError(f"Unknown direction group '{target_directions}'. Available: {list(direction_groups.keys())}")
    elif isinstance(target_directions, int):
        target_directions = [target_directions]
    
    nei_1_list = data_raw[amp]['nei_1']
    dist_matrix = data_raw[amp]['dist_matrix']
    N = len(nei_1_list)
    
    # Initialize results
    direction_distances = {direction: [] for direction in target_directions}
    
    for i, neighbors in enumerate(nei_1_list):
        for j in neighbors:
            if j == i:
                continue
                
            direction = direction_judger(
                indice_center=i,
                indice_neighboor=j,
                coors=S,
                step_x=_dist,
                step_y=_dist
            )
            
            if direction in target_directions:
                dist = dist_matrix[i, j]
                if dist != 0:  # Exclude zero distances
                    direction_distances[direction].append(dist)
    
    # Convert lists to numpy arrays
    for direction in direction_distances:
        direction_distances[direction] = np.array(direction_distances[direction])
    
    return direction_distances

def extract_distances_for_amp_directional(amp, data_raw, target_directions='all', S=None, _dist=None):
    """
    Extract distance values for a specific amp and directions
    
    Parameters:
    amp: amplitude value
    data_raw: data dictionary
    target_directions: directions to include ('all' for traditional method, or specific directions)
    S: coordinate array (needed for directional analysis)
    _dist: step distance (needed for directional analysis)
    """
    if target_directions == 'all' or (S is None or _dist is None):
        # Traditional method - all distances
        values = []
        for element in data_raw[amp]['dist_2_of_nei_1']:
            for value in element:
                if value != 0:
                    values.append(value)
        return {'all': np.array(values)}
    else:
        # Directional method
        return extract_directional_distances(amp, target_directions, data_raw, S, _dist)


def get_direction_name(direction):
    """Convert direction number to simple label"""
    return str(direction)

def analyze_specific_amp_directional(amp, data_raw, target_directions='all', S=None, _dist=None, show_plot=True, if_merged_flag=False):
    """
    Analyze thresholds for a specific amp with directional selection - simplified output
    
    Parameters:
    amp: amplitude value
    data_raw: data dictionary
    target_directions: directions to include ('all' for traditional method, or specific directions)
    S: coordinate array (needed for directional analysis)
    _dist: step distance (needed for directional analysis)
    show_plot: whether to display plots
    if_merged_flag: if True, merge all directional values before analysis
    """
    # Extract distances by direction
    direction_distances = extract_distances_for_amp_directional(amp, data_raw, target_directions, S, _dist)
    
    if not direction_distances or all(len(values) == 0 for values in direction_distances.values()):
        print(f"No distance data available for amp {amp} and specified directions.")
        return None
    
    # Merge all directions if flag is True
    if if_merged_flag:
        merged_values = []
        for direction, values in direction_distances.items():
            if len(values) > 0:
                merged_values.extend(values)
        
        if len(merged_values) == 0:
            print(f"No distance data available for amp {amp} after merging.")
            return None
            
        merged_values = np.array(merged_values)
        direction_distances = {'merged': merged_values}
    
    # Analyze each direction (or merged data)
    direction_results = {}
    
    for direction, values in direction_distances.items():
        if len(values) == 0:
            continue
            
        thresholds = compute_thresholds(values)
        direction_results[direction] = {
            'values': values,
            'thresholds': thresholds
        }
    
    # Plot if requested
    if show_plot and direction_results:
        threshold_colors = {
            'Otsu': 'red',
            'Yen': 'blue', 
            'Li': 'green',
            'Triangle': 'purple'
        }
        
        n_directions = len(direction_results)
        fig_width = min(34, max(8, n_directions * 4))
        fig, axes = plt.subplots(1, n_directions, figsize=(fig_width, 6))
        
        if n_directions == 1:
            axes = [axes]
        
        for idx, (direction, results) in enumerate(direction_results.items()):
            values = results['values']
            thresholds = results['thresholds']
            
            # Histogram only
            ax = axes[idx]
            ax.hist(values, bins=30, edgecolor='black', alpha=0.6, label='Distance Data')
            
            for method, value in thresholds.items():
                ax.axvline(value, linestyle='--', linewidth=2, 
                          color=threshold_colors[method], 
                          label=f'{method}: {value:.3f}')
            
            title_label = f"{target_directions}" if direction == 'merged' else f'Direction {direction}'
            ax.set_title(f'{title_label} (Amp = {amp})')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return direction_results

def analyze_all_amps_directional(data_raw, target_directions='all', S=None, _dist=None, 
                                methods=['Otsu', 'Yen'], show_plot=True,
                                merge_directions = True):
    """
    Analyze thresholds across all amps with directional selection
    """
    direction_str = str(target_directions) if not isinstance(target_directions, list) else f"{len(target_directions)} directions"
    
    print(f"\n{'='*80}")
    print(f"DIRECTIONAL ANALYSIS FOR ALL AMPS | DIRECTIONS = {direction_str}")
    print(f"{'='*80}")
    
    # Prepare data structures
    plot_data = []
    threshold_records = []

    
    for amp, amp_data in data_raw.items():
        direction_distances = extract_distances_for_amp_directional(amp, data_raw, target_directions, S, _dist)
        
        if merge_directions:
            # Merge all distances from different directions for this amp
            merged_values = []
            for direction, values in direction_distances.items():
                if len(values) > 0:
                    merged_values.extend(values)
            
            if len(merged_values) > 0:
                # Store merged plot data
                for value in merged_values:
                    plot_data.append({
                        'amp': amp, 
                        'distance': value, 
                        'direction': 'merged'  # Use 'merged' as direction label
                    })
                
                # Compute thresholds for merged data
                merged_values = np.array(merged_values)
                thresholds = compute_thresholds(merged_values)
                
                # Store threshold data
                for method in methods:
                    if method in thresholds:
                        threshold_records.append({
                            'amp': amp,
                            'direction': 'merged',
                            'method': method,
                            'threshold': thresholds[method]
                        })
        else:
            for direction, values in direction_distances.items():
                if len(values) == 0:
                    continue
                    
                direction_name = get_direction_name(direction)
                
                # Store plot data
                for value in values:
                    plot_data.append({
                        'amp': amp, 
                        'distance': value, 
                        'direction': direction,
                        'direction_name': direction_name
                    })

                # merge the 'distance' in plot_data different directions
                
                # Compute thresholds and statistics
                thresholds = compute_thresholds(values)
                
                # Store threshold data
                for method in methods:
                    if method in thresholds:
                        threshold_records.append({
                            'amp': amp,
                            'direction': direction,
                            'direction_name': direction_name,
                            'method': method,
                            'threshold': thresholds[method]
                        })
    
    # Convert to DataFrames
    df = pd.DataFrame(plot_data)
    threshold_df = pd.DataFrame(threshold_records)
    
    if df.empty:
        print("No data available for analysis.")
        return None, None
    
    # Sort amps
    amp_order = sorted(df['amp'].unique(), key=lambda x: float(x))
    
    # # Plot if requested
    if show_plot and not df.empty:
        method_colors = {
            'Otsu': 'red',
            'Yen': 'blue',
            'Li': 'green', 
            'Triangle': 'purple'
        }
        
        unique_directions = sorted(df['direction'].unique())
        n_directions = len(unique_directions)
        
        if n_directions == 1:
            # Single direction analysis
            fig, axes = plt.subplots(2, 1, figsize=(17, 12))
            
            direction = unique_directions[0]
            direction_name = get_direction_name(direction)
            direction_df = df[df['direction'] == direction]
            direction_threshold_df = threshold_df[threshold_df['direction'] == direction]
            
            # Boxplot
            ax1 = axes[0]
            sns.boxplot(data=direction_df, x='amp', y='distance', order=amp_order, ax=ax1)
            
            for method, group in direction_threshold_df.groupby('method'):
                if method in methods:
                    x_positions = [amp_order.index(a) for a in group['amp']]
                    y_values = group['threshold'].values
                    ax1.plot(x_positions, y_values, marker='o', linestyle='--', 
                            color=method_colors[method], linewidth=2, markersize=6,
                            label=f'{method} Threshold')
            
            ax1.set_title(f"Distance Distribution - Direction {direction}")
            ax1.set_xlabel("Amplitude")
            ax1.set_ylabel("Distance")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)

            # Threshold trends
            ax2 = axes[1]
            for method in methods:
                method_data = direction_threshold_df[direction_threshold_df['method'] == method].sort_values('amp')
                if not method_data.empty:
                    ax2.plot(method_data['amp'].astype(float), method_data['threshold'],
                            marker='o', linestyle='-', color=method_colors[method],
                            linewidth=2, markersize=6, label=f'{method} Threshold')
            
            ax2.set_title(f"Threshold Trends - Direction {direction}")
            ax2.set_xlabel("Amplitude")
            ax2.set_ylabel("Threshold Value")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
        else:
            # Multiple directions analysis
            fig_width = n_directions * 17
            fig, axes = plt.subplots(2, n_directions, figsize=(fig_width, 12))
            
            if n_directions == 1:
                axes = axes.reshape(2, 1)
            
            for idx, direction in enumerate(unique_directions):
                direction_name = get_direction_name(direction)
                direction_df = df[df['direction'] == direction]
                direction_threshold_df = threshold_df[threshold_df['direction'] == direction]
                
                # Boxplot
                ax1 = axes[0, idx]
                sns.boxplot(data=direction_df, x='amp', y='distance', order=amp_order, ax=ax1)
                
                for method, group in direction_threshold_df.groupby('method'):
                    if method in methods:
                        x_positions = [amp_order.index(a) for a in group['amp']]
                        y_values = group['threshold'].values
                        ax1.plot(x_positions, y_values, marker='o', linestyle='--', 
                                color=method_colors[method], linewidth=2, markersize=4,
                                label=f'{method}')
                
                ax1.set_title(f"{direction_name} Direction")
                ax1.set_xlabel("Amplitude")
                ax1.set_ylabel("Distance")
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
                
                # Threshold trends
                ax2 = axes[1, idx]
                for method in methods:
                    method_data = direction_threshold_df[direction_threshold_df['method'] == method].sort_values('amp')
                    if not method_data.empty:
                        ax2.plot(method_data['amp'].astype(float), method_data['threshold'],
                                marker='o', linestyle='-', color=method_colors[method],
                                linewidth=2, markersize=4, label=f'{method}')
                
                ax2.set_title(f"Thresholds - {direction_name}")
                ax2.set_xlabel("Amplitude")
                ax2.set_ylabel("Threshold")
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    return df, threshold_df

def run_directional_threshold_analysis(data_raw, specific_amp=5.0, target_directions='cardinal', 
                                     S=None, _dist=None, threshold_methods=['Otsu', 'Yen']):
    """
    Run complete directional threshold analysis
    
    Parameters:
    data_raw: data dictionary
    specific_amp: amplitude to analyze in detail
    target_directions: 'all', 'cardinal', 'diagonal', 'horizontal', 'vertical', int, or list
    S: coordinate array (required for directional analysis)
    _dist: step distance (required for directional analysis)
    threshold_methods: list of threshold methods to use
    """
    print("DIRECTIONAL THRESHOLD ANALYSIS")
    print("="*80)
    
    # Analyze specific amp
    direction_results = analyze_specific_amp_directional(
        specific_amp, data_raw, target_directions, S, _dist, show_plot=True)
    
    # Analyze all amps
    df_all, threshold_df_all = analyze_all_amps_directional(
        data_raw, target_directions, S, _dist, methods=threshold_methods, show_plot=True)
    
    return {
        'specific_amp': {
            'amp': specific_amp,
            'directions': target_directions,
            'results': direction_results
        },
        'all_amps': {
            'data': df_all,
            'thresholds': threshold_df_all,
            'directions': target_directions
        }
    }

# Example usage:
# results = run_directional_threshold_analysis(
#     data_raw, 
#     specific_amp=5.0, 
#     target_directions='cardinal',  # or [2, 4, 6, 8] or 'diagonal' etc.
#     S=S, 
#     _dist=_dist, 
#     threshold_methods=['Otsu', 'Yen']
# )

# For traditional analysis (all directions combined):
# results = run_directional_threshold_analysis(
#     data_raw, 
#     specific_amp=5.0, 
#     target_directions='all',  # This will use the original method
#     threshold_methods=['Otsu', 'Yen']
# )

"""
the directional mean/std analysis

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Dict, Tuple
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def analyze_directional_distances(data_raw: dict, S: np.ndarray, _dist: float, 
                                plot_background_std: bool = True,
                                plot_normalized_difference: bool = True,
                                plot_box_with_thresholds: bool = True,
                                threshold_method: str = 'Otsu',
                                labels: np.ndarray = None,
                                **direction_groups) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Analyze how average/std of directional distances change with amplitude
    
    Parameters:
    -----------
    data_raw : dict
        Dictionary with amp keys containing neighbor and distance data
    S : np.ndarray
        Coordinate array
    _dist : float
        Step distance for direction calculation
    plot_background_std : bool, default True
        Whether to plot overall std (all directions) as background
    plot_normalized_difference : bool, default True
        Whether to plot normalized difference between two groups (only works with exactly 2 groups)
    plot_box_with_thresholds : bool, default True
        Whether to plot box plots with thresholds as first subplot (only works with exactly 2 groups)
    threshold_method : str, default 'Otsu'
        Threshold method to use ('Otsu', 'Yen', 'Li', 'Triangle')
    **direction_groups : keyword arguments
        Each argument represents a direction group:
        - key: group name (will be used in legend)
        - value: int (single direction) or list of ints (multiple directions)
        
    Example usage:
    analyze_directional_distances(
        data_raw, S, _dist,
        North=2,
        Cardinal=[2, 4, 6, 8],
        Diagonal=[1, 3, 7, 9],
        East_West=[4, 6]
    )
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with columns: ['amp', 'direction_group', 'mean', 'std', 'count']
        Plus 'overall_std', 'normalized_difference', and 'threshold' if applicable
    fig : matplotlib.Figure
        Figure with subplots showing box plots (if 2 groups), mean and std vs amp
    """
    
    if not direction_groups:
        raise ValueError("At least one direction group must be specified")
    
    # Get sorted amplitude values
    amps = sorted(data_raw.keys())
    
    # Initialize results storage
    results = []
    overall_stats = {}  # Store overall statistics for background plotting
    box_plot_data = {}  # Store data for box plots
    threshold_data = {}  # Store threshold data
    
    # Process each amplitude
    for amp in amps:
        print(f"Processing amplitude: {amp}")
        
        # Calculate overall statistics (all directions) for background
        if plot_background_std or plot_normalized_difference:
            all_directions_list = [1, 2, 3, 4, 6, 7, 8, 9]  # All 8 directions
            overall_direction_info = extract_direction_info(amp, all_directions_list, data_raw, S, _dist)
            
            # Collect all distances from all directions
            all_overall_distances = []
            for direction in all_directions_list:
                if direction in overall_direction_info:
                    distances = overall_direction_info[direction]['distances']
                    valid_distances = distances[~np.isnan(distances)]
                    all_overall_distances.extend(valid_distances.tolist())
            
            if len(all_overall_distances) > 0:
                overall_std = np.std(all_overall_distances)
                overall_mean = np.mean(all_overall_distances)
            else:
                overall_std = np.nan
                overall_mean = np.nan
            
            overall_stats[amp] = {'std': overall_std, 'mean': overall_mean}
        
        # Initialize box plot data for this amplitude
        if plot_box_with_thresholds and len(direction_groups) == 2:
            box_plot_data[amp] = {}
            threshold_data[amp] = {}
        
        # Process each direction group
        for group_name, directions in direction_groups.items():
            # Ensure directions is a list
            if isinstance(directions, int):
                target_directions = [directions]
            else:
                target_directions = list(directions)
            
            # Extract direction information for all target directions
            direction_info = extract_direction_info(amp, target_directions, data_raw, S, _dist)
            
            # Collect all valid distances for this group
            all_distances = []
            
            if len(target_directions) == 1:
                # Single direction
                direction = target_directions[0]
                if direction in direction_info:
                    distances = direction_info[direction]['distances']
                    valid_distances = distances[~np.isnan(distances)]
                    all_distances = valid_distances.tolist()
            else:
                # Multiple directions - merge distances
                for direction in target_directions:
                    if direction in direction_info:
                        distances = direction_info[direction]['distances']
                        valid_distances = distances[~np.isnan(distances)]
                        all_distances.extend(valid_distances.tolist())
            
            # Calculate statistics
            if len(all_distances) > 0:
                mean_dist = np.mean(all_distances)
                std_dist = np.std(all_distances)
                count = len(all_distances)
                
                # Calculate threshold for box plot
                if plot_box_with_thresholds and len(direction_groups) == 2:
                    try:
                        thresholds = compute_thresholds(np.array(all_distances))
                        threshold_value = thresholds.get(threshold_method, np.nan)
                    except:
                        threshold_value = np.nan
                    
                    box_plot_data[amp][group_name] = all_distances
                    threshold_data[amp][group_name] = threshold_value
            else:
                mean_dist = np.nan
                std_dist = np.nan
                count = 0
                threshold_value = np.nan
                print(f"  Warning: No valid distances found for {group_name}")
                
                if plot_box_with_thresholds and len(direction_groups) == 2:
                    box_plot_data[amp][group_name] = []
                    threshold_data[amp][group_name] = np.nan
            
            # Store results
            result_dict = {
                'amp': amp,
                'direction_group': group_name,
                'mean': mean_dist,
                'std': std_dist,
                'count': count
            }
            
            # Add overall std if requested
            if plot_background_std or plot_normalized_difference:
                result_dict['overall_std'] = overall_stats[amp]['std']
                result_dict['overall_mean'] = overall_stats[amp]['mean']
            
            # Add threshold if calculated
            if plot_box_with_thresholds and len(direction_groups) == 2:
                result_dict['threshold'] = threshold_value
            
            results.append(result_dict)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate normalized difference if exactly 2 groups and requested
    if plot_normalized_difference and len(direction_groups) == 2:
        group_names = list(direction_groups.keys())
        
        # Calculate difference between means of the two groups
        diff_results = []
        for amp in amps:
            amp_data = df[df['amp'] == amp]
            if len(amp_data) == 2:
                group1_mean = amp_data[amp_data['direction_group'] == group_names[0]]['mean'].iloc[0]
                group2_mean = amp_data[amp_data['direction_group'] == group_names[1]]['mean'].iloc[0]
                overall_std_val = amp_data['overall_std'].iloc[0] if 'overall_std' in amp_data.columns else np.nan
                
                if not (np.isnan(group1_mean) or np.isnan(group2_mean) or np.isnan(overall_std_val) or overall_std_val == 0):
                    normalized_diff = abs(group1_mean - group2_mean) / overall_std_val
                else:
                    normalized_diff = np.nan
                
                diff_results.append({
                    'amp': amp,
                    'normalized_difference': normalized_diff,
                    'raw_difference': abs(group1_mean - group2_mean) if not (np.isnan(group1_mean) or np.isnan(group2_mean)) else np.nan
                })
        
        diff_df = pd.DataFrame(diff_results)
        # Merge with main DataFrame
        df = df.merge(diff_df, on='amp', how='left')
    
    elif plot_normalized_difference and len(direction_groups) != 2:
        print("Warning: Normalized difference can only be calculated with exactly 2 direction groups.")
    
    # Create visualization with two-row layout when box plot is enabled
    if plot_box_with_thresholds and len(direction_groups) == 2:
        # Two-row layout: box plot on top, other plots below
        n_plots_bottom = 2
        if plot_normalized_difference:
            n_plots_bottom = 3
        
        fig = plt.figure(figsize=(7*n_plots_bottom, 12))
        
        # Use subplot2grid for spanning columns
        # First row: box plot (spans all columns)
        ax_box = plt.subplot2grid((2, n_plots_bottom), (0, 0), colspan=n_plots_bottom)
        
        # Second row: other plots
        ax1 = plt.subplot2grid((2, n_plots_bottom), (1, 0))
        ax2 = plt.subplot2grid((2, n_plots_bottom), (1, 1))
        
        axes = [ax_box, ax1, ax2]
        
        if plot_normalized_difference:
            ax3 = plt.subplot2grid((2, n_plots_bottom), (1, 2))
            axes.append(ax3)
    else:
        # Original single-row layout
        n_plots = 2
        if plot_normalized_difference and len(direction_groups) == 2:
            n_plots = 3
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
    
    plot_idx = 0
    
    # Plot box plots with thresholds (first row if enabled and 2 groups)
    if plot_box_with_thresholds and len(direction_groups) == 2:
        # Use the box plot axis
        group_names = list(direction_groups.keys())
        colors = colors_cover_2[:2]
        
        # Prepare data for box plot
        box_positions = []
        box_data = []
        box_colors = []
        threshold_positions = []
        threshold_values = []
        threshold_colors = []
        
        for i, amp in enumerate(amps):
            for j, group_name in enumerate(group_names):
                if amp in box_plot_data and group_name in box_plot_data[amp]:
                    pos = i * len(group_names) + j
                    box_positions.append(pos)
                    box_data.append(box_plot_data[amp][group_name])
                    box_colors.append(colors[j])
                    
                    # Add threshold line
                    if amp in threshold_data and group_name in threshold_data[amp]:
                        threshold_val = threshold_data[amp][group_name]
                        if not np.isnan(threshold_val):
                            threshold_positions.append(pos)
                            threshold_values.append(threshold_val)
                            threshold_colors.append(colors[j])
        
        # Create box plot
        if box_data:
            bp = ax_box.boxplot(box_data, positions=box_positions, widths=0.35, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Prepare threshold data for line plots
            threshold_data_by_group = {}
            for group_name in group_names:
                threshold_data_by_group[group_name] = {'x': [], 'y': []}
            
            for i, (pos, thresh_val, color) in enumerate(zip(threshold_positions, threshold_values, threshold_colors)):
                # Determine which group this threshold belongs to
                group_idx = pos % len(group_names)
                group_name = group_names[group_idx]
                
                threshold_data_by_group[group_name]['x'].append(pos)
                threshold_data_by_group[group_name]['y'].append(thresh_val)
            
            # Plot threshold lines for each group
            for i, group_name in enumerate(group_names):
                if threshold_data_by_group[group_name]['x']:
                    ax_box.plot(threshold_data_by_group[group_name]['x'], 
                               threshold_data_by_group[group_name]['y'], 
                               color=colors[i], linestyle='--', marker='o', 
                               linewidth=2, markersize=4,
                               label=f'{group_name} {threshold_method} Threshold')
        
        # Set x-axis labels
        amp_positions = [i * len(group_names) + 0.5 for i in range(len(amps))]
        ax_box.set_xticks(amp_positions)
        ax_box.set_xticklabels([f'{amp}' for amp in amps], rotation = 45)
        ax_box.set_xlabel('Amplitude')#, fontsize=12)
        ax_box.set_ylabel('Distance')#, fontsize=12)
        ax_box.set_title(f'Distance Distribution & Thresholds ({threshold_method})')#, fontsize=14)
        ax_box.grid(True, alpha=0.3)
        
        # Create legend (combine box colors and threshold lines)
        legend_elements = []
        for i, group_name in enumerate(group_names):
            # Box plot legend
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=f'{group_name} Distribution'))
            # Threshold line legend (only if there's threshold data)
            if threshold_data_by_group[group_name]['x']:
                legend_elements.append(plt.Line2D([0], [0], color=colors[i], linestyle='--', marker='o', 
                                                 linewidth=2, markersize=4, label=f'{group_name} {threshold_method} Threshold'))
        ax_box.legend(handles=legend_elements, loc='upper right')
        
        plot_idx = 1  # Next plots start from index 1
    
    # Second row plots (or first row if no box plot)
    ax1 = axes[plot_idx]
    ax2 = axes[plot_idx + 1]
    plot_idx += 2
    
    # Define colors for different groups using custom palette
    colors = colors_cover_2[:len(direction_groups)]
    color_map = dict(zip(direction_groups.keys(), colors))
    
    # Plot mean distances
    if plot_background_std and 'overall_std' in df.columns:
        # Plot overall std as background
        overall_data = df.groupby('amp')['overall_std'].first()
        ax1.fill_between(overall_data.index, 0, overall_data.values, 
                        alpha=0.65, color='lightgray', label='Overall Std. (background)')
    
    for group_name in direction_groups.keys():
        group_data = df[df['direction_group'] == group_name]
        if not group_data.empty:
            ax1.plot(group_data['amp'], group_data['mean'], 
                    'o-', color=color_map[group_name], label=group_name, 
                    linewidth=2, markersize=6)
    
    ax1.set_xlabel('Amplitude')# , fontsize=12)
    ax1.set_ylabel('Mean Distance')# , fontsize=12)
    ax1.set_title('Mean Distance vs Amplitude')# , fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot std distances
    if plot_background_std and 'overall_std' in df.columns:
        # Plot overall std as background
        overall_data = df.groupby('amp')['overall_std'].first()
        ax2.plot(overall_data.index, overall_data.values, 
                '--', color='lightgray', linewidth=3, alpha=0.7, label='Overall Std.')
    
    for group_name in direction_groups.keys():
        group_data = df[df['direction_group'] == group_name]
        if not group_data.empty:
            ax2.plot(group_data['amp'], group_data['std'], 
                    'o-', color=color_map[group_name], label=group_name, 
                    linewidth=2, markersize=6)
    
    ax2.set_xlabel('Amplitude')# , fontsize=12)
    ax2.set_ylabel('Standard Deviation of Distance')# , fontsize=12)
    ax2.set_title('Distance Std. vs Amplitude')# , fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot normalized difference if applicable
    if plot_normalized_difference and len(direction_groups) == 2 and 'normalized_difference' in df.columns:
        ax3 = axes[plot_idx]
        
        diff_data = df.groupby('amp')['normalized_difference'].first()
        valid_data = diff_data.dropna()
        
        if not valid_data.empty:
            ax3.plot(valid_data.index, valid_data.values, 
                    'o-', color='red', linewidth=2, markersize=6, 
                    label=f'|{list(direction_groups.keys())[0]} - {list(direction_groups.keys())[1]}| / Overall Std.')
            ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Reference (1.0)')
        
        ax3.set_xlabel('Amplitude') # , fontsize=12)
        ax3.set_ylabel('Normalized Difference')# , fontsize=12)
        ax3.set_title('Normalized Difference vs Amplitude')# , fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Add labels if provided
    if labels is not None:
        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.xaxis.set_minor_locator(MultipleLocator(1)) 
            ax.text(0.02, 1.02, f'({label})', transform=ax.transAxes, fontsize=26, fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    
    plt.tight_layout()
    
    # Print summary
    print("\nSummary:")
    print(f"Analyzed {len(amps)} amplitude values")
    print(f"Direction groups: {list(direction_groups.keys())}")
    for group_name, directions in direction_groups.items():
        print(f"  {group_name}: {directions}")
    if plot_box_with_thresholds and len(direction_groups) == 2:
        print(f"Box plot with {threshold_method} thresholds included")
    
    return df, fig

def print_direction_summary(df: pd.DataFrame):
    """Print a summary table of the analysis results"""
    print("\nDetailed Results:")
    print("=" * 80)
    
    for group in df['direction_group'].unique():
        group_data = df[df['direction_group'] == group]
        print(f"\n{group}:")
        print(f"{'Amp':<8} {'Mean':<10} {'Std':<10} {'Count':<8}")
        print("-" * 40)
        for _, row in group_data.iterrows():
            print(f"{row['amp']:<8} {row['mean']:<10.4f} {row['std']:<10.4f} {row['count']:<8}")

# Example usage function
# def example_usage():
#     """
#     Example of how to use the analysis function
#     """
#     # Example call (you would replace with your actual data)
#     # df, fig = analyze_directional_distances(
#     #     data_raw, S, _dist,
#     #     North=2,
#     #     South=8,
#     #     East=6,
#     #     West=4,
#     #     Cardinal=[2, 4, 6, 8],
#     #     Diagonal=[1, 3, 7, 9],
#     #     Horizontal=[4, 6],
#     #     Vertical=[2, 8],
#     #     All_Directions=[1, 2, 3, 4, 6, 7, 8, 9]
#     # )
#     # print_direction_summary(df)
#     # plt.show()
#     pass