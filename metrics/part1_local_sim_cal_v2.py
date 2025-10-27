import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import jaccard_score
import gc
import os
import json
import numexpr as ne
import time

from numba import jit

@jit(nopython=True)
def fast_merge_scale(base_data, S, amp, X_cols):
    n_rows = base_data.shape[0]
    s_cols = S.shape[1]
    result = np.empty((n_rows, X_cols + s_cols))
    
    # Scale and copy X portion
    for i in range(n_rows):
        for j in range(X_cols):
            result[i, j] = base_data[i, j] * amp
    
    # Copy S portion
    for i in range(n_rows):
        for j in range(s_cols):
            result[i, X_cols + j] = S[i, j]
    
    return result


def calculate_local_similarity(data1, data2, dx, dy, neighbors1_cache=None, metric='euclidean'):
    """  
    INPUT: 
    data1: array of (X, Y) with the shape of (N, 2)
    data2: feature array of (X, Y, feature) with the shape of (N, M+2)
    dx, dy: range of searching neighbors in data1    
    metric: the method of finding neighbors in data2

    OUTPUT:
    nei_1: neighbors1_list, for neighbors searched in data1
    dist_matrix: dist_matrix, calculated for data2
    dist_2_of_nei_1: dist2_of_neighbors1_list, distance between nei_1 and the point i in data2
    nei_2: neighbors2_list, searched neighbors around point i in data 2
    dist_2: dist2_list, corrsponding distance of nei_2
    jaccard_scores: jaccard_scores, similarity between nei_1 and nei_2
    """


    N = data1.shape[0]
    neighbors1_list = []
    dist2_of_neighbors1_list = []
    neighbors2_list = []
    dist2_list = []
    jaccard_scores = []

    # Step 1: Build full distance matrix in data2
    dist_matrix = pairwise_distances(data2, metric = metric)  # shape (N, N)

    # Step 2: get nei_1 by local window in data1
    for i in range(N):
        # Use cached neighbors if available
        if neighbors1_cache is None:
            if i == 0:
                print('Neighbor_1 is not cached!')
            xi, yi = data1[i]
            mask_x = np.abs(data1[:, 0] - xi) <= dx * 1.5
            mask_y = np.abs(data1[:, 1] - yi) <= dy * 1.5
            mask = mask_x & mask_y
            nei_1 = np.where(mask)[0]
            neighbors1_list.append(nei_1)
        else:
            if i == 0:
                print('Neighbor_1 is cached!')
            nei_1 = neighbors1_cache[i]

        # Step 3: dist_2_of_nei_1
        dist_to_neighbors1 = dist_matrix[i, nei_1]
        dist2_of_neighbors1_list.append(dist_to_neighbors1)

        max_dist = np.max(dist_to_neighbors1)
        # nei_2: all points within max_dist in data2 space
        nei_2 = np.where(dist_matrix[i] <= max_dist)[0]
        neighbors2_list.append(nei_2)
        dist2 = dist_matrix[i, nei_2]
        dist2_list.append(dist2)

        # Step 4: Jaccard similarity between nei_1 and nei_2
        # Convert neighbors_list to set
        set1 = set(nei_1)
        set2 = set(nei_2)

        intersection = set1 & set2
        union = set1 | set2

        # when both lists are empty, define the score as 0 
        # since the result is not meaningful
        jac_score = len(intersection) / len(union) if len(union) > 0 else 0.0

        jaccard_scores.append(jac_score)

    return {
        'nei_1': neighbors1_cache if neighbors1_cache is not None else neighbors1_list,
        'dist_matrix': dist_matrix,
        'dist_2_of_nei_1': dist2_of_neighbors1_list,
        'nei_2': neighbors2_list,
        'dist_2': dist2_list,
        'jaccard_scores': jaccard_scores
    }

def find_grid_spacing_robust(points, tolerance=1e-6):
    """
    More robust method using histogram of differences
    """
    X = points[:, 0]
    Y = points[:, 1]
    
    # Calculate all pairwise differences
    X_diffs = []
    Y_diffs = []
    
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dx_temp = abs(X[i] - X[j])
            dy_temp = abs(Y[i] - Y[j])
            
            if dx_temp > tolerance:
                X_diffs.append(dx_temp)
            if dy_temp > tolerance:
                Y_diffs.append(dy_temp)
    
    # Find the most common small difference (likely the grid spacing)
    X_diffs = np.array(X_diffs)
    Y_diffs = np.array(Y_diffs)
    
    # Use histogram to find most frequent spacing
    if len(X_diffs) > 0:
        hist_x, bins_x = np.histogram(X_diffs, bins=50)
        dx = bins_x[np.argmax(hist_x)]
    else:
        dx = 0
        
    if len(Y_diffs) > 0:
        hist_y, bins_y = np.histogram(Y_diffs, bins=50)
        dy = bins_y[np.argmax(hist_y)]
    else:
        dy = 0
    
    return dx, dy

def main_calculation(save_path_0, save_folder, X, S, coor_merge_flag,
                     target_name = 'similarity_score.pickle', amp_values = [0]):
    
    save_path = save_path_0 + save_folder
    os.makedirs(save_path, exist_ok=True)
    
    save_target = save_path + "/" + target_name

    # check the size of X
    print(f"Data shape: {X.shape}")
    
    # SINGLE NORMALIZATION: Normalize X once and reuse
    print("Normalizing X once...")
    X_min = np.min(X)
    X_max = np.max(X)
    X_normalized = (X - X_min) / (X_max - X_min)
    print(f"Normalized data has the min and max as {np.min(X_normalized)}, {np.max(X_normalized)}")
    
    # Clear original X to save memory
    del X
    gc.collect()

    # find the dx, dy in spatial spacings
    dx, dy = find_grid_spacing_robust(S)
    print(f'the spatial spacing between points are ({dx}, {dy})')

    # check the size
    if not X_normalized.shape[0] == S.shape[0]:
        print(f"The calculation with {save_folder} is stopped!")
        return
    else: 
        print(f"Now working for {save_folder}")

    # PRE-MERGE: Prepare base data before the loop
    if coor_merge_flag:
        X_cols = X_normalized.shape[1]
        # Pre-merge X_normalized with S once
        base_data = np.hstack((X_normalized, S))
        del X_normalized
        gc.collect()
    else:
        X_cols = X_normalized.shape[1]
        base_data = X_normalized
        del X_normalized
        gc.collect()

    # PRE_CACHED: before the amplitude loop, compute this ONCE
    print("Pre-computing spatial neighbors...")
    neighbors1_cache = []
    for i in range(S.shape[0]):
        xi, yi = S[i]
        mask_x = np.abs(S[:, 0] - xi) <= dx * 1.5
        mask_y = np.abs(S[:, 1] - yi) <= dy * 1.5
        mask = mask_x & mask_y
        nei_1 = np.where(mask)[0]
        neighbors1_cache.append(nei_1)

    # calculation
    # Warm up Numba
    if coor_merge_flag:
        print("Warming up Numba function...")
        _ = fast_merge_scale(base_data[:100], S[:100], 1.0, X_cols)
        print("Numba ready")

    results_dict = {}

    for amp in amp_values:
        print(f"Now amp is {amp}")
        
        # Timing: Array creation
        array_start = time.time()
        
        if coor_merge_flag:
            X_updated = fast_merge_scale(base_data, S, amp, X_cols)
            # X_updated = np.empty((base_data.shape[0], X_cols + S.shape[1])\
            #                      , dtype=base_data.dtype)
            
            # # Use numexpr for parallel scaling
            # X_part = base_data[:, :X_cols]
            # X_updated[:, :X_cols] = ne.evaluate('X_part * amp')
            # X_updated[:, X_cols:] = S
        else:
            X_updated = base_data * amp

        print(f"Now, data has a shape of {np.shape(X_updated)}")
        array_time = time.time() - array_start    
        print(f"Array creation time: {array_time:.4f} seconds")

        calc_start = time.time()
        result = calculate_local_similarity(data1=S, data2=X_updated,
                                            neighbors1_cache=neighbors1_cache, 
                                          dx=dx, dy=dy, metric=metric)
        calc_time = time.time() - calc_start
        print(f"Similarity calculation time: {calc_time:.4f} seconds")
        results_dict[amp] = result
        
        # PROGRESSIVE CLEANUP: Clear intermediate arrays after each amplitude
        del X_updated
        # Note: 'result' is stored in results_dict, so we don't delete it
        # But if result contains large intermediate arrays, we could clean those:
        # if 'dist_matrix' in result:
        #     del result['dist_matrix']  # Remove the large distance matrix if not needed
        gc.collect()
        
        print(f"Completed amp {amp}, memory cleaned up")

    # Final save
    with open(save_target, "wb") as f:
        pickle.dump(results_dict, f)
    
    # save some info into the .json
    settings = {
        "dx": float(dx),
        "dy": float(dy),
        "plot_config": {
            "mathtext.fontset": "stix",          # Math text font
            "font.family": "STIXGeneral",        # General font family
            "axes.unicode_minus": False,         # Disable unicode minus
            "font.size": 14,                     # General font size
            "axes.titlesize": 16,                # Title font size
            "axes.labelsize": 14,                # X and Y label font size
            "xtick.labelsize": 12,               # X tick label font size
            "ytick.labelsize": 12,               # Y tick label font size
            "legend.fontsize": 12,               # Legend font size
            "figure.titlesize": 18               # Figure title font size
        }
    }

    script_dir = os.getcwd()  
    with open(script_dir + '/' + 'config.json', 'w') as f:
        json.dump(settings, f, indent = 4)

    print(f"All calculations completed and saved to {save_target}")


# load the library used here
import numpy as np
import umap
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


# Get the path to the parent folder of the notebook
script_dir = os.getcwd()  # should be 'project/notebooks'
utils_path = os.path.abspath(os.path.join(script_dir, '..', 'utilis'))
print(script_dir)
# Add to sys.path
if utils_path not in sys.path:
    sys.path.append(utils_path)

from function import *
from utilis import *  

# 
from pathlib import Path 
import pickle

####################### input ###########################
# as, s
# 1,  3;  1
# 4,  6;  2
# 8,  9;  3

from part1_data_lodaer import *
# 1. 
data_type = 'as'
sample_se = '2'
# X, S = data_loader_ovpe_2021_optimized(data_type=data_type)
# X, S = data_loader_hvpe_2021_optimized(data_type=data_type)
X, S = data_loader_ffc_2024_optimized(data_type=data_type, sample_se = sample_se)

# 2.
save_folder_0 = f'{script_dir}/results_folder/'
# save_folder = '20250908_OVPE_2021_' + data_type 
# save_folder = '20250909_HVPE_2021_' + data_type 
save_folder = f'20251002_ffc_2024_{data_type}_{sample_se}' 
target_name = 'similarity_score.pickle'

# 3.
# amp_values = np.arange(0, 10.1, 0.1) # ovpe

# amp_values = np.arange(10.1, 30.1, 0.1)
amp_values = np.arange(0, 30.5, 0.5)

# 4. 
coor_merge_flag = True # false means only XRD

# 5.
metric = 'euclidean' # 'euclidean' in default


##############################################################

if __name__ == "__main__" : 
    # loop over specific files and save target
    main_calculation(
        save_path_0 = save_folder_0,
        save_folder = save_folder,
        X= X,
        S = S,
        coor_merge_flag = coor_merge_flag,
        target_name = target_name,
        amp_values = amp_values
    )