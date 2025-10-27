# function of selecting points by various K
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

# class comparison_K_std():
#     def __init__(self, path_of_umap_15d,  K,  S, dim = 2):
#         self.path_of_umap_15d = path_of_umap_15d
#         self.K = K
#         self.S = S
#         self.dim = dim


#     def _adj_search(self, data = None,  _dist = None):
#         if data == None:
#             data = self.S
#         _adj_matrix = np.zeros([len(data), len(data)], dtype = np.int8)    

#         for i in range(len(data)):
#             for j in range(len(data)):

#                 # skip the case of raw point
#                 if i == j:
#                     continue
                
#                 x_i, y_i = data[i]
#                 x_j, y_j = data[j]
#                 if abs(x_i - x_j) <= _dist and abs(y_i - y_j) <= _dist:
#                     _adj_matrix[i, j] = 1
    

#         return _adj_matrix

#     # build the KNN graph of UMAP results

#     def umap_high_d_read(self, ):
#         path_of_umap_15d = self.path_of_umap_15d
#         dim = self.dim

#         if dim > 2:
#             _df_umap = pd.read_csv(path_of_umap_15d, index_col=None, header=None)
#         elif dim == 2:
#             _df_umap = pd.read_csv(path_of_umap_15d, index_col=0, header=0)
        
#         df_arr = _df_umap.values

#         print(df_arr.shape)

#         return df_arr

#     def KNN_anomaly_finding(self,adj_matrix = None,  data_high_dim = None, K_input = None):
    

#         # import the umap plot for the KNN graph
#         if data_high_dim == None:
#             _arr_umap_15d = self.umap_high_d_read()
#         else:
#             _arr_umap_15d = data_high_dim

#         if K_input == None:
#             K = self.K
#         else:
#             K = K_input

#         print(f'Now K is {K}')

#         num_nodes = _arr_umap_15d.shape[0]
#         _KNN_umap = NearestNeighbors(n_neighbors=K, metric='euclidean')
#         _KNN_umap.fit(_arr_umap_15d)
#         _neighbors_umap_15d = _KNN_umap.kneighbors(return_distance=False)

#         _adj_umap_15d = np.zeros((num_nodes, num_nodes), dtype = int)
#         for i, neighbors_index in enumerate(_neighbors_umap_15d):
#             _adj_umap_15d[i, neighbors_index] = 1

#         # symmetrize the matrix
#         _adj_umap_15d_sm = np.maximum(_adj_umap_15d, _adj_umap_15d.T)


#         # find those in adj_matrix, but not in _adj_umap_15d
#         record_matrix_1 = np.zeros((num_nodes, num_nodes), dtype = int)
#         record_matrix_2 = np.zeros((num_nodes, num_nodes), dtype = int)
#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 if adj_matrix[i, j] == 1 and _adj_umap_15d_sm[i, j] == 0:
#                     # print(i, j, '1')
#                     record_matrix_1[i, j] = 1 # in round but not in umap
#                 elif adj_matrix[i, j] == 0 and _adj_umap_15d_sm[i, j] == 1:
#                     # print(i, j, '2')
#                     record_matrix_2[i, j] = 2 # not in round but in umap

#         record_matrix_1_sum = np.sum(record_matrix_1, axis = 0)

#         max = np.max(record_matrix_1_sum)

#         # for i in range(max + 1):
#         #     num_i = np.sum(record_matrix_1_sum == i)
#         #     print(f"{i}:, {num_i} of K = {K}")

#         indices_anomaly = np.where(record_matrix_1_sum > 0)[0]

#         return indices_anomaly


#     def top_std_selection(self, std_values, percentage_threshold):
#         """
#         Selects indices of points with the highest standard deviation.

#         Parameters:
#             std_values (array): Standard deviation values for each point.
#             percentage_threshold (float): Top X% threshold (e.g., 0.1 for top 10%).

#         Returns:
#             selected_indices (set): Indices of selected points.
#         """
#         num_selected = int(len(std_values) * percentage_threshold)
#         if num_selected == 0:
#             return set()
        
#         # Get the indices of the top `num_selected` std values
#         top_indices = np.argsort(std_values)[:num_selected]   
#         return set(top_indices)


#     def compare_selection_overlap(self, data_high_dim = None, data_std_values = None
#                                   , K_values = None, percentage_thresholds = None
#                                   , _adj_search_dist = 5):
#         """
#         Compares the overlap between selections from std-based and K-based methods.

#         Parameters:
#             raw_data (array): The original data used for selection_by_K.
#             std_values (array): Standard deviation values for each point.
#             selection_by_K (function): Function that returns an array of selected indices for a given K.
#             K_values (list): List of K values to test.
#             percentage_thresholds (list): List of percentage thresholds for std-based selection.

#         Returns:
#             overlap_results (dict): Dictionary with {(K, percentage)} as keys and Jaccard similarity as values.
#         """
#         overlap_results = {}
#         adj_matrix = self._adj_search(_dist = _adj_search_dist)
#         overlap_indices = {}
#         selected_by_K_ind = {}
#         selected_by_pct_ind = {}

#         for K in K_values:
#             selected_by_K = \
#                 set(self.KNN_anomaly_finding(adj_matrix = adj_matrix, data_high_dim = data_high_dim
#                                                       ,  K_input = K))  # Convert to set
            
#             selected_by_K_ind[K] = selected_by_K

#             for pct in percentage_thresholds:
#                 selected_by_std = self.top_std_selection(data_std_values, pct)  # Get indices from std-based method
                
#                     # Compute Jaccard similarity
#                 intersection = selected_by_K & selected_by_std
#                 union = selected_by_K | selected_by_std
#                 jaccard_score = len(intersection) / len(union) if union else 0

#                 # Store results
#                 overlap_results[(K, pct)] = jaccard_score
#                 overlap_indices[(K, pct)] = intersection  # Store overlapping indices
#                 selected_by_pct_ind[(K, pct)]  = selected_by_std

#         return overlap_results, overlap_indices, selected_by_K_ind, selected_by_pct_ind
    

class comparison_K_fixed_pct():
    def __init__(self, path_of_umap_15d,  K,  S, dim = 2):
        self.path_of_umap_15d = path_of_umap_15d
        self.K = K
        self.S = S
        self.dim = dim


    def _adj_search(self, data = None,  _dist = None):
        if data == None:
            data = self.S
        _adj_matrix = np.zeros([len(data), len(data)], dtype = np.int8)    

        for i in range(len(data)):
            for j in range(len(data)):

                # skip the case of raw point
                if i == j:
                    continue
                
                x_i, y_i = data[i]
                x_j, y_j = data[j]
                if abs(x_i - x_j) <= _dist and abs(y_i - y_j) <= _dist:
                    _adj_matrix[i, j] = 1
    

        return _adj_matrix

    # build the KNN graph of UMAP results

    def umap_high_d_read(self, ):
        path_of_umap_15d = self.path_of_umap_15d
        dim = self.dim

        if dim > 2:
            _df_umap = pd.read_csv(path_of_umap_15d, index_col=None, header=None)
        elif dim == 2:
            _df_umap = pd.read_csv(path_of_umap_15d, index_col=0, header=0)
        
        df_arr = _df_umap.values

        print(df_arr.shape)

        return df_arr

    def KNN_anomaly_finding(self,adj_matrix = None,  data_high_dim = None, K_input = None):
    

        # import the umap plot for the KNN graph
        if data_high_dim == None:
            _arr_umap_15d = self.umap_high_d_read()
        else:
            _arr_umap_15d = data_high_dim

        if K_input == None:
            K = self.K
        else:
            K = K_input

        print(f'Now K is {K}')

        num_nodes = _arr_umap_15d.shape[0]
        _KNN_umap = NearestNeighbors(n_neighbors=K, metric='euclidean')
        _KNN_umap.fit(_arr_umap_15d)
        _neighbors_umap_15d = _KNN_umap.kneighbors(return_distance=False)

        _adj_umap_15d = np.zeros((num_nodes, num_nodes), dtype = int)
        for i, neighbors_index in enumerate(_neighbors_umap_15d):
            _adj_umap_15d[i, neighbors_index] = 1

        # symmetrize the matrix
        _adj_umap_15d_sm = np.maximum(_adj_umap_15d, _adj_umap_15d.T)


        # find those in adj_matrix, but not in _adj_umap_15d
        record_matrix_1 = np.zeros((num_nodes, num_nodes), dtype = int)
        record_matrix_2 = np.zeros((num_nodes, num_nodes), dtype = int)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] == 1 and _adj_umap_15d_sm[i, j] == 0:
                    # print(i, j, '1')
                    record_matrix_1[i, j] = 1 # in round but not in umap
                elif adj_matrix[i, j] == 0 and _adj_umap_15d_sm[i, j] == 1:
                    # print(i, j, '2')
                    record_matrix_2[i, j] = 2 # not in round but in umap

        record_matrix_1_sum = np.sum(record_matrix_1, axis = 0)

        max = np.max(record_matrix_1_sum)

        # for i in range(max + 1):
        #     num_i = np.sum(record_matrix_1_sum == i)
        #     print(f"{i}:, {num_i} of K = {K}")

        indices_anomaly = np.where(record_matrix_1_sum > 0)[0]

        return indices_anomaly


    def compare_selection_overlap(self, data_high_dim = None, data_std_values = None
                                  , K_values = None, percentage_thresholds = None
                                  , _adj_search_dist = 5, coeff = 1):
        """
        Compares the overlap between selections from std-based and K-based methods.

        Parameters:
            raw_data (array): The original data used for selection_by_K.
            std_values (array): Standard deviation values for each point.
            selection_by_K (function): Function that returns an array of selected indices for a given K.
            K_values (list): List of K values to test.
            percentage_thresholds (list): List of percentage thresholds for std-based selection.

        Returns:
            overlap_results (dict): Dictionary with {(K, percentage)} as keys and Jaccard similarity as values.
        """
        overlap_results = {}
        adj_matrix = self._adj_search(_dist = _adj_search_dist)
        overlap_indices = {}
        selected_by_K_ind = {}
        selected_by_pct_ind = {}

        for K in K_values:
            selected_by_K = \
                set(self.KNN_anomaly_finding(adj_matrix = adj_matrix, data_high_dim = data_high_dim
                                                      ,  K_input = K))  # Convert to set
            
            selected_by_K_ind[K] = selected_by_K
            from skimage.filters import threshold_otsu
            threshold_ot = threshold_otsu(data_std_values) * coeff
            selected_by_std = set(np.where(data_std_values < threshold_ot)[0])
            intersection = selected_by_K & selected_by_std
            union = selected_by_K | selected_by_std
            jaccard_score = len(intersection) / len(union) if union else 0

            # Store results
            overlap_results[K] = jaccard_score
            overlap_indices[K] = intersection  # Store overlapping indices
            selected_by_pct_ind[K]  = selected_by_std

        return overlap_results, overlap_indices, selected_by_K_ind, selected_by_pct_ind, threshold_ot

class K_anonmaly_matrix_producer():
    def __init__(self, path_of_umap_15d,  K,  S, dim = 2):
        self.path_of_umap_15d = path_of_umap_15d
        self.K = K
        self.S = S
        self.dim = dim


    def umap_high_d_read(self, ):
        path_of_umap_15d = self.path_of_umap_15d
        dim = self.dim

        if dim > 2:
            _df_umap = pd.read_csv(path_of_umap_15d, index_col=None, header=None)
        elif dim == 2:
            _df_umap = pd.read_csv(path_of_umap_15d, index_col=0, header=0)
        
        df_arr = _df_umap.values

        return df_arr
    
    def _adj_search(self, data=None, _dist=1.0):
        if data is None:
            data = self.S

        tol = 0.5 * _dist
        N = len(data)
        _adj_matrix = np.zeros((N, N), dtype=np.int8)

        for i in range(N):
            x_i, y_i = data[i]
            for j in range(i + 1, N):  # upper triangle only
                x_j, y_j = data[j]

                dx = abs(x_i - x_j)
                dy = abs(y_i - y_j)

                if abs(dx) <= _dist + tol and abs(dy) <= _dist + tol and (dx > tol or dy > tol):
                    _adj_matrix[i, j] = 1
                    _adj_matrix[j, i] = 1  # ensure symmetry

        return _adj_matrix


    def KNN_anomaly_producer(self,adj_matrix = None,  data_high_dim = None
                             , K_input = None, _adj_search_dist = 5):
    

        # import the umap plot for the KNN graph
        if data_high_dim == None:
            _arr_umap_15d = self.umap_high_d_read()
        else:
            _arr_umap_15d = data_high_dim

        if K_input == None:
            K = self.K
        else:
            K = K_input

        if adj_matrix == None:
            adj_matrix = self._adj_search(_dist = _adj_search_dist)

        print(f'Now K is {K}')

        num_nodes = _arr_umap_15d.shape[0]
        _KNN_umap = NearestNeighbors(n_neighbors=K, metric='euclidean')
        _KNN_umap.fit(_arr_umap_15d)
        _neighbors_umap_15d = _KNN_umap.kneighbors(return_distance=False)

        _adj_umap_15d = np.zeros((num_nodes, num_nodes), dtype = int)
        for i, neighbors_index in enumerate(_neighbors_umap_15d):
            _adj_umap_15d[i, neighbors_index] = 1

        # symmetrize the matrix
        _adj_umap_15d_sm = np.maximum(_adj_umap_15d, _adj_umap_15d.T)


        # find those in adj_matrix, but not in _adj_umap_15d
        record_matrix_0 = np.zeros((num_nodes, num_nodes), dtype = int)
        record_matrix_1 = np.zeros((num_nodes, num_nodes), dtype = int)
        record_matrix_2 = np.zeros((num_nodes, num_nodes), dtype = int)
        record_matrix_3 = np.zeros((num_nodes, num_nodes), dtype = int)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] == 0 and _adj_umap_15d_sm[i, j] == 0:
                    # print(i, j, '1')
                    record_matrix_0[i, j] = 1 # not in round and not in umap
                elif adj_matrix[i, j] == 1 and _adj_umap_15d_sm[i, j] == 0:
                    # print(i, j, '1')
                    record_matrix_1[i, j] = 1 # in round but not in umap
                elif adj_matrix[i, j] == 0 and _adj_umap_15d_sm[i, j] == 1:
                    # print(i, j, '2')
                    record_matrix_2[i, j] = 1 # not in round but in umap
                elif adj_matrix[i, j] == 1 and _adj_umap_15d_sm[i, j] == 1:
                    # print(i, j, '2')
                    record_matrix_3[i, j] = 1 # in round and in umap

        return record_matrix_0, record_matrix_1, record_matrix_2, record_matrix_3
    

def direction_judger(indice_center, indice_neighboor, coors, step_x, step_y):
    """  
    Determines the relative position of a neighbor point with respect to a center point.

    INPUT: 
    indice_center: int, index of the center point in the coordinate array (X, Y) 
    indice_neighboor: int, index of the neighbor point 
                      whose direction is to be determined
    coors: array, coordinates (X, Y) with shape (NoP, 2)
    step_x, step_y: float, step intervals between neighboring points in the x- and y-directions

    OUTPUT:
    The relative position as follows:
        1 2 3
        4 c 6
        7 8 9
    Returns:
        int: a marker representing the relative position
        None: if the points are not neighbors
    """
    
    x_c, y_c = coors[indice_center]
    x_n, y_n = coors[indice_neighboor]

    # Compute relative movement normalized by step sizes and round to nearest integer
    marker_x = int(round((x_n - x_c) / step_x))
    marker_y = int(round((y_n - y_c) / step_y))

    # Ensure values are in the valid range before checking the dictionary
    if marker_x not in {-1, 0, 1} or marker_y not in {-1, 0, 1}:
        print(f"Points are not neighboring with marker_x = {marker_x}, marker_y = {marker_y}")
        return None

    # Mapping grid positions to numbers
    position_map = {
        (-1,  1): 1, (0,  1): 2, (1,  1): 3,
        (-1,  0): 4, (0,  0): 5, (1,  0): 6,
        (-1, -1): 7, (0, -1): 8, (1, -1): 9,
    }

    marker = position_map.get((marker_x, marker_y))

    if marker == 5:
        print("The input point is the center point!")

    return int(marker)

from sklearn.neighbors import BallTree


def NN_producer(data, _dist_x, _dist_y):
    """
    Finds nearest neighbors based on separate conditions for X and Y distances.

    Parameters:
        data (array): Coordinates (X, Y), shape (N, 2)
        _dist_x (float): Maximum allowed X distance for neighbor consideration
        _dist_y (float): Maximum allowed Y distance for neighbor consideration

    Returns:
        neighbors_dict (dict): {index: list of neighbor indices} based on X and Y distance conditions
    """
    N = len(data)
    neighbors_dict = {i: [] for i in range(N)}

    for i in range(N):
        x_i, y_i = data[i]

        for j in range(N):
            if i == j:
                continue  # Skip self
            x_j, y_j = data[j]
            tol_x = 0.5 * _dist_x
            tol_y = 0.5 * _dist_y
            # Check both X and Y distance conditions
            if abs(x_i - x_j) <= _dist_x + tol_x and abs(y_i - y_j) <= _dist_y + tol_y:
                neighbors_dict[i].append(j)  # Keep append for order

    return neighbors_dict




def jaccard_similarity_knn(data1, data2, neighbors_dict1, metric="euclidean", epsilon=1e-6):
    """
    Computes Jaccard similarity using position-dependent nearest neighbors.

    Parameters:
        data1 (array): Original coordinates (X, Y), shape (N, 2)
        data2 (array): UMAP-reduced coordinates (X, Y), shape (N, 2)
        neighbors_dict1 (dict): Neighbors from NN_producer(data1)
        metric (str): Distance metric for BallTree (default: "euclidean")
        decimal_places (int): Number of decimal places to round r_max_2 (default: 7)

    Returns:
        avg_jaccard (float): Average Jaccard similarity
        std_jaccard (float): Standard deviation of Jaccard similarity
        median_jaccard (float): Median Jaccard similarity
        jaccard_scores (array): Jaccard similarity per point (N,)
        neighbors_dict2 (dict): {index: set of neighbors} for `data2`
        distances_ls (list): List of truncated neighbor distances for each point in `data2`
        distances2_ls (list): List of distances from center points to their neighbors in `data2`
        r_max_2 (dict): Maximum distance of neighbors in `data2` (rounded up)
    """
    N = data1.shape[0]
    tree2 = BallTree(data2, metric=metric)

    distances_ls = []
    r_max_2 = {}

    distances2_ls = []

    distance_mismatches = {}  # find if there are any mismatches

    # Step 1: Compute r_max_2 (max distance of each point's neighbors in data2)
    for i in range(N):
        if not neighbors_dict1[i]:  # Handle case with no neighbors
            r_max_2[i] = 0
            continue
        
        neighbor_indices = list(neighbors_dict1[i])
        distances = np.linalg.norm(data2[neighbor_indices] - data2[i], axis=1)
        
        if distances.size > 0:
            max_distance = np.max(distances)
            r_max_2[i] = max_distance + epsilon  # Safe buffer for float32
        else:
            r_max_2[i] = 0

        distances_ls.append(distances)

    # Step 2: Find new neighbors in `data2` within r_max_2
    neighbors_dict2 = {}
    for i in range(N):
        if r_max_2[i] > 0:  # Ensure radius is meaningful
            neighbors_arr = tree2.query_radius([data2[i]], r=r_max_2[i])[0]
            ordered_neighbors = [j for j in neighbors_arr if j != i]

            # Compute distances between center and new_neighbors in `data2`
            neighbors_dict2[i] = ordered_neighbors 
            distances_to_neighbors = np.linalg.norm(data2[ordered_neighbors] - data2[i], axis=1)
            distances2_ls.append(distances_to_neighbors)
        
        else:
            neighbors_dict2[i] = set()  # No valid neighbors

    # compare to find the mismatches
    for i in range(N):
        common_neighbors = set(neighbors_dict1[i]) & set(neighbors_dict2[i])
        if not common_neighbors:
            continue

        # Get neighbor lists
        n1 = list(neighbors_dict1[i])
        n2 = list(neighbors_dict2[i])
        d1 = distances_ls[i]
        d2 = distances2_ls[i]

        # Map neighbor index to distance
        d1_map = {idx: dist for idx, dist in zip(n1, d1)}
        d2_map = {idx: dist for idx, dist in zip(n2, d2)}

        mismatches = []
        for j in common_neighbors:
            if j in d1_map and j in d2_map:
                dist1 = d1_map[j]
                dist2 = d2_map[j]
                if not np.isclose(dist1, dist2, rtol=1e-5, atol=1e-8):
                    mismatches.append((j, dist1, dist2))
            else:
                mismatches.append((j, d1_map.get(j, 'missing'), d2_map.get(j, 'missing')))

        if mismatches:
            distance_mismatches[i] = mismatches


    # Step 3: Compute Jaccard similarity
    jaccard_scores = np.array([
        0.0 if len(neighbors_dict1[i]) == 0 and len(neighbors_dict2[i]) == 0 else
        len(set(neighbors_dict1[i]) & set(neighbors_dict2[i])) / len(set(neighbors_dict1[i]) | set(neighbors_dict2[i]))
        for i in range(N)
    ])

    # Compute statistical summaries
    avg_jaccard = np.mean(jaccard_scores)
    std_jaccard = np.std(jaccard_scores)
    median_jaccard = np.median(jaccard_scores)

    return avg_jaccard, std_jaccard, median_jaccard,\
         jaccard_scores, neighbors_dict2,\
              distances_ls, distances2_ls, r_max_2, distance_mismatches




# lets' extract the boundary points!
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union

class BoundaryExtractor:
    """
    Extracts boundary points around specific labeled areas.
    Follows KISS, YAGNI, and SOLID principles.
    """
    
    def __init__(self, points: np.ndarray, step_x: int, step_y: int, labels: np.ndarray):
        """
        Initialize the boundary extractor.
        
        Args:
            points: Array of shape (N, 2) with (X, Y) coordinates
            step_x: Interval between adjacent points in X direction
            step_y: Interval between adjacent points in Y direction  
            labels: Array of labels where -1 indicates boundary, 1 indicates area
        """
        self.points = points
        self.step_x = step_x
        self.step_y = step_y
        self.labels = labels
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate input parameters."""
        if len(self.points) != len(self.labels):
            raise ValueError("Points and labels must have same length")
        if self.points.shape[1] != 2:
            raise ValueError("Points must have shape (N, 2)")
    
    def extract_boundaries(self, target_labels: Union[int, List[int]]) -> Tuple[np.ndarray, dict]:
        """
        Extract boundary points around areas with specific labels.
        
        Args:
            target_labels: Single label or list of labels to focus on
            
        Returns:
            boundary_points: Array of boundary coordinates
            area_boundaries: Dict mapping each target label to its boundary points
        """
        if isinstance(target_labels, int):
            target_labels = [target_labels]
        
        # Get boundary points (labeled as -1)
        boundary_mask = self.labels == -1
        boundary_points = self.points[boundary_mask]
        
        area_boundaries = {}
        all_boundary_points = []
        
        for label in target_labels:
            # Get points for this specific area
            area_mask = self.labels == label
            area_points = self.points[area_mask]
            
            if len(area_points) == 0:
                area_boundaries[label] = np.array([]).reshape(0, 2)
                continue
            
            # Find boundary points adjacent to this area
            adjacent_boundaries = self._find_adjacent_boundaries(boundary_points, area_points)
            area_boundaries[label] = adjacent_boundaries
            
            if len(adjacent_boundaries) > 0:
                all_boundary_points.append(adjacent_boundaries)
        
        # Combine all boundary points
        if all_boundary_points:
            combined_boundaries = np.vstack(all_boundary_points)
            # Remove duplicates
            combined_boundaries = np.unique(combined_boundaries, axis=0)
        else:
            combined_boundaries = np.array([]).reshape(0, 2)
        
        return combined_boundaries, area_boundaries
    
    def _find_adjacent_boundaries(self, boundary_points: np.ndarray, area_points: np.ndarray) -> np.ndarray:
        """
        Find boundary points that are adjacent to area points.
        
        Args:
            boundary_points: Potential boundary points
            area_points: Points in the target area
            
        Returns:
            Adjacent boundary points
        """
        if len(boundary_points) == 0 or len(area_points) == 0:
            return np.array([]).reshape(0, 2)
        
        adjacent_boundaries = []
        
        for boundary_point in boundary_points:
            if self._is_adjacent_to_area(boundary_point, area_points):
                adjacent_boundaries.append(boundary_point)
        
        return np.array(adjacent_boundaries) if adjacent_boundaries else np.array([]).reshape(0, 2)
    
    def _is_adjacent_to_area(self, boundary_point: np.ndarray, area_points: np.ndarray) -> bool:
        """
        Check if a boundary point is adjacent to any point in the area.
        Adjacent means distance is step_x or step_y in X or Y direction.
        
        Args:
            boundary_point: Single boundary point [x, y]
            area_points: Array of area points
            
        Returns:
            True if boundary point is adjacent to any area point
        """
        for area_point in area_points:
            dx = abs(boundary_point[0] - area_point[0])
            dy = abs(boundary_point[1] - area_point[1])
            
            # Check if adjacent in X direction (same Y, distance = step_x)
            if dx == self.step_x and dy == 0:
                return True
            
            # Check if adjacent in Y direction (same X, distance = step_y)  
            if dx == 0 and dy == self.step_y:
                return True
        
        return False
    
    def plot_results(self, target_labels: Union[int, List[int]], 
                    boundary_points: np.ndarray, area_boundaries: dict):
        """
        Create an overlapped plot showing points, areas, and extracted boundaries.
        
        Args:
            target_labels: Labels that were processed
            boundary_points: All extracted boundary points
            area_boundaries: Dictionary of boundaries per area
        """
        plt.figure(figsize=(10, 8))
        
        # Plot all points with their labels
        for label in np.unique(self.labels):
            mask = self.labels == label
            points_subset = self.points[mask]
            
            if label == -1:
                plt.scatter(points_subset[:, 0], points_subset[:, 1], 
                           c='gray', alpha=0.6, s=20, label='Boundary points')
            elif label in (target_labels if isinstance(target_labels, list) else [target_labels]):
                plt.scatter(points_subset[:, 0], points_subset[:, 1], 
                           c='blue', alpha=0.7, s=30, label=f'Area {label}')
            else:
                plt.scatter(points_subset[:, 0], points_subset[:, 1], 
                           c='lightgray', alpha=0.3, s=15, label=f'Other area {label}')
        
        # Highlight extracted boundaries
        if len(boundary_points) > 0:
            plt.scatter(boundary_points[:, 0], boundary_points[:, 1], 
                       c='red', s=50, marker='x', linewidths=2, 
                       label='Extracted boundaries', zorder=5)
        
        plt.xlabel('X')
        plt.ylabel('Y') 
        plt.title('Points, Areas, and Extracted Boundaries')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def indices_of_s_in_S(S, s):
    idxs = []
    for pt in s:
        idx = np.where((S == pt).all(axis=1))[0][0]
        idxs.append(idx)
    return np.array(idxs)
# area_bounds_idx = indices_of_s_in_S(S, area_bounds[region_number])
# area_bounds_idx


# Example usage function
def extract_area_boundaries(points: np.ndarray, step_x: int, step_y: int, 
                           labels: np.ndarray, target_labels: Union[int, List[int]],
                           plot: bool = True) -> Tuple[np.ndarray, dict]:
    """
    Main function to extract boundaries around specific areas.
    
    Args:
        points: Array of shape (N, 2) with (X, Y) coordinates
        step_x: Interval between adjacent points in X direction
        step_y: Interval between adjacent points in Y direction
        labels: Array where -1 = boundary, other values = area labels
        target_labels: Label(s) of area(s) to focus on
        plot: Whether to create visualization
        
    Returns:
        boundary_points: Extracted boundary coordinates
        area_boundaries: Boundaries for each target area
    """
    extractor = BoundaryExtractor(points, step_x, step_y, labels)
    boundary_points, area_boundaries = extractor.extract_boundaries(target_labels)
    
    if plot:
        extractor.plot_results(target_labels, boundary_points, area_boundaries)
    
    return boundary_points, area_boundaries