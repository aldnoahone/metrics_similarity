# A simulation of simple XRD
import numpy as np
import matplotlib.pyplot as plt


# Miller indices for hexagonal structure
hkl_list = [(0, 0, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 2), (2, 0, 0), (2, 0, 2)]
# Define a list of 20 distinct colors
colors_cover = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#f0e442", "#0072b2", "#56b4e9", "#cc79a7", "#d55e00",
    "#009e73", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3"
]

colors_cover_2 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",  # Standard Matplotlib colors
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#f0e442", "#0072b2", "#56b4e9", "#cc79a7", "#d55e00",  # ggplot2-inspired colors
    "#009e73", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",  # ColorBrewer colors
    "#ff1493", "#00ced1", "#ff4500", "#daa520", "#4682b4",  # Additional colors
    "#32cd32", "#8a2be2", "#ff6347", "#2e8b57", "#6a5acd",
    "#deb887", "#00bfff", "#adff2f", "#5f9ea0", "#ff69b4",
    "#ffa500", "#40e0d0", "#f08080", "#a52a2a", "#808000"
]   
# Calculate diffraction angles (2θ) using Bragg's law
def calculate_bragg_angles(hkl, a, c, wavelength):
    h, k, l = hkl
    # Reciprocal lattice spacings
    d_hkl_inv_sq = ((4 / 3) * ((h**2 + h*k + k**2) / a**2) + (l**2 / c**2))
    if d_hkl_inv_sq <= 0:
        return None  # Skip invalid reflections
    d_hkl = 1 / np.sqrt(d_hkl_inv_sq)
    # Bragg's law: nλ = 2d sinθ
    theta = np.arcsin(wavelength / (2 * d_hkl))
    if np.isnan(theta):  # Skip if no solution for θ
        return None
    return 2 * np.degrees(theta)

# Function to calculate interplanar spacing for a hexagonal unit cell
def calculate_d_hkl(hkl, a, c):
    h, k, l = hkl
    # Interplanar spacing formula for hexagonal systems
    d_hkl_inv_sq = (4 / 3) * ((h**2 + h * k + k**2) / a**2) + (l**2 / c**2)
    if d_hkl_inv_sq > 0:
        return 1 / np.sqrt(d_hkl_inv_sq)
    else:
        return None  # Invalid d-spacing
    
from scipy.special import wofz

# Function to calculate Voigt profile
def voigt_profile(x, center, fwhm_g, fwhm_l):
    """
    Generate a Voigt profile.
    :param x: Array of x-values
    :param center: Peak center
    :param fwhm_g: Full width at half maximum of Gaussian
    :param fwhm_l: Full width at half maximum of Lorentzian
    :return: Voigt profile values
    """
    sigma_g = fwhm_g / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation for Gaussian
    gamma_l = fwhm_l / 2  # Convert FWHM to half-width at half maximum for Lorentzian
    z = ((x - center) + 1j * gamma_l) / (sigma_g * np.sqrt(2))
    return np.real(wofz(z)) / (sigma_g * np.sqrt(2 * np.pi))


# Function to calculate lattice constant under strain
def calculate_lattice_a_with_strain(a, strain):
    return a * (1 + strain)

# Function to calculate shifted 2θ value for a given strain
def calculate_shifted_angle(strain, para):
    hkl, a, c, wavelength = para
    a_strained = calculate_lattice_a_with_strain(a, strain)
    return calculate_bragg_angles(hkl, a_strained, c, wavelength)

from sklearn.decomposition import PCA
import plotly.graph_objects as go
import pandas as pd

def pca_and_draw(raw_data, x_positions, strain_states, indices_t8):
    # Perform PCA on the profiles for all positions
    pca = PCA(n_components=2)
    embedded_2d_all = pca.fit_transform(raw_data)

    # Create a DataFrame for all data points
    pca_df_2d_all = pd.DataFrame({
        'PCA_1': embedded_2d_all[:, 0],
        'PCA_2': embedded_2d_all[:, 1],
        'Position': x_positions,
        'Strain': strain_states,
        'Category': 'All Points'
    })

    # Create scatter plot for all points
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=pca_df_2d_all['PCA_1'],
        y=pca_df_2d_all['PCA_2'],
        mode='markers',
        marker=dict(
            size=5,
            color=pca_df_2d_all['Strain'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Strain State')
        ),
        customdata=pca_df_2d_all['Position'],  # Pass df['i'] to customdata
        hovertemplate="X: %{x}<br>Y: %{y}<br>strain: %{marker.color:.6f}<br>Position: %{customdata}<extra></extra>",
        name='All Points'
    ))

    # Create scatter plot for selected points
    x_selected = pca_df_2d_all['PCA_1'][indices_t8].to_numpy()
    y_selected = pca_df_2d_all['PCA_2'][indices_t8].to_numpy()
    fig_selected = go.Figure()
    for i in range(len(indices_t8)):
        fig_selected.add_trace(go.Scatter(
            x=[x_selected[i]], 
            y=[y_selected[i]], 
            mode='markers',
            marker=dict(size=10, color=colors_cover[i]),
            name=f"Point {i + 1}"
        ))


    # Combine both plots into a single figure
    fig_combined = go.Figure()
    fig_combined.add_traces(fig_all.data + fig_selected.data)

    # Update layout for the combined figure
    fig_combined.update_layout(
        title="PCA Visualization of XRD Profiles (All and Selected Points)",
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        legend=dict(x=1.3, y=1),
        showlegend=True
    )


    # Show combined plot
    return fig_combined, pca_df_2d_all

def pca_and_draw_2D(raw_data, coor_arr, strain_states, indices_t8):
    # Perform PCA on the profiles for all positions
    pca = PCA(n_components=2)
    embedded_2d_all = pca.fit_transform(raw_data)

    # Create a DataFrame for all data points
    pca_df_2d_all = pd.DataFrame({
        'PCA_1': embedded_2d_all[:, 0],
        'PCA_2': embedded_2d_all[:, 1],
        'Strain': strain_states,
        'Category': 'All Points'
    })

    # Create scatter plot for all points
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=pca_df_2d_all['PCA_1'],
        y=pca_df_2d_all['PCA_2'],
        mode='markers',
        marker=dict(
            size=5,
            color=pca_df_2d_all['Strain'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Strain State')
        ),
        customdata=coor_arr,  # Pass df['i'] to customdata
        hovertemplate="X: %{x}<br>Y: %{y}<br>strain: %{marker.color:.6f}<br>Position: %{customdata}<extra></extra>",
        name='All Points'
    ))

    # color_selected = np.array(range(len(indices_t8)))
    # Create scatter plot for selected points
    x_selected = pca_df_2d_all['PCA_1'][indices_t8].to_numpy()
    y_selected = pca_df_2d_all['PCA_2'][indices_t8].to_numpy()
    fig_selected = go.Figure()
    for i in range(len(indices_t8)):
        fig_selected.add_trace(go.Scatter(
            x=[x_selected[i]], 
            y=[y_selected[i]], 
            mode='markers',
            marker=dict(size=10, color=colors_cover_2[i]),
            name=f"Point {i + 1}"
        ))


    # Combine both plots into a single figure
    fig_combined = go.Figure()
    fig_combined.add_traces(fig_all.data + fig_selected.data)

    # Update layout for the combined figure
    fig_combined.update_layout(
        title="PCA Visualization of XRD Profiles (All and Selected Points)",
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        legend=dict(x=1.3, y=1),
        showlegend=True
    )

    # Show combined plot
    return fig_combined, pca_df_2d_all

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_and_draw_2D_mat(raw_data, coor_arr, strain_states, indices_t8, colors_cover_2):
    # Perform PCA on the profiles for all positions
    pca = PCA(n_components=2)
    embedded_2d_all = pca.fit_transform(raw_data)

    # Create a DataFrame for all data points
    pca_df_2d_all = pd.DataFrame({
        'PCA_1': embedded_2d_all[:, 0],
        'PCA_2': embedded_2d_all[:, 1],
        'Strain': strain_states
    })

    # Create scatter plot for all points
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        pca_df_2d_all['PCA_1'], pca_df_2d_all['PCA_2'],
        c=pca_df_2d_all['Strain'], cmap='viridis', s=5, alpha=0.7
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Strain State")

    # Highlight selected points
    selected_x = pca_df_2d_all['PCA_1'].iloc[indices_t8].to_numpy()
    selected_y = pca_df_2d_all['PCA_2'].iloc[indices_t8].to_numpy()
    
    for i, (x, y, color) in enumerate(zip(selected_x, selected_y, colors_cover_2)):
        ax.scatter(x, y, color=color, s=50, edgecolors='black', label=f"Point {i+1}")

    # Format plot
    ax.set_title("PCA Visualization of XRD Profiles")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(loc="upper right", fontsize=10, markerscale=1.5)

    # Show plot
    plt.show()

    return fig, pca_df_2d_all


import umap
def umap_and_draw(umap_coeff, raw_data, x_positions, strain_states, indices_t8):
    n_neighbors, n_components, n_epochs, min_dist = umap_coeff
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, n_epochs=n_epochs, min_dist = min_dist)
    embedded_2d_all = reducer.fit_transform(raw_data)

    # Create a DataFrame for all data points
    umap_df_2d_all = pd.DataFrame({
        'UMAP_1': embedded_2d_all[:, 0],
        'UMAP_2': embedded_2d_all[:, 1],
        'Position': x_positions,
        'Strain': strain_states,
        'Category': 'All Points'
    })
    # Create scatter plot for all points
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=umap_df_2d_all['UMAP_1'],
        y=umap_df_2d_all['UMAP_2'],
        mode='markers',
        marker=dict(
            size=5,
            color=umap_df_2d_all['Strain'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Strain State')
        ),
        customdata=umap_df_2d_all['Position'],  # Pass df['i'] to customdata
        hovertemplate="X: %{x}<br>Y: %{y}<br>strain: %{marker.color:.6f}<br>Position: %{customdata}<extra></extra>",

        name='All Points'
    ))

    # Create scatter plot for selected points
    fig_selected = go.Figure()
    x_selected = umap_df_2d_all['UMAP_1'][indices_t8].to_numpy()
    y_selected = umap_df_2d_all['UMAP_2'][indices_t8].to_numpy()
    fig_selected = go.Figure()
    for i in range(len(indices_t8)):
        fig_selected.add_trace(go.Scatter(
            x=[x_selected[i]], 
            y=[y_selected[i]], 
            mode='markers',
            marker=dict(size=10, color=colors_cover[i]),
            name=f"Point {i + 1}"
        ))

    # Combine both plots into a single figure
    fig_combined = go.Figure()
    fig_combined.add_traces(fig_all.data + fig_selected.data)

    # Update layout for the combined figure
    fig_combined.update_layout(
        title="UMAP Visualization of XRD Profiles (All and Selected Points)",
        xaxis_title="UMAP Component 1",
        yaxis_title="UMAP Component 2",
        legend=dict(x=1.3, y=1),
        showlegend=True
    )

    # Show combined plot
    return fig_combined, umap_df_2d_all

def umap_and_draw_2D(umap_coeff, raw_data, coor_arr, strain_states, indices_t8):
    n_neighbors, n_components, n_epochs, min_dist = umap_coeff
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, n_epochs=n_epochs, min_dist = min_dist)
    embedded_2d_all = reducer.fit_transform(raw_data)

    # Create a DataFrame for all data points
    umap_df_2d_all = pd.DataFrame({
        'UMAP_1': embedded_2d_all[:, 0],
        'UMAP_2': embedded_2d_all[:, 1],
        'Strain': strain_states,
        'Category': 'All Points'
    })
    # Create scatter plot for all points
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=umap_df_2d_all['UMAP_1'],
        y=umap_df_2d_all['UMAP_2'],
        mode='markers',
        marker=dict(
            size=5,
            color=umap_df_2d_all['Strain'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Strain State')
        ),
        customdata=coor_arr,  # pass extra info
        hovertemplate="X: %{x}<br>Y: %{y}<br>strain: %{marker.color:.6f}<br>Position: %{customdata}<extra></extra>",

        name='All Points'
    ))

    
    # Create scatter plot for selected points
    fig_selected = go.Figure()
    x_selected = umap_df_2d_all['UMAP_1'][indices_t8].to_numpy()
    y_selected = umap_df_2d_all['UMAP_2'][indices_t8].to_numpy()
    fig_selected = go.Figure()
    for i in range(len(indices_t8)):
        fig_selected.add_trace(go.Scatter(
            x=[x_selected[i]], 
            y=[y_selected[i]], 
            mode='markers',
            marker=dict(size=10, color=colors_cover_2[i]),
            name=f"Point {i + 1}"
        ))

    # Combine both plots into a single figure
    fig_combined = go.Figure()
    fig_combined.add_traces(fig_all.data + fig_selected.data)

    # Update layout for the combined figure
    fig_combined.update_layout(
        title="UMAP Visualization of XRD Profiles (All and Selected Points)",
        xaxis_title="UMAP Component 1",
        yaxis_title="UMAP Component 2",
        legend=dict(x=1.3, y=1),
        showlegend=True
    )

    # Show combined plot
    return fig_combined, umap_df_2d_all

# # Combine 'i' and 'j' into customdata as a 2D array
# df['customdata'] = df[['i', 'j']].values.tolist()

# # Create scatter plot
# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=df['x'],
#     y=df['y'],
#     mode='markers',
#     marker=dict(size=10, color=df['z'], colorscale='Viridis'),
#     customdata=df['customdata'],  # Pass both 'i' and 'j'
#     hovertemplate=(
#         "X: %{x}<br>"
#         "Y: %{y}<br>"
#         "Z: %{marker.color:.2f}<br>"
#         "Info 1: %{customdata[0]}<br>"
#         "Info 2: %{customdata[1]}<extra></extra>"
#     )
# ))

# in the calculation of spatial neighbors, 
# the default radius is set as 10

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import KDTree

# built the KDtree in the spatial array

def neighbors_search(data, radius = 10):
    """
    INPUT:
    
    data the array of the spatial positions with a shape of Nx2

    OUTPUT:
    a list that stores the index of neighbors around each point

    """
    kdtree = KDTree(data)
    N = len(data)
    results = []

    for i in range(N):
        # Find all neighbors within the radius (including the center point itself)
        neighbors_idx = kdtree.query_ball_point(data[i], radius)
        results.append(neighbors_idx)
    
    return results

# Function to calculate average cosine similarity with neighbors within a given radius
def avg_cosine_similarity_within_radius(data, neighbors_index):
    """  
    INPUT:
    data: the array that stores the raw measurement results or that is reduced into a low-dimensional space,
          with a shape of NxD, of which D is the size of dimensionality of the data 
    neighbors_index: a list that stores the index around each point within the specific radius

    OUTPUT:
    An array that stores the averaged similarity from the spatial neighbors
    
    """

    N = len(data)

    avg_similarities = np.zeros(N)
    
    # Step 2: Compute the cosine similarity matrix (NxN) for the data
    cos_sim_matrix = cosine_similarity(data)

    # Step 3: For each point, find neighbors within the specified radius
    for i in range(N):
        # Find all neighbors within the radius (including the center point itself)
        neighbors_idx = np.array(neighbors_index[i])

        # If there are neighbors (excluding the point itself), calculate average cosine similarity
        if len(neighbors_idx) > 1:
            similarities = [cos_sim_matrix[i, j] for j in neighbors_idx if i != j]  # Exclude self
            avg_similarities[i] = np.mean(similarities) if similarities else 0
        else:
            avg_similarities[i] = 0  # No neighbors, so similarity is 0
    
    return avg_similarities

# Function to calculate average Euclidean distance with neighbors within a given radius
def avg_distance_within_radius(data, neighbors_index):
    """  
    INPUT:
    data: the array that stores the raw measurement results or that is reduced into a low-dimensional space,
          with a shape of NxD, of which D is the size of dimensionality of the data  
    neighbors_index: a list that stores the index around each point within the specific radius

    OUTPUT:
    An array that stores the averaged elucidean distances
    
    """

    N = len(data)
    avg_distances = np.zeros(N)

    # Step 4: For each point, find neighbors within the specified radius
    for i in range(N):
        # Find all neighbors within the radius (including the center point itself)
        neighbors_idx =  neighbors_index[i]
        
        # If there are neighbors (excluding the point itself), calculate average distance
        if len(neighbors_idx) > 1:
            distances = [np.linalg.norm(np.abs(data[i] - data[j])) for j in neighbors_idx if i != j]
            avg_distances[i] = np.mean(distances) if distances else 0
        else:
            avg_distances[i] = 0  # No neighbors, so distance is 0
    
    return avg_distances

# Define the function
def linear_calculate_f(X, Y, x_s, y_s, x_e, y_e, parameter_y):
    """
    Calculates f(x, y) = Ax*x^2 + bx*x + Ay*y^2 + by*y for given X, Y meshgrid.

    Parameters:
    X (ndarray): X-coordinates from the meshgrid.
    Y (ndarray): Y-coordinates from the meshgrid.
    x_s (float): the start of response at x direction
    x_e (float): the end of response at x direction
    y_s (float): the start of response at y direction
    y_e (float): the end of response at y direction

    Returns:
    ndarray: Calculated values of f(X, Y) for the meshgrid.
    """

    Ax = (x_e - x_s) / (np.max(X) - np.min(X))
    Ay = (y_e - y_s) / (np.max(Y) - np.min(Y))

    return Ax * X + x_s + Ay * Y + y_s + Y * parameter_y



def find_indices_in_range(data, x_range, y_range=None):
    """
    Finds the indices of elements in a 1D or 2D array that fall within the specified range.

    Parameters:
    - data (ndarray): Input array.
        If 1D, it is treated as X.
        If 2D, it is treated as (X, Y) with shape (Number of samples, 2).
    - x_range (tuple): Range for X (min, max).
    - y_range (tuple, optional): Range for Y (min, max) if data is 2D.

    Returns:
    - indices (ndarray): Indices of elements that fall within the specified range.
    """
    data = np.asarray(data)

    if data.ndim == 1:
        # 1D case: only check X range
        indices = np.where((data >= x_range[0]) & (data <= x_range[1]))[0]
    elif data.ndim == 2 and y_range is not None:
        # 2D case: check both X and Y ranges
        indices = np.where(
            (data[:, 0] >= x_range[0]) & (data[:, 0] <= x_range[1]) &  # X range
            (data[:, 1] >= y_range[0]) & (data[:, 1] <= y_range[1])    # Y range
        )[0]
    else:
        raise ValueError("For 2D input, y_range must be provided.")

    return indices

def replace_function_values(data, indices
                            , x_values, f_1_function
                            , g_function, h_function
                            , coor
                            , peak_magnitudes
                            , fwhms
                            , para
                            ):
    """
    Replace values in a function array data with new funtion for specified indices.

    raw function h(g(f(X_positions)), x_values, *parm = [peak, fwhm_g, fwhm_l])

    In this research, the function is:
    strain = f(X_positions)
    peak_angle = g(strain)
    XRD_profiles = peak_magnitudes[i, j] * voigt_profile(x_values, peak_angle, fwhm, fwhm)/ \
                            np.max(voigt_profile(x_values, angle, fwhm, fwhm))

    for now, I hope to replace the strain as
    strain  = f_1(X_positions)

    Parameters:
    - data (ndarray): The original function array with shape (Number of samples, x_values).
    - x_values (ndarray): The x-values array used in the function.
    - f_1_function (function): A function that computes the strain.
    - g_function (function): A function that computes the peak.
    - h_function (function): A function that computes the xrd_profiles.
    - indices (ndarray): Indices of elements to replace in the function array.
    - coor (ndarray): Coordinates, like X-posiitons and Y-positions.
    - peak_magnitudes (ndarray): The array for peak magnitudes.
    - fwhms (ndarray): The array of fwhm for calculating the width. In this research
    , we suppose there is only one fwhm
    - para (list): The list of parameters for calculating the peak_angle 

    Returns:
    - updated_data (ndarray): array with replaced values.
    """
    data = np.copy(data)  # Avoid modifying the original data

    # Loop over indices and replace values
    for idx in indices:
        _positions = coor[idx]
        strain = f_1_function(x = _positions)
        peak_angle = g_function(strain = strain, para = para)
        peak_magnitude = peak_magnitudes[idx]
        fwhm = fwhms[idx]

        new_XRD_profiles = peak_magnitude * voigt_profile(x_values, peak_angle, fwhm, fwhm)/ \
                            np.max(voigt_profile(x_values, peak_angle, fwhm, fwhm))

        data[idx] = new_XRD_profiles  # Replace the row in the data array

    return data


## similarity calculation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
def _cosine(data, N = 10):
    correlation_matrix = cosine_similarity(data)
    # find the highset values
    top_indices = np.argsort(correlation_matrix, axis=1)[:, -N:]
    return top_indices

def _pearson(data, N = 10):
    correlation_matrix = np.corrcoef(data)
    # find the highset values
    top_indices = np.argsort(correlation_matrix, axis=1)[:, -N:]
    return top_indices

def _eudis(data, N = 10):
    correlation_matrix = pairwise_distances(data, metric='euclidean')
    # find the smallest values
    top_indices = np.argsort(correlation_matrix, axis=1)[:, 0:N]
    return top_indices

from scipy.stats import ks_2samp, ttest_ind
def _kstest_partial(data, filtered_indices, N = 10, sort_flag = "ks"):
    NoX = len(data)
    top_indices = np.zeros((NoX, N), dtype=int)
    top_scores = np.zeros((NoX, N))
    top_pvalues = np.zeros((NoX, N))

    for i in range(NoX):
        results_ks = []
        results_p = []
        for j in filtered_indices[i]:
            if i != j:
                ks_stat, p_value = ks_2samp(data[i], data[j])
                results_ks.append(ks_stat)
                results_p.append(p_value)
            elif i == j:
                results_ks.append(0)
                results_p.append(1)

        # find the smallest values
        sorted_indices_ks = np.argsort(results_ks)[:N]
        sorted_indices_p = np.argsort(results_p)[-N:]
        if sort_flag == 'ks':
            sorted_indices = sorted_indices_ks
        elif sort_flag == 'p':
            sorted_indices = sorted_indices_p
        top_indices[i] = np.array(filtered_indices[i])[sorted_indices]
        top_scores[i] = np.array(results_ks)[sorted_indices]
        top_pvalues[i] = np.array(results_p)[sorted_indices]

    return top_scores, top_pvalues, top_indices

def _ttest_partial(data, filtered_indices, N = 10, sort_flag = "ks"):
    NoX = len(data)
    top_indices = np.zeros((NoX, N), dtype=int)
    top_scores = np.zeros((NoX, N))
    top_pvalues = np.zeros((NoX, N))

    for i in range(NoX):
        results_t = []
        results_p = []
        for j in filtered_indices[i]:
            if i != j:
                t_stat, p_value = ttest_ind(data[i], data[j])
                results_t.append(t_stat)
                results_p.append(p_value)
            elif i == j:
                results_t.append(0)
                results_p.append(1)

        # find the smallest values
        sorted_indices_ks = np.argsort(results_t)[:N]
        sorted_indices_p = np.argsort(results_p)[-N:]
        if sort_flag == 'ks':
            sorted_indices = sorted_indices_ks
        elif sort_flag == 'p':
            sorted_indices = sorted_indices_p
        top_indices[i] = np.array(filtered_indices[i])[sorted_indices]
        top_scores[i] = np.array(results_t)[sorted_indices]
        top_pvalues[i] = np.array(results_p)[sorted_indices]

    return top_scores, top_pvalues, top_indices

from sklearn.neighbors import NearestNeighbors
def _adj_search(data, _dist):

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

def _adj_search_with_umap(data, _dist, umap_data, n_neighbors = 24):

    _adj_matrix_umap = np.zeros([len(data), len(data)], dtype = np.int8)    

    for i in range(len(data)):
        for j in range(len(data)):

            # skip the case of raw point
            if i == j:
                continue
            
            x_i, y_i = data[i]
            x_j, y_j = data[j]
            if abs(x_i - x_j) <= _dist and abs(y_i - y_j) <= _dist:
                _adj_matrix_umap[i, j] = 1
 

    # Step 2: Create KNN graph based on UMAP-reduced data
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(umap_data)
    knn_graph = knn.kneighbors_graph(umap_data).toarray()
    
    # Step 3: Filter adjacency matrix using the KNN graph
    for i in range(len(data)):
        for j in range(len(data)):
            if _adj_matrix_umap[i, j] == 1 and knn_graph[i, j] == 0:
                _adj_matrix_umap[i, j] = 0

    return _adj_matrix_umap

from plotly.subplots import make_subplots
def neighbors_based_visualization(x_positions, average_similarities_co, average_similarities_ed, selcted_index):
    map_data = {
        'x':  x_positions,
        'y1': average_similarities_co,
        'y2': average_similarities_ed
    }

    df = pd.DataFrame(map_data)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,  # Two rows, one column
        shared_xaxes=True,  # Share the X-axis
        vertical_spacing=0.15,  # Space between plots
        subplot_titles=("Average similarities by cosine similarity"
                        , "Average similarities by euclidean distance")
    )

    # Add the first line plot (y1) to the first subplot
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y1'],
        mode='lines',
        name='Line 1 (Y1)',
        line=dict(color='blue')
    ), row=1, col=1)

    # Add scatter points for y1
    for idx, color in zip(selcted_index, colors_cover):
        fig.add_trace(go.Scatter(
            x=[df['x'][idx]],
            y=[df['y1'][idx]],
            mode='markers',
            marker=dict(color=color, size=10),
            name=f'Scatter on Y1 at {df["x"][idx]}'
        ), row=1, col=1)

    # Add the second line plot (y2) to the second subplot
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y2'],
        mode='lines',
        name='Line 2 (Y2)',
        line=dict(color='green')
    ), row=2, col=1)

    # Add scatter points for y2
    for idx, color in zip(selcted_index, colors_cover):
        fig.add_trace(go.Scatter(
            x=[df['x'][idx]],
            y=[df['y2'][idx]],
            mode='markers',
            marker=dict(color=color, size=10),
            name=f'Scatter on Y2 at {df["x"][idx]}'
        ), row=2, col=1)

    # Update layout
    fig.update_layout(
        title="Two Subplots for Y1 and Y2 with Highlighted Scatter Points",
        xaxis_title="X-axis",
        yaxis=dict(title="cosine similarity"),
        yaxis2=dict(title="euclidean distance"),
        height=600,  # Adjust figure height
        showlegend=True  # Show legend
    )
    return fig


def generate_neighboring_points(x, y, dx, dy, x_min=None, x_max=None, y_min=None, y_max=None):
    """
    Generate 9 points in a 3x3 grid centered at (x, y) with boundary constraints
    
    Parameters:
    -----------
    x, y : float
        Center coordinates
    dx, dy : float
        Step sizes in x and y directions
    x_min, x_max : float, optional
        X-coordinate boundaries. If None, no constraint applied
    y_min, y_max : float, optional  
        Y-coordinate boundaries. If None, no constraint applied
    
    Returns:
    --------
    points : np.ndarray
        Array of shape (N, 2) where N <= 9, containing valid [x, y] coordinates
        Only includes points within the specified boundaries
    valid_directions : list
        List of direction numbers (1-9) corresponding to valid points
        
    Note: Center point (direction 5) is always included if within bounds
    """
    
    # Define relative positions in 3x3 grid
    relative_positions = [
        (-dx,  dy),  # 1: top-left
        ( 0,   dy),  # 2: top-center  
        ( dx,  dy),  # 3: top-right
        (-dx,   0),  # 4: middle-left
        ( 0,    0),  # 5: center
        ( dx,   0),  # 6: middle-right
        (-dx, -dy),  # 7: bottom-left
        ( 0,  -dy),  # 8: bottom-center
        ( dx, -dy),  # 9: bottom-right
    ]
    
    # Generate absolute positions and check boundaries
    valid_points = []
    valid_directions = []
    
    for direction, (rel_x, rel_y) in enumerate(relative_positions, 1):
        point_x = x + rel_x
        point_y = y + rel_y
        
        # Check if point is within boundaries
        is_valid = True
        
        if x_min is not None and point_x < x_min:
            is_valid = False
        if x_max is not None and point_x > x_max:
            is_valid = False
        if y_min is not None and point_y < y_min:
            is_valid = False
        if y_max is not None and point_y > y_max:
            is_valid = False
        
        if is_valid:
            valid_points.append([point_x, point_y])
            valid_directions.append(direction)
    
    return np.array(valid_points), valid_directions

def element_search(raw_array, target_array):
    x_t, y_t = target_array
    for i, element in enumerate(raw_array):
        x, y = element
        if x == x_t and y == y_t:
            return i
        
    return None