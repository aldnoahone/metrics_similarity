import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

# Set STIX fonts globally
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['mathtext.fontset'] = 'stix'

# control the fontsize!
mpl.rcParams['font.size'] = 14          # Default font size
mpl.rcParams['axes.titlesize'] = 16     # Title font size
mpl.rcParams['axes.labelsize'] = 14     # X and Y label font size
mpl.rcParams['xtick.labelsize'] = 12    # X tick label font size
mpl.rcParams['ytick.labelsize'] = 12    # Y tick label font size
mpl.rcParams['legend.fontsize'] = 12    # Legend font size
mpl.rcParams['figure.titlesize'] = 18   # Figure title font size

def sample_plot(ax, S, data, label=None, colorbar_title=None, amp = None, set_ticks_flag = False):
    X_cor = S[:, 0]
    Y_cor = S[:, 1]
    grid_x, grid_y = np.mgrid[X_cor.min():X_cor.max():100j, Y_cor.min():Y_cor.max():100j]
    grid_z = griddata((X_cor, Y_cor), data, (grid_x, grid_y), method='nearest')
    img = ax.imshow(grid_z.T, extent=(X_cor.min(), X_cor.max(), Y_cor.min(), Y_cor.max()), 
                    origin='lower', cmap='rainbow', aspect="auto")

    if not set_ticks_flag:
        cbar = plt.colorbar(img, ax=ax, orientation="horizontal"
                            , fraction=0.019, pad=0.08)
    else:
        cbar = plt.colorbar(img, ax=ax, orientation="horizontal"
                            , fraction=0.019, pad=0.12)    
    # set the ticks
    data_array = np.array(data)
    vmin, vmax = data_array.min(), data_array.max()
    vmid = (vmin + vmax) / 2
    cbar.set_ticks([vmin, vmid, vmax])

    if colorbar_title:
        cbar.set_label(colorbar_title, fontsize=14
                       # , fontweight='bold'
                       , rotation=0, labelpad=5)

    if not set_ticks_flag:
        ax.set_xticks([])
        ax.set_yticks([])

    # Let matplotlib auto-determine major tick positions first
    # Then use those to create automatic minor ticks
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())  # Auto minor ticks
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())  # Auto minor ticks

    # Hide major ticks but keep minor ticks
    ax.tick_params(which='major', length=8, width=1, labelsize=12)  # Hide major ticks
    ax.tick_params(which='minor', length=4, width=1, labelsize=12)  # Show minor ticks

    ax.set_xlabel(r'X [$\mu$m]')
    ax.set_ylabel(r'Y [$\mu$m]')

    if amp or amp == 0:
        if not amp == 'ini':
            ax.set_title(f"amp = {amp}") 
        else:
            ax.set_title(fr"amp = $\infty$") 
    if label is not None:
        ax.text(-0.2, 1.05, f'({label})', transform=ax.transAxes, fontsize=26, fontweight='bold', color='black',
                    bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    ax.set_aspect('equal') 
    plt.tight_layout()


from matplotlib.lines import Line2D 
def link_plot(S, link_matrix, ax, label = None, legend_flag = True,
              edge_label_notes = None, edge_color_map = None):
    ax.scatter(S[:, 0], S[:, 1], c = 'blue', s = 1)

    # visualize the edges
    legend_handles = {} 
    N = edge_label_notes.shape[0]
    for i in range(N):
        for j in range(N):
            if link_matrix[i, j] == 1:  # If there's a link from node i to node j
                x_i, y_i = S[i]  # Coordinates of node i
                x_j, y_j = S[j]  # Coordinates of node j
                
                edge_label = edge_label_notes[i, j]  # Get the edge label
                edge_color =  edge_color_map.get(edge_label, "black")  # Default to gray if label is unknown

                ax.plot([x_i, x_j], [y_i, y_j], color=edge_color, alpha=0.7, linewidth=1.5)  # Draw edge
                # Add to legend dictionary (avoid duplicates)
                if edge_label not in legend_handles:
                    legend_handles[edge_label] = Line2D([0], [0], color=edge_color, linewidth=2, label=f"Direction {edge_label}")

    # Step 6: Add Legend with Lines
    sorted_labels = sorted(legend_handles.keys()) 
    sorted_legend = [legend_handles[label] for label in sorted_labels]  # Sort handles accordingly

    # Step 7: Add Legend with Sorted Lines
    if legend_flag == True:
        ax.legend(handles=sorted_legend, 
                bbox_to_anchor=(1.02, 1.0),  # x=1.05 (outside), y=1.0 (top)
                loc='upper left', 
                title="Directions",
                borderaxespad=0,
                frameon=True,                # Optional: add frame
                fancybox=True)              # Optional: rounded corners

    ax.set_xlabel(f"X [$\mu$m]")
    ax.set_ylabel(f"Y [$\mu$m]")

    # ax.set_xticks([])
    # ax.set_yticks([])

    ax.set_aspect('equal', adjustable='box')
    # ax.set_aspect('auto') 

    if label is not None:
        ax.text(
        0.02, 1.02# 1.02
        , f'({label})', transform=ax.transAxes, fontsize=26, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0, edgecolor='none')
        )

    plt.tight_layout()

def link_plot_2(S, link_matrix, ax, label = None):
    ax.scatter(S[:, 0], S[:, 1], c = 'blue', s = 1)

    # visualize the edges
    #legend_handles = {} 
    N = link_matrix.shape[0]
    for i in range(N):
        for j in range(N):
            if link_matrix[i, j] == 1:  # If there's a link from node i to node j
                x_i, y_i = S[i]  # Coordinates of node i
                x_j, y_j = S[j]  # Coordinates of node j
                
                ax.plot([x_i, x_j], [y_i, y_j], color='black', alpha=0.7, linewidth=0.75)  # Draw edge
               
    ax.set_xlabel(f"X [$\mu$m]")
    ax.set_ylabel(f"Y [$\mu$m]")

    # ax.set_xticks([])
    # ax.set_yticks([])

    ax.set_aspect('equal', adjustable='box')

    if label is not None:
        ax.text(
        0.02, 1.02# 1.02
        , f'({label})', transform=ax.transAxes, fontsize=26, fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0, edgecolor='none')
        )

    plt.tight_layout()


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

COLORS_COVER_2 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",  # Standard Matplotlib colors
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#f0e442", "#0072b2", "#56b4e9", "#cc79a7", "#d55e00",  # ggplot2-inspired colors
    "#009e73", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",  # ColorBrewer colors
    "#ff1493", "#00ced1", "#ff4500", "#daa520", "#4682b4",  # Additional colors
    "#32cd32", "#8a2be2", "#ff6347", "#2e8b57", "#6a5acd",
    "#deb887", "#00bfff", "#adff2f", "#5f9ea0", "#ff69b4",
    "#ffa500", "#40e0d0", "#f08080", "#a52a2a", "#808000"
]  

def create_label_color_mapping(max_labels=50):
    """
    Create a fixed label-to-color mapping that can be reused across functions
    
    Parameters:
    -----------
    max_labels : int
        Maximum number of labels to create mapping for
    
    Returns:
    --------
    dict : mapping from label (int) to color (hex string)
    """
    
    label_to_color = {}
    
    # Special cases (common in segmentation)
    label_to_color[-1] = COLORS_COVER_2[7]  # boundaries -> gray
    label_to_color[0] = COLORS_COVER_2[0]   # background -> blue
    
    # Regular regions: map sequentially starting from index 1
    for i in range(1, max_labels):
        if i >= 7:
            color_index = i % len(COLORS_COVER_2) + 1
            label_to_color[i] = COLORS_COVER_2[color_index]
        elif i < 7:
            color_index = i % len(COLORS_COVER_2)
            label_to_color[i] = COLORS_COVER_2[color_index]
    
    return label_to_color


# def create_label_heatmap(ax=None, S = None, markers = None,colors=COLORS_COVER_2, figsize=(10, 8)
#                          , label = None, shape = (61, 25)):
#     """
#     Simple matplotlib heatmap with discrete label-based colors
    
#     Parameters:
#     -----------
#     S : array-like, shape (n, 2)
#         Coordinates for the heatmap
#     markers : array-like
#         Label array (can be 1D or 2D)
#     ax : matplotlib axis object, optional
#         Axis to plot on. If None, creates new figure and axis
#     colors : list of hex colors or colormap name
#         Custom color list or matplotlib colormap name (default: uses custom palette)
#     figsize : tuple
#         Figure size (width, height), only used if ax is None
    
#     Returns:
#     --------
#     im : matplotlib image object (for colorbar creation)
#     """
#     if S is not None:
#         x_coords = S[:, 0].reshape(shape)
#         y_coords = S[:, 1].reshape(shape)
#         extent = [x_coords.min(), x_coords.max(),
#                     y_coords.min(), y_coords.max()]

    
#     # Flatten markers if needed
#     markers_flat = markers.reshape(-1) if markers.ndim > 1 else markers
    
#     # Get unique labels
#     unique_labels = np.unique(markers_flat)
#     n_labels = len(unique_labels)
    
#     # Create discrete colormap
#     if colors is None:
#         print("Please input a valid colors")
    
#     if isinstance(colors, list):
#         # Use custom color list
#         selected_colors = colors[:n_labels]  # Take only needed colors
#         if n_labels > len(colors):
#             print(f"Warning: Need {n_labels} colors but only {len(colors)} provided. Cycling through colors.")
#             selected_colors = [colors[i % len(colors)] for i in range(n_labels)]
#         discrete_cmap = mcolors.ListedColormap(selected_colors)
#     else:
#         # Use matplotlib colormap
#         base_cmap = plt.cm.get_cmap(colors)
#         color_values = base_cmap(np.linspace(0, 1, n_labels))
#         discrete_cmap = mcolors.ListedColormap(color_values)
    
#     # Create normalization for discrete values
#     bounds = np.arange(len(unique_labels) + 1) - 0.5
#     norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)
    
#     # Map original labels to sequential indices
#     label_to_index = {label: i for i, label in enumerate(unique_labels)}
#     markers_indexed = np.array([label_to_index[label] for label in markers_flat])
    
#     # Reshape back to original shape if needed
#     if markers.ndim > 1:
#         markers_indexed = markers_indexed.reshape(markers.shape)
    
#     # Create axis if not provided
#     if ax is None:
#         fig, ax = plt.subplots(figsize=figsize)
    
#     # Create heatmap
#     im = ax.imshow(markers_indexed, cmap=discrete_cmap, norm=norm, 
#                    aspect='equal', origin='upper', extent=extent)
    
#     # Create colorbar with discrete blocks
#     cbar = plt.colorbar(im, ax=ax, boundaries=bounds, ticks=range(n_labels))
    
#     # Set colorbar labels to original label values
#     cbar.set_ticklabels([str(label) for label in unique_labels])
#     cbar.set_label('Labels', rotation=270, labelpad=20)
    
#     # Set titles and labels
#     # ax.set_title('Watershed Segmentation Results')
#     # ax.set_xlabel('X Coordinate')
#     # ax.set_ylabel('Y Coordinate')
#     ax.set_xticks([])
#     ax.set_yticks([])

#     # add a label to the figure
#     if label is not None:
#         ax.text(-0.02, 1.02, f'({label})', transform=ax.transAxes, fontsize=26, fontweight='bold', color='black',
#                     bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    
#     return im

def get_color_for_label(label, label_to_color):
    """
    Get color for a specific label, with fallback
    
    Parameters:
    -----------
    label : int
        Label value
    label_to_color : dict
        Label-to-color mapping
    
    Returns:
    --------
    str : hex color code
    """
    if label in label_to_color:
        return label_to_color[label]
    else:
        # Fallback: use label as index in color palette
        return COLORS_COVER_2[label % len(COLORS_COVER_2)]

def create_label_heatmap(ax=None, S=None, markers=None, colors=None, figsize=(10, 8), 
                         label=None, shape=(61, 25), label_to_color=None
                         , set_ticks_flag = False):
    """
    Simple matplotlib heatmap with discrete label-based colors
    
    Parameters:
    -----------
    ax : matplotlib axis object, optional
        Axis to plot on. If None, creates new figure and axis
    S : array-like, shape (n, 2), optional
        Coordinates for the heatmap (used to set extent)
    markers : array-like
        Label array (can be 1D or 2D)
    colors : list of hex colors or colormap name, optional
        Custom color list or matplotlib colormap name (default: COLORS_COVER_2)
    figsize : tuple
        Figure size (width, height), only used if ax is None
    label : str, optional
        Label to add to the plot (like 'a', 'b', 'c')
    shape : tuple
        Shape to reshape coordinates if S is provided
    label_to_color : dict, optional
        Pre-defined mapping from labels to hex colors (overrides colors parameter)
    
    Returns:
    --------
    im : matplotlib image object (for colorbar creation)
    """
    
    # Set default colors if none provided
    if colors is None:
        colors = COLORS_COVER_2
    
    # Calculate extent if S is provided
    extent = None
    if S is not None:
        x_coords = S[:, 0].reshape(shape)
        y_coords = S[:, 1].reshape(shape)
        extent = [x_coords.min(), x_coords.max(),
                  y_coords.min(), y_coords.max()]

    # Flatten markers if needed
    markers_flat = markers.reshape(-1) if markers.ndim > 1 else markers
    
    # Get unique labels
    unique_labels = np.unique(markers_flat)
    n_labels = len(unique_labels)
    
    # Create color mapping
    if label_to_color is not None:
        # Use pre-defined label-to-color mapping (GLOBAL_LABEL_COLORS)
        selected_colors = []
        for label_val in unique_labels:
            selected_colors.append(get_color_for_label(label_val, label_to_color))
        discrete_cmap = mcolors.ListedColormap(selected_colors)
        
    elif isinstance(colors, list):
        # Use custom color list
        selected_colors = colors[:n_labels]  # Take only needed colors
        if n_labels > len(colors):
            print(f"Warning: Need {n_labels} colors but only {len(colors)} provided. Cycling through colors.")
            selected_colors = [colors[i % len(colors)] for i in range(n_labels)]
        discrete_cmap = mcolors.ListedColormap(selected_colors)
    else:
        # Use matplotlib colormap
        base_cmap = plt.cm.get_cmap(colors)
        color_values = base_cmap(np.linspace(0, 1, n_labels))
        discrete_cmap = mcolors.ListedColormap(color_values)
    
    # Create normalization for discrete values
    bounds = np.arange(len(unique_labels) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)
    
    # Map original labels to sequential indices
    label_to_index = {label_val: i for i, label_val in enumerate(unique_labels)}
    markers_indexed = np.array([label_to_index[label_val] for label_val in markers_flat])
    
    # Reshape back to original shape if needed
    if markers.ndim > 1:
        markers_indexed = markers_indexed.reshape(markers.shape)
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(markers_indexed, cmap=discrete_cmap, norm=norm, 
                   aspect='equal', origin='upper', extent=extent)
    
    # Create colorbar with discrete blocks
    cbar = plt.colorbar(im, ax=ax, boundaries=bounds, ticks=range(n_labels))
    
    # Set colorbar labels to original label values
    cbar.set_ticklabels([str(label_val) for label_val in unique_labels])
    cbar.set_label('Labels', rotation=270, labelpad=20)
    
    # Remove axis ticks and labels as requested
    if not set_ticks_flag:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add label to the figure if provided
    if label is not None:
        ax.text(-0.02, 1.042, f'({label})', transform=ax.transAxes, fontsize=26, 
                fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    return im


