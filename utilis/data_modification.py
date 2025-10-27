"""  
This script is for transfering the raw data into 
2d/1d, and corresponding angle shapes

1. Raw data are supposed to be 5d with a shape of (NoF, NoS, No3, Y, X), i.e., (NoOmega, Y, X, NoPhi, No2theta)
2. resulting data are acquired by integrating over the other axis

"""

import numpy as np
import pickle
from pathlib import Path
from functions_RSM import angular_position_to_plot_v_4


def data3D_to_1D(raw_data, save_flag=False, save_folder=None,
                 produce_1D_xs_flag=False, diffraction_info=None, 
                 set_info=None, image_clip_info=None):
    """ 
    Convert 3D diffraction data to 1D projections along different axes.
    
    Parameters:
    -----------
    raw_data : numpy.ndarray
        Input data with shape (NoOmega, Y, X, NoPhi, No2theta)
    save_flag : bool, default=False
        Whether to save the results to file
    save_folder : str or Path, optional
        Directory to save results (required if save_flag=True)
    produce_1D_xs_flag : bool, default=False
        Whether to calculate angular position arrays
    diffraction_info : dict, optional
        Diffraction parameters (required if produce_1D_xs_flag=True)
    set_info : dict, optional
        Setup information (required if produce_1D_xs_flag=True)
    image_clip_info : dict, optional
        Image clipping parameters (required if produce_1D_xs_flag=True)

    Returns:
    --------
    dict
        Dictionary containing 1D intensity arrays and optionally angular arrays
        Keys: 'o_I', 'phi_I', 'tt_I' and optionally 'X_o', 'X_tt', 'X_phi'
    """
    
    # Validate inputs
    if save_flag and save_folder is None:
        raise ValueError("save_folder must be provided when save_flag=True")
    
    if produce_1D_xs_flag and any(param is None for param in [diffraction_info, set_info, image_clip_info]):
        raise ValueError("diffraction_info, set_info, and image_clip_info must be provided when produce_1D_xs_flag=True")
    
    # Transpose: (NoOmega, Y, X, NoPhi, No2theta) -> (Y, X, NoOmega, NoPhi, No2theta)
    raw_data_T = np.transpose(raw_data, (1, 2, 0, 3, 4))
    _, _, shape2, shape3, shape4 = raw_data_T.shape 
    
    # Reshape to (Y*X, NoOmega, NoPhi, No2theta) for easier processing
    raw_data_T_arr = raw_data_T.reshape(-1, shape2, shape3, shape4)

    # Define axis indices for clarity
    axis_omega = 1
    axis_phi = 2
    axis_2theta = 3

    # Calculate 1D projections by summing over two axes
    data_o = np.sum(raw_data_T_arr, axis=(axis_phi, axis_2theta))
    data_phi = np.sum(raw_data_T_arr, axis=(axis_omega, axis_2theta))
    data_tt = np.sum(raw_data_T_arr, axis=(axis_omega, axis_phi))

    # Create base data dictionary
    data_dict = {
        'o_I': data_o,
        'phi_I': data_phi,
        'tt_I': data_tt
    }

    # Add angular position arrays if requested
    if produce_1D_xs_flag:
        aptp = angular_position_to_plot_v_4(
            diffraction_info=diffraction_info,
            set_info=set_info,
            image_clip_info=image_clip_info
        )
        
        data_dict.update({
            'X_o': aptp.omega_calculator(),
            'X_tt': aptp.ttheta_calculator(),
            'X_phi': aptp.phi_refer_calculator()
        })

    # Save to file if requested
    if save_flag:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        
        save_target = save_folder / '1D_I.pickle'
        with open(save_target, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Data saved to: {save_target}")

    return data_dict


def data_intergrate(data, axis_intergrate, axis_info):
    _, axis_phi, axis_2theta, axis_omega = axis_info 
    if axis_intergrate == '2theta':
        raw_array = np.sum(data, axis = axis_2theta)
        column_flag = ['phi', 'omega']
    if axis_intergrate == 'omega':
        raw_array = np.sum(data, axis = axis_omega)
        column_flag = ['phi', '2theta']
    if axis_intergrate == 'phi':
        raw_array = np.sum(data, axis = axis_phi)
        column_flag = ['2theta', 'omega']

    return raw_array, column_flag


# def data3D_to_2D(raw_data, save_flag=False, save_folder=None,
#                  produce_1D_xs_flag=False, diffraction_info=None, 
#                  set_info=None, image_clip_info=None):
#     """  
#         Convert 3D diffraction data to 2D projections along different axes.
        
#         Parameters:
#         -----------
#         raw_data : numpy.ndarray
#             Input data with shape (NoOmega, Y, X, NoPhi, No2theta)
#         save_flag : bool, default=False
#             Whether to save the results to file
#         save_folder : str or Path, optional
#             Directory to save results (required if save_flag=True)
#         produce_1D_xs_flag : bool, default=False
#             Whether to calculate angular position arrays
#         diffraction_info : dict, optional
#             Diffraction parameters (required if produce_1D_xs_flag=True)
#         set_info : dict, optional
#             Setup information (required if produce_1D_xs_flag=True)
#         image_clip_info : dict, optional
#             Image clipping parameters (required if produce_1D_xs_flag=True)

#         Returns:
#         --------
#         dict
#             Dictionary containing 1D intensity arrays and optionally angular arrays
#             Keys: 'o_tt', 'phi_tt', 'tt_omega' and optionally '[tt, o]', '[tt, phi]', '[o,phi]'
    
#     """

#     # prepare the shape of the raw data for following dealing
#     # transpose the raw data
#     NoOmega, Noa2, Noa3, NoPhi, No2theta = raw_data.shape
#     raw_data_T = np.transpose(raw_data, (1, 2, 3, 4, 0))
#     print(raw_data_T.shape)
#     raw_data_T_f = raw_data_T.reshape(Noa2*Noa3, NoPhi, No2theta, NoOmega)
#     print(raw_data_T_f.shape)

#     # so, for raw_data_T_f, we have
#     axis_P = 0
#     axis_phi = 1
#     axis_2theta = 2
#     axis_omega = 3
#     axis_info = [axis_P, axis_phi, axis_2theta, axis_omega]
#     for axis_intergrate in ['2theta', 'phi', 'omega']:
#         raw_array, column_flag = data_intergrate(raw_data_T_f, axis_intergrate, axis_info)

#         N = raw_array.shape[0]

#         for i in range(N):
#             point_array = raw_array[i, :, :]
#             aptp = angular_position_to_plot_v_4(diffraction_info=diffraction_info,
#                 set_info=set_info,
#                 image_clip_info=image_clip_info
#                 )
            