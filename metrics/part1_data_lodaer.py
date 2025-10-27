"""   
load the data for the preparation of the calculation

"""
import numpy as np
import pickle

# def data_loader_2024A(
#         X_path_0, X_name,
#         coor_path_0, coor_name
#         ):
#     # 1. load the raw data X, with a shape of (N, N_of_features)
#     X_path = X_path_0 + X_name
#     X = np.load(X_path)['X']
#     # 2. load the coor, with a shape of (N, 2) for (X, Y)
#     _coor_path = coor_path_0 + coor_name
#     with open(_coor_path, 'rb') as f:
#         _coor_raw = pickle.load(f)
#     S = _coor_raw['coor']
#     print(f"X and S have shapes of {X.shape}, {S.shape}")

#     return X, S

# def data_loader_ovpe_2021(
#         X_path_0 = '/mnt/y/VSC_projects/OVPE_remake/check/',
#         coor_path_0 = "/mnt/y/VSC_projects/OVPE_remake/check/coor_remake.pickle",
#         data_type = 'as'
#         ):
    
#     # 1. load the raw data and X
#     X_name = 'raw_file_' + data_type + '.pickle'
#     X_file_name = data_type + '_raw'
#     X_path = X_path_0 + X_name
#     with open(X_path, 'rb') as f:
#         raw_data = pickle.load(f)
#     print(f'raw file has the keys of {raw_data.keys()}')
#     X = raw_data[X_file_name]

#     # 2. load the coor
#     coor_name = 'coor_' + data_type  
#     S = raw_data[coor_name]

#     print(f"X and S have shapes of {X.shape}, {S.shape}")

#     return X, S

# def data_loader_hvpe_2021(
#         X_path_0 = '/mnt/y/results/2021B/HVPE_sciocs_raw_data/pickle/',
#         data_type = 's',
#     ):
#     # 1. load the raw data
#     X_path = X_path_0 + data_type + '.pickle'
#     with open(X_path, 'rb') as f:
#         raw_data = pickle.load(f)
    
#     # 2. load the X and S
#     X = raw_data['raw']
#     S = raw_data['coor']

#     print(f"X and S have shapes of {X.shape}, {S.shape}")

#     return X, S

import numpy as np
import pickle
import gc

def optimize_array_dtype(array):
    """Convert float64 to float32 for 50% memory reduction"""
    if array.dtype == np.float64:
        return array.astype(np.float32)
    return array

def data_loader_ovpe_2021_optimized(
    X_path_0='/mnt/y/VSC_projects/OVPE_remake/check/',
    coor_path_0="/mnt/y/VSC_projects/OVPE_remake/check/coor_remake.pickle",
    data_type='as',
    results_type = 'ALL' # 'ALL' or 'X' or 'S'
):
    """
    Memory-optimized data loader with immediate cleanup and dtype optimization
    """
    X = np.array([])
    S = np.array([])

    # 1. load the raw data and X
    if results_type == 'ALL' or results_type == 'X':
        X_name = 'raw_file_' + data_type + '.pickle'
        X_file_name = data_type + '_raw'
        X_path = X_path_0 + X_name
        
        with open(X_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        print(f'raw file has the keys of {raw_data.keys()}')
        
        # Extract arrays
        X = raw_data[X_file_name]
        S = raw_data['coor_' + data_type]

        # IMMEDIATE CLEANUP: Delete raw_data right after extraction
        del raw_data
        gc.collect()

        # DTYPE OPTIMIZATION: Convert to float32 for 50% memory reduction
        X = optimize_array_dtype(X)
        S = optimize_array_dtype(S)
    
    elif results_type == 'X' :

        X_name = 'raw_file_' + data_type + '.pickle'
        X_file_name = data_type + '_raw'
        X_path = X_path_0 + X_name
        
        with open(X_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        print(f'raw file has the keys of {raw_data.keys()}')
        
        # Extract arrays
        X = raw_data[X_file_name]
        # IMMEDIATE CLEANUP: Delete raw_data right after extraction
        del raw_data
        gc.collect()      

        X = optimize_array_dtype(X)

    elif results_type == 'S':
        with open(coor_path_0, 'rb') as f:
            S_raw = pickle.load(f)
        if data_type == 's':
            S = S_raw['coor_s']
        elif data_type == 'as':
            S = S_raw['coor_as']
    
    print(f"X and S have shapes of {X.shape}, {S.shape}")
    print(f"Memory usage: X={X.nbytes/1024**2:.1f}MB, S={S.nbytes/1024**2:.1f}MB")
    
    return X, S

# def data_loader_2024A_optimized(X_path_0, X_name, coor_path_0, coor_name):
#     """
#     Memory-optimized version of data_loader_2024A
#     """
#     # 1. load the raw data X
#     X_path = X_path_0 + X_name
#     X_data = np.load(X_path)
#     X = X_data['X']
    
#     # IMMEDIATE CLEANUP
#     del X_data
#     gc.collect()
    
#     # 2. load the coordinates
#     _coor_path = coor_path_0 + coor_name
#     with open(_coor_path, 'rb') as f:
#         _coor_raw = pickle.load(f)
    
#     S = _coor_raw['coor']
    
#     # IMMEDIATE CLEANUP
#     del _coor_raw
#     gc.collect()
    
#     # DTYPE OPTIMIZATION
#     X = optimize_array_dtype(X)
#     S = optimize_array_dtype(S)
    
#     print(f"X and S have shapes of {X.shape}, {S.shape}")
#     print(f"Memory usage: X={X.nbytes/1024**2:.1f}MB, S={S.nbytes/1024**2:.1f}MB")
    
#     return X, S

def data_loader_hvpe_2021_optimized(
    X_path_0='/mnt/y/results/2021B/HVPE_sciocs_raw_data/pickle/',
    data_type='s',
    coor_path_0 = "/mnt/y/results/2021B/HVPE_sciocs_raw_data/pickle/coor.pickle",
    results_type = 'ALL' # 'ALL' or 'X' or 'S'
):
    """
    Memory-optimized version of data_loader_hvpe_2021
    """
    X = np.array([])
    S = np.array([])

    if results_type == 'ALL':
        # 1. load the raw data
        X_path = X_path_0 + data_type + '.pickle'
        with open(X_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # 2. extract arrays
        X = raw_data['raw']
        S = raw_data['coor']
        
        # IMMEDIATE CLEANUP
        del raw_data
        gc.collect()
        
        # DTYPE OPTIMIZATION
        X = optimize_array_dtype(X)
        S = optimize_array_dtype(S)

    elif results_type == 'X':
        # 1. load the raw data
        X_path = X_path_0 + data_type + '.pickle'
        with open(X_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # 2. extract arrays
        X = raw_data['raw']
        
        # IMMEDIATE CLEANUP
        del raw_data
        gc.collect()
        
        # DTYPE OPTIMIZATION
        X = optimize_array_dtype(X)

    elif results_type == 'S':
        with open(coor_path_0, 'rb') as f:
            S_raw = pickle.load(f)
        S = S_raw['coor']
    
    print(f"X and S have shapes of {X.shape}, {S.shape}")
    print(f"Memory usage: X={X.nbytes/1024**2:.1f}MB, S={S.nbytes/1024**2:.1f}MB")
    
    return X, S

def data_loader_ffc_2024_optimized(
    X_path_0='/home/aldnoah/vsc-projects/Na-flux_2024/output_save_3D/flatten/',
    coor_path_0='/home/aldnoah/vsc-projects/transfer/experimental_data_2024A/data_folder/',
    data_type='as',
    sample_se = '1',
    results_type = 'ALL' # 'ALL' or 'X' or 'S'
):
    """
    Memory-optimized data loader with immediate cleanup and dtype optimization
    """
    X = np.array([])
    S = np.array([])

    # obtain the full name of the data
    lookup = {
            'as': {
                '1': '1',
                '2': '4',
                '3': '8',
            },
            's': {
                '1': '3',
                '2': '6',
                '3': '9'
            }
        }
    file_se = lookup[data_type][sample_se]
    coor_name = f'coor_{sample_se}.pickle'
    # the name of S
    _coor_path = coor_path_0 + coor_name
    #  the name of X
    X_name = '3d_' + file_se + '.npz'
    X_file_name = 'X'
    X_path = X_path_0 + X_name

    # 1. load the raw data and X
    if results_type == 'ALL':

        # Extract arrays
        X = np.load(X_path)[X_file_name]
        with open(_coor_path, 'rb') as f:
            _coor_raw = pickle.load(f)
        S = _coor_raw['coor']

        gc.collect()

        # DTYPE OPTIMIZATION: Convert to float32 for 50% memory reduction
        X = optimize_array_dtype(X)
        S = optimize_array_dtype(S)
    
    elif results_type == 'X' :
        X = np.load(X_path)[X_file_name]

        
        gc.collect()      

        X = optimize_array_dtype(X)

    elif results_type == 'S':
        with open(_coor_path, 'rb') as f:
            _coor_raw = pickle.load(f)
        S = _coor_raw['coor']
    
    print(f"X and S have shapes of {X.shape}, {S.shape}")
    print(f"Memory usage: X={X.nbytes/1024**2:.1f}MB, S={S.nbytes/1024**2:.1f}MB")
    
    return X, S