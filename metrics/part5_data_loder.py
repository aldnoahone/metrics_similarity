import pickle
import numpy as np

def label_data_loader(data_folder_path = None
                , region_detetion_data_name = 'region_detection.pickle'):
    
    # labels
    if data_folder_path:
        with open(data_folder_path + '/' + region_detetion_data_name, 'rb') as f:
            region_detetion_data =  pickle.load(f)
        region_detetion = region_detetion_data['markers'].reshape(-1)
    else:
        region_detetion = None
    
    return region_detetion

def data_loder_features(data_folder):
    X_phi_data_name = 'X_phi_directional_features_map.pickle'
    X_o_data_name = 'X_o_directional_features_map.pickle'
    X_tt_data_name = 'X_tt_directional_features_map.pickle'
    
    with open(data_folder + '/' + X_phi_data_name, 'rb') as f:
        X_phi_data =  pickle.load(f)
    
    with open(data_folder + '/' + X_o_data_name, 'rb') as f:
        X_o_data =  pickle.load(f)
    
    with open(data_folder + '/' + X_tt_data_name, 'rb') as f:
        X_tt_data =  pickle.load(f)

    return X_phi_data, X_o_data, X_tt_data