"""  
Some data and setting info of the nanoXRD experiment
that will be used in part3

"""
import numpy as np
import gc
def exp_set_loader_OVPE_2021(data_type = 's'):
    info_dict = {
        "as": {
            "diffraction_info":[
                1001.2093 # camera_L
                ,78.420# ttheta_base
                ,373.3235 # BSP 
                ,1.5498 # wl 
            ],
            'set_info':[
                671406/10000 # omega_start
                ,673006/10000 # omega_end
                ,20/10000 # omega_step
            ],
            'image_clip_info': [380, 49, 168, 168],
            'raw_data_path': "/mnt/y/results/2021B/HVPE_remake/2-81_2021B_o_remake_as.npz"
        },

        "s": {
            "diffraction_info":[
                1001.2093 # camera_L
                ,67.840# ttheta_base
                ,373.3235 # BSP 
                ,1.5498 # wl 
            ],
            'set_info':[
                337550/10000 # omega_start
                ,339150/10000 # omega_end
                ,20/10000 # omega_step
            ],
            'image_clip_info': [357, 0, 168, 168],
            'raw_data_path':  "/mnt/y/results/2021B/HVPE_remake/2-81_2021B_o_remake_s.npz"
        }
    }

    # extract the info
    info = []
    for name in ['diffraction_info', 'set_info', 'image_clip_info', 'raw_data_path']:
        info.append(info_dict[data_type][name])
    diffration_info, set_info, image_clip_info, raw_data_path = info
    raw_data = np.load(raw_data_path)['X']
    print(raw_data.shape) # NoP, NoOmega, NoPhi, No2theta

    return diffration_info, set_info, image_clip_info, raw_data

def exp_set_loader_ffc_2024(data_type = 's', sample_se = '2'):
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
    info_dict = {
        "as": {
            "diffraction_info":[
                993.2649, 79.024, 470.4567, 1.5498
            ],
            'set_info':[
                660238/10000, # axis1st_start, 
                662238/10000, # axis1st_end, 
                20/10000, # axis1st_step, 
            ],
        },

        "s": {
            "diffraction_info":[
                993.2649, 68.306, 470.4567, 1.5498
            ],
            'set_info':[
                326558/10000, # axis1st_start, 
                328558/10000, # axis1st_end, 
                20/10000, # axis1st_step, 
            ],
        }
    }

    image_clip_info = [401,162,112,112]
    raw_data_path = f"/home/aldnoah/vsc-projects/Na-flux_2024" \
    f"/output_save_3D/3d_{file_se }/3D_{file_se}.npz"

    # extract the info
    info = []
    for name in ['diffraction_info', 'set_info']:
        info.append(info_dict[data_type][name])
    diffration_info, set_info= info
    raw_data = np.load(raw_data_path)['X']
    
    # Transpose: (NoOmega, Y, X, NoPhi, No2theta) -> (Y, X, NoOmega, NoPhi, No2theta)
    raw_data = np.transpose(raw_data, (1, 2, 0, 3, 4))
    _, _, shape2, shape3, shape4 = raw_data.shape 
    raw_data = raw_data.reshape(-1, shape2, shape3, shape4)
    print(raw_data.shape) # NoP, NoOmega, NoPhi, No2theta
    
    return diffration_info, set_info, image_clip_info, raw_data