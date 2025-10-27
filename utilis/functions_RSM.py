import numpy as np
import numpy.ma as ma
import pandas as pd

from scipy.interpolate import griddata
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

from pathlib import Path
plt.rcParams['font.size'] = 56  # Change the font size
# plt.rcParams['text.usetex'] = False
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['text.usetex'] = True
# Useu latex to cover all of the font
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.unicode_minus'] = False

#### ver1.0 info###
     #  ver1.0: 20231130 
################
#### ver1.1 info###
    # 1. in funtion rsm_to_plt, change the mapping method from 
     # ax.scatter to ply.contourf
    # 2. add a new ploting function that direclt map the angular-Intensity
     # information without transforming to the RSM
#################
###### ver1.2 info####
    # 1. add a function of rsm_to_csv() 
    # to save the produced rsm plots
    # 2. Modify the omega calculation with the consideration
    # of if the first omega is needed
####################
##### ver1.2.1 info####
    # 1. adjust the padiing of output plots
    # 2. change the set of save to ensure the completeness of output
##########################

#### ver 1.2.2 info ####
    # in class angular_position_to_plot_v()
    # add three new variables
    # ttheta_range, omega_range, phi_range

##########################

class rsm_calculation():
     def __init__(self, diffraction_info, image_clip_info, set_info, raw_array):
          self.diffraction_info = diffraction_info # camera_L, ttheta_base, BSP, wl
          self.image_clip_info = image_clip_info # X_start, Y_start, X_range, Y_range from imageJ
          self.set_info = set_info # omega_s, omega_e, omega_step

          self.raw_array = raw_array

     def omega_2theta_tp_rsm(self,  ):
          camera_L, ttheta_base, BSP, wl = self.diffraction_info
          X_start, Y_start, X_range, Y_range = self.image_clip_info
          ttheta_deg = []
          ttheta_deg = [ttheta_base + (i-BSP)*100*(10 ** -3)\
              /camera_L*180/np.pi for i in range(X_start,X_start+X_range)] 
          return ttheta_deg

     def raw_to_rsm(self, first_omega_flag = False):
          raw_array = self.raw_array
          
          # transform the size of raw array into (NoOmega, No2theta)
          if len(raw_array.shape) == 3: # it is supposed that raw_array here has a size of (NoOmega, NoPhi, No2theta)
               omega_2theta = np.sum(raw_array, axis = 1)
          ## obtain the size of raw_array
          elif len(raw_array.shape) == 2: # it is supposed that raw_array here is already transformed into (NoOmega, No2theta)
               omega_2theta = raw_array
          NoF = omega_2theta.shape[0]
          NoC = omega_2theta.shape[1]

          ## obtain the Intensity, omega, and 2theta array
          # the Intensity array
          
          I_omega_2theta_flatten = omega_2theta.flatten()

          # calculate the 2theat values
          ttheta_deg = self.omega_2theta_tp_rsm()

          # define the array to store the values
          cor_omega = np.zeros([NoF * NoC, ],dtype = np.float32)
          cor_2theta = np.zeros([NoF * NoC, ],dtype = np.float32)
          qx_array = np.zeros([NoF * NoC, ],dtype = np.float32)
          qz_array = np.zeros([NoF * NoC, ],dtype = np.float32)

          # calculate the omega values, and put 2theta and omega values into the 
          # map array
          omega_s, omega_e, omega_step = self.set_info
          for i in range(NoF):
               for j in range(NoC):
                    cor_2theta[NoC * i + j] = ttheta_deg[j] 
                    if first_omega_flag == True: # which means include the first omega
                        cor_omega[NoC * i + j] = omega_s + i * omega_step
                    elif first_omega_flag == False: # which doesn not indluce the first omega
                        cor_omega[NoC * i + j] = omega_s + (i + 1) * omega_step
          
          ## calculate the corresponding qx and qz
          _, _, _, wl = self.diffraction_info
          for k in range(NoF*NoC):    
               qx_array[k] = 2/wl * np.sin(np.radians(cor_2theta[k]/2)) * np.sin(np.radians(cor_2theta[k]/2-cor_omega[k]))
               qz_array[k] = 2/wl * np.sin(np.radians(cor_2theta[k]/2)) * np.cos(np.radians(cor_2theta[k]/2-cor_omega[k]))

          return qx_array, qz_array, I_omega_2theta_flatten, cor_omega, cor_2theta
     
     def rsm_to_csv(self, csv_save_path, csv_save_name):
        csv_save = csv_save_path + '/' + csv_save_name + '.csv'
        qx_array, qz_array, I_omega_2theta, cor_omega, cor_2theta = self.raw_to_rsm()
        
        results = pd.DataFrame()

        results['qx']     = qx_array
        results['qz']     = qz_array
        results['I']      = I_omega_2theta
        results['omega']  = cor_omega
        results['2theta'] = cor_2theta

        results.to_csv(csv_save, index=None, header=True)

     def rsm_to_plot(self, Normalize_flag = True, save_flag = False, save_path = None,
                     gridplot_number = 500, log_flag = True
                     , Threshold_flag_ratio = False, Threshold_ratio = 20
                     , Threshold_flag_value = False, Threshold_value = 0.1
                     , RSM_calculation_flag = True, RSM_array = None
                     , Label_mode = False, n_c = 2
                     , plt_text_dict = None):
            
            if RSM_calculation_flag == True:
                qx_array, qz_array, I_omega_2theta, _, _ = self.raw_to_rsm()
            else:
                 qx_array = RSM_array[:,0]
                 qz_array = RSM_array[:,1]
                 I_omega_2theta = RSM_array[:,2]
          
            ## calculate the gridplot [QXi, QZi] from the raw [qx, qz]
            qxi = np.linspace(qx_array.min(), qx_array.max(), gridplot_number)
            qzi = np.linspace(qz_array.min(), qz_array.max(), gridplot_number)
            QXi, QZi = np.meshgrid(qxi, qzi)

            ## calculate corresponing Ii based on the gridplot [QXi, QZi]
            Ii = griddata(points = (qx_array, qz_array)
                            , values = I_omega_2theta
                        ,xi = (QXi, QZi),  method = 'linear')

            Ii = np.nan_to_num(Ii)
            
            ## add threshold if nexessary, which is set as False for default
            if Threshold_flag_ratio == True:
                threshold = Ii.max()/Threshold_ratio ########## change

                binary_mask = Ii >= threshold
                
                Ii = ma.array(Ii, mask= ~binary_mask)

                Ii = Ii.filled(0)
            
            elif Threshold_flag_value == True:
                threshold = Threshold_value ########## change

                binary_mask = Ii >= threshold

                Ii = ma.array(Ii, mask= ~binary_mask)

                Ii = Ii.filled(0)

            ## add if normalize is necessary, which is set as True in default
            if Normalize_flag == True:
                Ii = (Ii - Ii.min())/ (Ii.max()- Ii.min()) + 1e-10 ## avoid the log10 problem
                # define the levels correspning to the 1e-10
                levels = np.arange(-10, 0.25, 0.25)    
                
            elif Normalize_flag == False:
                # step = np.log10(Ii.max())/20
                # print(step)
                Ii = Ii + 1
                # levels = np.arange(0, np.log10(Ii.max()) + 0.25, 0.25)
                levels = np.arange(0, 6 + 0.125/2, 0.125)

            ## map the results[QXi, QZi, Intensity]
            ## the color is based on log(Intensity)

            if Label_mode == True:
                cmap = cm.get_cmap('Set3', n_c)
            else:
                cmap = cm.get_cmap('gist_ncar', len(levels))

            fig,ax = plt.subplots(1,1, figsize = (10,10))
            ax = plt.gca()
            # levels_0 = np.linspace(Ii_log.min(),1,100)
            ax.tick_params(bottom=False,
                        left=False,
                        right=False,
                        top=False)
            print(Ii.shape)
            if log_flag == True and Label_mode == False:
                Ii_log = np.log10(Ii)
                map_0 = plt.contourf(QXi, QZi, Ii_log, cmap=cmap, levels = levels)
                # cbar = fig.colorbar(map_0, ax = ax)
                x_major_locator = MultipleLocator(0.001)
                x_minor_locator = MultipleLocator(0.0002)
                # cbar.set_label("log(Intensity)")

            elif log_flag == False and Label_mode == False:
                map_0 = plt.contourf(QXi, QZi, Ii, cmap=cmap, levels = levels)
                # cbar = fig.colorbar(map_0, ax = ax)
                x_major_locator = MultipleLocator(0.001)
                x_minor_locator = MultipleLocator(0.0002)
                # cbar.set_label("Intensity")

            elif log_flag == True and Label_mode == True:
                Ii_log = np.log10(Ii)
                map_0 = plt.contourf(QXi, QZi, Ii_log, cmap=cmap, levels = levels)
                # cbar = fig.colorbar(map_0, ax = ax)
                x_major_locator = MultipleLocator(0.0001)
                x_minor_locator = MultipleLocator(0.00005)
                # cbar.set_label("label")

            elif log_flag == False and Label_mode == True:
                map_0 = plt.contourf(QXi, QZi, Ii, cmap=cmap, levels = levels)
                # cbar = fig.colorbar(map_0, ax = ax)
                x_major_locator = MultipleLocator(0.0001)
                x_minor_locator = MultipleLocator(0.00005)
                # cbar.set_label("label")
            # set details of the plot
                
            if Normalize_flag == True:    
                cbar_ticks = np.arange(-10, 1, 2)  # Ticks from -10 to 0 with step of 2
            elif Normalize_flag == False:
                cbar_ticks = np.arange(0, levels.max() + 1, 2)   

            cbar = plt.colorbar(map_0)
            cbar.set_ticks(cbar_ticks)
            
            if log_flag == True:
                cbar.set_label("log(Intensity)")
            elif log_flag == False:
                cbar.set_label("Intensity")
                
            ax.xaxis.set_major_locator(x_major_locator)
            ax.xaxis.set_minor_locator(x_minor_locator)

            y_major_locator = MultipleLocator(0.0005)
            y_minor_locator = MultipleLocator(0.0001)
            ax.yaxis.set_major_locator(y_major_locator)
            ax.yaxis.set_minor_locator(y_minor_locator)

            plt.xlabel(r'$Q_x\ [\AA^{-1}$]')
            plt.ylabel(r'$Q_z\ [\AA^{-1}$]')
            
            if plt_text_dict != None:
                plt.figtext(plt_text_dict['x'], plt_text_dict['y'], plt_text_dict['plt_text'], ha="center"
                    , fontsize=18, bbox={"facecolor":"white", "alpha":1.0, "pad":5})            
            
            # Adjust padding
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

            # plt.tight_layout()
            if save_flag == True:
                plt.savefig(save_path, dpi = 300, bbox_inches='tight')

            else:
                 return fig
            
class angular_position_to_plot():
    
    def __init__ (self, diffraction_info, set_info, image_clip_info):
        self.diffraction_info = diffraction_info
        self.set_info = set_info
        self.image_clip_info = image_clip_info

    def phi_refer_calculator(self, phi_base = 0, BSP = 0 ):
        camera_L, _, _, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        phi_refer_deg = []
        phi_refer_deg = [phi_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(Y_start,Y_start+Y_range)] 
        return phi_refer_deg
    
    def ttheta_calculator(self,  ):
        camera_L, ttheta_base, BSP, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        ttheta_deg = []
        ttheta_deg = [ttheta_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(X_start,X_start+X_range)] 
        return ttheta_deg


    def omega_calculator(self, if_need_first_omega_flag = False):
        omega_s, omega_e, omega_step = self.set_info
        omega_deg = []
        omega_deg = np.arange(omega_s, omega_e + omega_step/2, omega_step)
        if if_need_first_omega_flag == False: # false means we do not need the first omega
            omega_deg = omega_deg[1:]
        return omega_deg

    def plot(self, column_flag = ['omega', 'phi'],
             Normalize_flag = True, save_flag = False, save_path = None,
                gridplot_number = 500, log_flag = True, if_need_first_omega_flag = False
                , Threshold_flag_ratio = False, Threshold_ratio = 20
                , Threshold_flag_value = False, Threshold_value = 0.1
                , angular_array = None
                , Label_mode = False, n_c = 2,
                plt_text_dict = None):

        # input the array with the 2D angular positioned Intensity
        # and transfer raw data into gridplot
        ref_deg_dict = {
            'omega':self.omega_calculator(if_need_first_omega_flag = if_need_first_omega_flag),
            '2theta':self.ttheta_calculator(),
            'phi':self.phi_refer_calculator()
        }

        rows_name, columns_name = column_flag
        
        print(rows_name, columns_name)
        
        rows_raw = ref_deg_dict[rows_name]
        columns_raw = ref_deg_dict[columns_name]
        
        print(np.shape(rows_raw), np.shape(columns_raw))

        # transforms the columns to Xs, rows to Ys
        Angular_1_raw, Angular_2_raw = np.meshgrid(columns_raw, rows_raw) 

        angular_1 = Angular_1_raw.ravel()
        angular_2 = Angular_2_raw.ravel()
        Intensity_map = angular_array.flatten()

        print(Angular_1_raw.shape, Angular_2_raw.shape, angular_array.shape)
    
        # calculate the gridplot [QXi, QZi] from the raw [qx, qz]
        angular_1i = np.linspace(angular_1.min(), angular_1.max(), gridplot_number)
        angular_2i = np.linspace(angular_2.min(), angular_2.max(), gridplot_number)
        A_1i, A_2i = np.meshgrid(angular_1i, angular_2i)

        ## calculate corresponing Ii based on the gridplot [QXi, QZi]
        Ii = griddata(points = (angular_1, angular_2)
                        , values = Intensity_map
                    ,xi = (A_1i, A_2i),  method = 'linear')

        Ii = np.nan_to_num(Ii)
        
        ## add threshold if nexessary, which is set as False for default
        if Threshold_flag_ratio == True:
            threshold = Ii.max()/Threshold_ratio ########## change

            binary_mask = Ii >= threshold

            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
            
        elif Threshold_flag_value == True:
            threshold = Threshold_value ########## change

            binary_mask = Ii >= threshold
            
            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
        
        ###########test#########
        # A_1i = np.array(Angular_1_raw)
        # A_2i = np.array(Angular_2_raw)
        # Ii = np.array(angular_array)
        ###########test#########

        ## add if normalize is necessary, which is set as True in default
        if Normalize_flag == True:
            Ii = (Ii - Ii.min())/ (Ii.max()- Ii.min()) + 1e-10 ## avoid the log10 problem
            # define the levels correspning to the 1e-10
            levels = np.arange(-10, 0.25, 0.125)
        elif Normalize_flag == False:
            Ii = Ii + 1
            levels = np.arange(0, 6.25, 0.125)

        ## map the results[QXi, QZi, Intensity]
        ## the color is based on log(Intensity)

        if Label_mode == True:
            cmap = cm.get_cmap('Set3', n_c)
        else:
            cmap = cm.get_cmap('gist_ncar', len(levels))

        fig,ax = plt.subplots(1,1, figsize = (10,10))
        ax = plt.gca()
        # levels_0 = np.linspace(Ii_log.min(),1,100)
        ax.tick_params(bottom=True,
                    left=True,
                    right=False,
                    top=False)

        if log_flag == True and column_flag == ['phi', 'omega']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.2)
            y_minor_locator = MultipleLocator(0.04)
            x_major_locator = MultipleLocator(0.05)
            x_minor_locator = MultipleLocator(0.01)
            # cbar.set_label("log(Intensity)")
            plt.ylabel(r'$\mathit{\varphi}$ [degree]')
            plt.xlabel(r'$\mathit{\omega}$  [degree]')
        
        elif log_flag == True and column_flag == ['phi', '2theta']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.2)
            y_minor_locator = MultipleLocator(0.04)
            x_major_locator = MultipleLocator(0.25)
            x_minor_locator = MultipleLocator(0.05)
            # cbar.set_label("log(Intensity)")
            plt.ylabel(r'$\mathit{\varphi}$ [degree]')
            plt.xlabel(r'2$\mathit{\theta}$ [degree]')
        
        elif log_flag == True and column_flag == ['2theta', 'omega']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.2)
            y_minor_locator = MultipleLocator(0.04)
            x_major_locator = MultipleLocator(0.05)
            x_minor_locator = MultipleLocator(0.01)
            # cbar.set_label("log(Intensity)")
            plt.ylabel(r'2$\mathit{\theta}$ [degree]')
            plt.xlabel(r'$\mathit{\omega}$ [degree]')
            
            
        elif log_flag == False and Label_mode == False:
            map_0 = ax.scatter(A_1i, A_2i, c = Ii, cmap=cmap)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.001)
            x_minor_locator = MultipleLocator(0.0002)
            # cbar.set_label("Intensity")

        elif log_flag == True and Label_mode == True:
            Ii_log = np.log10(Ii)
            map_0 = ax.scatter(A_1i, A_2i, c = Ii_log, cmap=cmap)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.0001)
            x_minor_locator = MultipleLocator(0.00005)
            # cbar.set_label("label")

        elif log_flag == False and Label_mode == True:
            map_0 = ax.scatter(A_1i, A_2i, c = Ii, cmap=cmap)
            ## cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.0001)
            x_minor_locator = MultipleLocator(0.00005)
            # cbar.set_label("label")
        # set details of the plot
        
        plt.tick_params(axis='both', which='major', length=11)
        plt.tick_params(axis='both', which='minor', length=5)

        if Normalize_flag == True:    
            cbar_ticks = np.arange(-10, 1, 2)  # Ticks from -10 to 0 with step of 2
        elif Normalize_flag == False:
            cbar_ticks = np.arange(0, levels.max() + 1, 2)   

        cbar = plt.colorbar(map_0)
        cbar.set_ticks(cbar_ticks)
        if log_flag == True:
            cbar.set_label(r'$\log_{10}(\mathrm{Intensity})$ [a.u.]')
        elif log_flag == False:
            cbar.set_label("Intensity [counts]")

        ax.xaxis.set_major_locator(x_major_locator)
        ax.xaxis.set_minor_locator(x_minor_locator)


        ax.yaxis.set_major_locator(y_major_locator)
        ax.yaxis.set_minor_locator(y_minor_locator)
        
        # Rotate x-axis labels
        # plt.xticks(rotation=45)

        # plt_text_dict['plt_text'] = 't'
        
        print(plt_text_dict['plt_text'])
        
        ### add some texts in the figure
        if plt_text_dict != None:
            plt.figtext(plt_text_dict['x'], plt_text_dict['y'], plt_text_dict['plt_text'], ha="center"
                        , fontsize=48, bbox={"facecolor":"white", "alpha":1.0, "pad":5})
            
        # Adjust padding
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # plt.tight_layout()
        if save_flag == True:
            plt.savefig(save_path, dpi = 300, bbox_inches='tight')

        else:
                return fig
        
class angular_position_to_plot_v():
    
    def __init__ (self, diffraction_info, set_info, image_clip_info
                  , ttheta_range = None
                  , omega_range  = None 
                  , phi_range    = None):
        self.diffraction_info = diffraction_info
        self.set_info = set_info
        self.image_clip_info = image_clip_info
        self.ttheta_range = ttheta_range
        self.omega_range  = omega_range
        self.phi_range    = phi_range

    def phi_refer_calculator(self, phi_base = 0, BSP = 0, phi_range = None):
        camera_L, _, _, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        phi_refer_deg = []
        phi_refer_deg = [phi_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(Y_start,Y_start+Y_range)] 
        if phi_range == None:
            phi_refer_deg = phi_refer_deg
        else:
            phi_refer_deg = phi_refer_deg[phi_range[0]:(phi_range[1] + 1)]
        return phi_refer_deg
    
    def ttheta_calculator(self,  tthe_range = None):
        camera_L, ttheta_base, BSP, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        ttheta_deg = []
        ttheta_deg = [ttheta_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(X_start,X_start+X_range)] 
        if tthe_range == None:
            ttheta_deg = ttheta_deg
        else:
            ttheta_deg = ttheta_deg[tthe_range[0]:(tthe_range[1] + 1)]

        return ttheta_deg


    def omega_calculator(self, if_need_first_omega_flag = False, omega_range = None):
        omega_s, omega_e, omega_step = self.set_info
        omega_deg = []
        omega_deg = np.arange(omega_s, omega_e + omega_step/2, omega_step)
        if if_need_first_omega_flag == False: # false means we do not need the first omega
            omega_deg = omega_deg[1:]

        if omega_range == None:
            omega_deg = omega_deg
        else:
            omega_deg = omega_deg[omega_range[0]:(omega_range[1] + 1)]

        return omega_deg

    def plot(self, column_flag = ['omega', 'phi'],
             Normalize_flag = True, save_flag = False, save_path = None,
                gridplot_number = 500, log_flag = True, if_need_first_omega_flag = False
                , Threshold_flag_ratio = False, Threshold_ratio = 20
                , Threshold_flag_value = False, Threshold_value = 0.1
                , angular_array = None
                , Label_mode = False, n_c = 2,
                plt_text_dict = None):

        # input the array with the 2D angular positioned Intensity
        # and transfer raw data into gridplot
        ref_deg_dict = {
            'omega':self.omega_calculator(if_need_first_omega_flag = if_need_first_omega_flag, omega_range=self.omega_range),
            '2theta':self.ttheta_calculator(tthe_range=self.ttheta_range),
            'phi':self.phi_refer_calculator(phi_range=self.phi_range)
        }

        rows_name, columns_name = column_flag
        
        print(rows_name, columns_name)
        
        rows_raw = ref_deg_dict[rows_name]
        columns_raw = ref_deg_dict[columns_name]
        
        print(np.shape(rows_raw), np.shape(columns_raw))

        # transforms the columns to Xs, rows to Ys
        Angular_1_raw, Angular_2_raw = np.meshgrid(columns_raw, rows_raw) 

        angular_1 = Angular_1_raw.ravel()
        angular_2 = Angular_2_raw.ravel()
        Intensity_map = angular_array.flatten()

        print(Angular_1_raw.shape, Angular_2_raw.shape, angular_array.shape)
    
        # calculate the gridplot [QXi, QZi] from the raw [qx, qz]
        angular_1i = np.linspace(angular_1.min(), angular_1.max(), gridplot_number)
        angular_2i = np.linspace(angular_2.min(), angular_2.max(), gridplot_number)
        A_1i, A_2i = np.meshgrid(angular_1i, angular_2i)

        ## calculate corresponing Ii based on the gridplot [QXi, QZi]
        Ii = griddata(points = (angular_1, angular_2)
                        , values = Intensity_map
                    ,xi = (A_1i, A_2i),  method = 'linear')

        Ii = np.nan_to_num(Ii)
        
        ## add threshold if nexessary, which is set as False for default
        if Threshold_flag_ratio == True:
            threshold = Ii.max()/Threshold_ratio ########## change

            binary_mask = Ii >= threshold

            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
            
        elif Threshold_flag_value == True:
            threshold = Threshold_value ########## change

            binary_mask = Ii >= threshold
            
            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
        
        ###########test#########
        # A_1i = np.array(Angular_1_raw)
        # A_2i = np.array(Angular_2_raw)
        # Ii = np.array(angular_array)
        ###########test#########

        ## add if normalize is necessary, which is set as True in default
        if Normalize_flag == True:
            Ii = (Ii - Ii.min())/ (Ii.max()- Ii.min()) + 1e-10 ## avoid the log10 problem
            # define the levels correspning to the 1e-10
            levels = np.arange(-10, 0.25, 0.125)
        elif Normalize_flag == False:
            Ii = Ii + 1
            levels = np.arange(0, 6.25, 0.125)

        ## map the results[QXi, QZi, Intensity]
        ## the color is based on log(Intensity)

        if Label_mode == True:
            cmap = cm.get_cmap('Set3', n_c)
        else:
            cmap = cm.get_cmap('gist_ncar', len(levels))

        fig,ax = plt.subplots(1,1, figsize = (10,10))
        ax = plt.gca()
        # levels_0 = np.linspace(Ii_log.min(),1,100)
        ax.tick_params(bottom=True,
                    left=True,
                    right=False,
                    top=False)

        if log_flag == True and column_flag == ['phi', 'omega']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.2)
            y_minor_locator = MultipleLocator(0.04)
            x_major_locator = MultipleLocator(0.05)
            x_minor_locator = MultipleLocator(0.01)
            # cbar.set_label("log(Intensity)")
            plt.ylabel(r'$\mathit{\varphi}$ [degree]', rotation=270, labelpad=55)
            plt.xlabel(r'$\mathit{\omega}$  [degree]')
        
        elif log_flag == True and column_flag == ['phi', '2theta']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.2)
            y_minor_locator = MultipleLocator(0.04)
            x_major_locator = MultipleLocator(0.15)
            x_minor_locator = MultipleLocator(0.03)
            # cbar.set_label("log(Intensity)")
            plt.ylabel(r'$\mathit{\varphi}$ [degree]', rotation=270, labelpad=55)
            plt.xlabel(r'2$\mathit{\theta}$ [degree]')
        
        elif log_flag == True and column_flag == ['2theta', 'omega']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.15)
            y_minor_locator = MultipleLocator(0.03)
            x_major_locator = MultipleLocator(0.05)
            x_minor_locator = MultipleLocator(0.01)
            # cbar.set_label("log(Intensity)")
            plt.ylabel(r'2$\mathit{\theta}$ [degree]', rotation=270, labelpad=55)
            plt.xlabel(r'$\mathit{\omega}$ [degree]')
            
            
        elif log_flag == False and Label_mode == False:
            map_0 = ax.scatter(A_1i, A_2i, c = Ii, cmap=cmap)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.001)
            x_minor_locator = MultipleLocator(0.0002)
            # cbar.set_label("Intensity")

        elif log_flag == True and Label_mode == True:
            Ii_log = np.log10(Ii)
            map_0 = ax.scatter(A_1i, A_2i, c = Ii_log, cmap=cmap)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.0001)
            x_minor_locator = MultipleLocator(0.00005)
            # cbar.set_label("label")

        elif log_flag == False and Label_mode == True:
            map_0 = ax.scatter(A_1i, A_2i, c = Ii, cmap=cmap)
            ## cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.0001)
            x_minor_locator = MultipleLocator(0.00005)
            # cbar.set_label("label")
        # set details of the plot
        
        plt.tick_params(axis='both', which='major', length=11)
        plt.tick_params(axis='both', which='minor', length=5)

        if Normalize_flag == True:    
            cbar_ticks = np.arange(-10, 1, 2)  # Ticks from -10 to 0 with step of 2
        elif Normalize_flag == False:
            cbar_ticks = np.arange(0, levels.max() + 1, 2)   

        cbar = plt.colorbar(map_0)
        cbar.set_ticks(cbar_ticks)
        
        if log_flag == True:
            cbar.set_label(r'$\log_{10}(\mathrm{Intensity})$ [a.u.]',rotation=-90, labelpad=55)
        elif log_flag == False:
            cbar.set_label("Intensity [counts]",rotation=-90, labelpad=55)

        ax.xaxis.set_major_locator(x_major_locator)
        ax.xaxis.set_minor_locator(x_minor_locator)


        ax.yaxis.set_major_locator(y_major_locator)
        ax.yaxis.set_minor_locator(y_minor_locator)

        # plt_text_dict['plt_text'] = 't'
        
        print(plt_text_dict['plt_text'])
        
        # add rotation
        for tick in ax.get_yticklabels():
                tick.set_rotation(270)
        for tick in cbar.ax.get_yticklabels():
                tick.set_rotation(270)
        # ax.set_ylabel(rotation=270, labelpad=15)

        ### add some texts in the figure
        if plt_text_dict != None:
            plt.figtext(plt_text_dict['x'], plt_text_dict['y'], plt_text_dict['plt_text'], ha="center"
                        ,rotation=-90
                        , fontsize=48, bbox={"facecolor":"white", "alpha":1.0, "pad":5})
            
        # Adjust padding
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # plt.tight_layout()
        if save_flag == True:
            plt.savefig(save_path, dpi = 300, bbox_inches='tight')

        else:
                return fig
        
class angular_position_to_plot_v_2():
    
    """
    Compared to the _v, remove the title and colorbar information

    """


    def __init__ (self, diffraction_info, set_info, image_clip_info
                  , ttheta_range = None
                  , omega_range  = None 
                  , phi_range    = None):
        self.diffraction_info = diffraction_info
        self.set_info = set_info
        self.image_clip_info = image_clip_info
        self.ttheta_range = ttheta_range
        self.omega_range  = omega_range
        self.phi_range    = phi_range

    def phi_refer_calculator(self, phi_base = 0, BSP = 0, phi_range = None):
        camera_L, _, _, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        phi_refer_deg = []
        phi_refer_deg = [phi_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(Y_start,Y_start+Y_range)] 
        if phi_range == None:
            phi_refer_deg = phi_refer_deg
        else:
            phi_refer_deg = phi_refer_deg[phi_range[0]:(phi_range[1] + 1)]
        return phi_refer_deg
    
    def ttheta_calculator(self,  tthe_range = None):
        camera_L, ttheta_base, BSP, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        ttheta_deg = []
        ttheta_deg = [ttheta_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(X_start,X_start+X_range)] 
        if tthe_range == None:
            ttheta_deg = ttheta_deg
        else:
            ttheta_deg = ttheta_deg[tthe_range[0]:(tthe_range[1] + 1)]

        return ttheta_deg


    def omega_calculator(self, if_need_first_omega_flag = False, omega_range = None):
        omega_s, omega_e, omega_step = self.set_info
        omega_deg = []
        omega_deg = np.arange(omega_s, omega_e + omega_step/2, omega_step)
        if if_need_first_omega_flag == False: # false means we do not need the first omega
            omega_deg = omega_deg[1:]

        if omega_range == None:
            omega_deg = omega_deg
        else:
            omega_deg = omega_deg[omega_range[0]:(omega_range[1] + 1)]

        return omega_deg

    def plot(self, column_flag = ['omega', 'phi'],
             Normalize_flag = True, save_flag = False, save_path = None,
                gridplot_number = 500, log_flag = True, if_need_first_omega_flag = False
                , Threshold_flag_ratio = False, Threshold_ratio = 20
                , Threshold_flag_value = False, Threshold_value = 0.1
                , angular_array = None
                , Label_mode = False, n_c = 2
                , plt_text_dict = None
                , x_guideline_flag = False, X_0 = 0
                , y_guideline_flag = False, Y_0 = 0):

        # input the array with the 2D angular positioned Intensity
        # and transfer raw data into gridplot
        ref_deg_dict = {
            'omega':self.omega_calculator(if_need_first_omega_flag = if_need_first_omega_flag, omega_range=self.omega_range),
            '2theta':self.ttheta_calculator(tthe_range=self.ttheta_range),
            'phi':self.phi_refer_calculator(phi_range=self.phi_range)
        }

        rows_name, columns_name = column_flag
        
        print(rows_name, columns_name)
        
        rows_raw = ref_deg_dict[rows_name]
        columns_raw = ref_deg_dict[columns_name]
        
        print(np.shape(rows_raw), np.shape(columns_raw))

        # transforms the columns to Xs, rows to Ys
        Angular_1_raw, Angular_2_raw = np.meshgrid(columns_raw, rows_raw) 

        angular_1 = Angular_1_raw.ravel()
        angular_2 = Angular_2_raw.ravel()
        Intensity_map = angular_array.flatten()

        print(Angular_1_raw.shape, Angular_2_raw.shape, angular_array.shape)
    
        # calculate the gridplot [QXi, QZi] from the raw [qx, qz]
        angular_1i = np.linspace(angular_1.min(), angular_1.max(), gridplot_number)
        angular_2i = np.linspace(angular_2.min(), angular_2.max(), gridplot_number)
        A_1i, A_2i = np.meshgrid(angular_1i, angular_2i)

        ## calculate corresponing Ii based on the gridplot [QXi, QZi]
        Ii = griddata(points = (angular_1, angular_2)
                        , values = Intensity_map
                    ,xi = (A_1i, A_2i),  method = 'linear')

        Ii = np.nan_to_num(Ii)
        
        ## add threshold if nexessary, which is set as False for default
        if Threshold_flag_ratio == True:
            threshold = Ii.max()/Threshold_ratio ########## change

            binary_mask = Ii >= threshold

            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
            
        elif Threshold_flag_value == True:
            threshold = Threshold_value ########## change

            binary_mask = Ii >= threshold
            
            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
        
        ###########test#########
        # A_1i = np.array(Angular_1_raw)
        # A_2i = np.array(Angular_2_raw)
        # Ii = np.array(angular_array)
        ###########test#########

        ## add if normalize is necessary, which is set as True in default
        if Normalize_flag == True:
            Ii = (Ii - Ii.min())/ (Ii.max()- Ii.min()) + 1e-10 ## avoid the log10 problem
            # define the levels correspning to the 1e-10
            levels = np.arange(-10, 0.25, 0.125)
        elif Normalize_flag == False:
            Ii = Ii + 1
            levels = np.arange(0, 6.25, 0.125)

        ## map the results[QXi, QZi, Intensity]
        ## the color is based on log(Intensity)

        if Label_mode == True:
            cmap = cm.get_cmap('Set3', n_c)
        else:
            cmap = cm.get_cmap('gist_ncar', len(levels))

        fig,ax = plt.subplots(1,1, figsize = (10,10))
        ax = plt.gca()
        # levels_0 = np.linspace(Ii_log.min(),1,100)
        ax.tick_params(bottom=True,
                    left=True,
                    right=False,
                    top=False)

        if log_flag == True and column_flag == ['phi', 'omega']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.1)
            y_minor_locator = MultipleLocator(0.02)
            x_major_locator = MultipleLocator(0.06)
            x_minor_locator = MultipleLocator(0.012)
            # cbar.set_label("log(Intensity)")
            # plt.ylabel(r'$\mathit{\varphi}$ [degree]', rotation=270, labelpad=55)
            # plt.xlabel(r'$\mathit{\omega}$  [degree]')
        
        elif log_flag == True and column_flag == ['phi', '2theta']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.1)
            y_minor_locator = MultipleLocator(0.02)
            x_major_locator = MultipleLocator(0.15)
            x_minor_locator = MultipleLocator(0.03)
            # cbar.set_label("log(Intensity)")
            # plt.ylabel(r'$\mathit{\varphi}$ [degree]', rotation=270, labelpad=55)
            # plt.xlabel(r'2$\mathit{\theta}$ [degree]')
        
        elif log_flag == True and column_flag == ['2theta', 'omega']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.15)
            y_minor_locator = MultipleLocator(0.03)
            x_major_locator = MultipleLocator(0.05)
            x_minor_locator = MultipleLocator(0.01)
            # cbar.set_label("log(Intensity)")
            # plt.ylabel(r'2$\mathit{\theta}$ [degree]', rotation=270, labelpad=55)
            # plt.xlabel(r'$\mathit{\omega}$ [degree]')
            
            
        elif log_flag == False and Label_mode == False:
            map_0 = ax.scatter(A_1i, A_2i, c = Ii, cmap=cmap)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.001)
            x_minor_locator = MultipleLocator(0.0002)
            # cbar.set_label("Intensity")

        elif log_flag == True and Label_mode == True:
            Ii_log = np.log10(Ii)
            map_0 = ax.scatter(A_1i, A_2i, c = Ii_log, cmap=cmap)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.0001)
            x_minor_locator = MultipleLocator(0.00005)
            # cbar.set_label("label")

        elif log_flag == False and Label_mode == True:
            map_0 = ax.scatter(A_1i, A_2i, c = Ii, cmap=cmap)
            ## cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.0001)
            x_minor_locator = MultipleLocator(0.00005)
            # cbar.set_label("label")
        # set details of the plot
        
        plt.tick_params(axis='both', which='major', length=11)
        plt.tick_params(axis='both', which='minor', length=5)

        if Normalize_flag == True:    
            cbar_ticks = np.arange(-10, 1, 2)  # Ticks from -10 to 0 with step of 2
        elif Normalize_flag == False:
            cbar_ticks = np.arange(0, levels.max() + 1, 2)   

        # cbar = plt.colorbar(map_0)
        # cbar.set_ticks(cbar_ticks)
        
        # if log_flag == True:
        #     cbar.set_label(r'$\log_{10}(\mathrm{Intensity})$ [a.u.]',rotation=-90, labelpad=55)
        # elif log_flag == False:
        #     cbar.set_label("Intensity [counts]",rotation=-90, labelpad=55)

        ax.xaxis.set_major_locator(x_major_locator)
        ax.xaxis.set_minor_locator(x_minor_locator)


        ax.yaxis.set_major_locator(y_major_locator)
        ax.yaxis.set_minor_locator(y_minor_locator)

        # add a guideline as specific position
        if x_guideline_flag == True:
            ax.axvline(x=X_0,  color='#444444', linestyle='--',  linewidth=9, label=f'X = {X_0}')
        elif y_guideline_flag == True:
            ax.axhline(y=Y_0,  color='#444444', linestyle='--',  linewidth=9, label=f'Y = {Y_0}')

        # add rotation
        for tick in ax.get_yticklabels():
                tick.set_rotation(270)
        # for tick in cbar.ax.get_yticklabels():
        #         tick.set_rotation(270)
        # ax.set_ylabel(rotation=270, labelpad=15)

        ### add some texts in the figure
        if plt_text_dict != None:
            print(plt_text_dict['plt_text'])
            plt.figtext(plt_text_dict['x'], plt_text_dict['y'], plt_text_dict['plt_text'], ha="center"
                        ,rotation=-90
                        , fontsize=48, bbox={"facecolor":"white", "alpha":1.0, "pad":5})
            
        # Adjust padding
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # plt.tight_layout()
        if save_flag == True:
            plt.savefig(save_path, dpi = 300, bbox_inches='tight')

        else:
                return fig
        
class angular_position_to_plot_v_3():
    
    """
    Compared to the _v, remove the title and colorbar information
    Compared to the _v_2, change the order of  x and y for input

    """


    def __init__ (self, diffraction_info, set_info, image_clip_info
                  , ttheta_range = None
                  , omega_range  = None 
                  , phi_range    = None):
        self.diffraction_info = diffraction_info
        self.set_info = set_info
        self.image_clip_info = image_clip_info
        self.ttheta_range = ttheta_range
        self.omega_range  = omega_range
        self.phi_range    = phi_range

    def phi_refer_calculator(self, phi_base = 0, BSP = 0, phi_range = None):
        camera_L, _, _, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        phi_refer_deg = []
        phi_refer_deg = [phi_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(Y_start,Y_start+Y_range)] 
        if phi_range == None:
            phi_refer_deg = phi_refer_deg
        else:
            phi_refer_deg = phi_refer_deg[phi_range[0]:(phi_range[1] + 1)]
        return phi_refer_deg
    
    def ttheta_calculator(self,  tthe_range = None):
        camera_L, ttheta_base, BSP, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        ttheta_deg = []
        ttheta_deg = [ttheta_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(X_start,X_start+X_range)] 
        if tthe_range == None:
            ttheta_deg = ttheta_deg
        else:
            ttheta_deg = ttheta_deg[tthe_range[0]:(tthe_range[1] + 1)]

        return ttheta_deg


    def omega_calculator(self, if_need_first_omega_flag = False, omega_range = None):
        omega_s, omega_e, omega_step = self.set_info
        omega_deg = []
        omega_deg = np.arange(omega_s, omega_e + omega_step/2, omega_step)
        if if_need_first_omega_flag == False: # false means we do not need the first omega
            omega_deg = omega_deg[1:]

        if omega_range == None:
            omega_deg = omega_deg
        else:
            omega_deg = omega_deg[omega_range[0]:(omega_range[1] + 1)]

        return omega_deg

    def plot(self, column_flag = ['omega', 'phi'],
             Normalize_flag = True, save_flag = False, save_path = None,
                gridplot_number = 500, log_flag = True, if_need_first_omega_flag = False
                , Threshold_flag_ratio = False, Threshold_ratio = 20
                , Threshold_flag_value = False, Threshold_value = 0.1
                , angular_array = None
                , Label_mode = False, n_c = 2
                , plt_text_dict = None
                , x_guideline_flag = False, X_0 = 0
                , y_guideline_flag = False, Y_0 = 0
                , guideline_color = 'white'):

        # input the array with the 2D angular positioned Intensity
        # and transfer raw data into gridplot
        ref_deg_dict = {
            'omega':self.omega_calculator(if_need_first_omega_flag = if_need_first_omega_flag, omega_range=self.omega_range),
            '2theta':self.ttheta_calculator(tthe_range=self.ttheta_range),
            'phi':self.phi_refer_calculator(phi_range=self.phi_range)
        }

        rows_name, columns_name = column_flag
        
        print(rows_name, columns_name)
        
        rows_raw = ref_deg_dict[rows_name]
        columns_raw = ref_deg_dict[columns_name]
        
        print(np.shape(rows_raw), np.shape(columns_raw))

        # transforms the columns to Xs, rows to Ys
        Angular_2_raw, Angular_1_raw = np.meshgrid(columns_raw, rows_raw) 

        angular_1 = Angular_1_raw.ravel()
        angular_2 = Angular_2_raw.ravel()
        Intensity_map = angular_array.flatten()

        print(Angular_1_raw.shape, Angular_2_raw.shape, angular_array.shape)
    
        # calculate the gridplot [QXi, QZi] from the raw [qx, qz]
        angular_1i = np.linspace(angular_1.min(), angular_1.max(), gridplot_number)
        angular_2i = np.linspace(angular_2.min(), angular_2.max(), gridplot_number)
        A_1i, A_2i = np.meshgrid(angular_1i, angular_2i)

        ## calculate corresponing Ii based on the gridplot [QXi, QZi]
        Ii = griddata(points = (angular_1, angular_2)
                        , values = Intensity_map
                    ,xi = (A_1i, A_2i),  method = 'linear')

        Ii = np.nan_to_num(Ii)
        
        ## add threshold if nexessary, which is set as False for default
        if Threshold_flag_ratio == True:
            threshold = Ii.max()/Threshold_ratio ########## change

            binary_mask = Ii >= threshold

            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
            
        elif Threshold_flag_value == True:
            threshold = Threshold_value ########## change

            binary_mask = Ii >= threshold
            
            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
        
        ###########test#########
        # A_1i = np.array(Angular_1_raw)
        # A_2i = np.array(Angular_2_raw)
        # Ii = np.array(angular_array)
        ###########test#########

        ## add if normalize is necessary, which is set as True in default
        if Normalize_flag == True:
            Ii = (Ii - Ii.min())/ (Ii.max()- Ii.min()) + 1e-10 ## avoid the log10 problem
            # define the levels correspning to the 1e-10
            levels = np.arange(-10, 0.25, 0.125)
        elif Normalize_flag == False:
            Ii = Ii + 1
            levels = np.arange(0, 6.25, 0.125)

        ## map the results[QXi, QZi, Intensity]
        ## the color is based on log(Intensity)

        if Label_mode == True:
            cmap = cm.get_cmap('Set3', n_c)
        else:
            cmap = cm.get_cmap('gist_ncar', len(levels))

        fig,ax = plt.subplots(1,1, figsize = (10,10))
        ax = plt.gca()
        # levels_0 = np.linspace(Ii_log.min(),1,100)
        ax.tick_params(bottom=True,
                    left=True,
                    right=False,
                    top=False)

        if log_flag == True and column_flag == ['phi', 'omega']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.1)
            x_minor_locator = MultipleLocator(0.02)
            y_major_locator = MultipleLocator(0.06)
            y_minor_locator = MultipleLocator(0.012)
            # cbar.set_label("log(Intensity)")
            # plt.ylabel(r'$\mathit{\varphi}$ [degree]', rotation=270, labelpad=55)
            # plt.xlabel(r'$\mathit{\omega}$  [degree]')
        
        elif log_flag == True and column_flag == ['phi', '2theta']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.1)
            x_minor_locator = MultipleLocator(0.02)
            y_major_locator = MultipleLocator(0.15)
            y_minor_locator = MultipleLocator(0.03)
            # cbar.set_label("log(Intensity)")
            # plt.ylabel(r'$\mathit{\varphi}$ [degree]', rotation=270, labelpad=55)
            # plt.xlabel(r'2$\mathit{\theta}$ [degree]')
        
        elif log_flag == True and column_flag == ['2theta', 'omega']:
            Ii_log = np.log10(Ii)
            map_0 = plt.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.15)
            x_minor_locator = MultipleLocator(0.03)
            y_major_locator = MultipleLocator(0.05)
            y_minor_locator = MultipleLocator(0.01)
            # cbar.set_label("log(Intensity)")
            # plt.ylabel(r'2$\mathit{\theta}$ [degree]', rotation=270, labelpad=55)
            # plt.xlabel(r'$\mathit{\omega}$ [degree]')
            
            
        elif log_flag == False and Label_mode == False:
            map_0 = ax.scatter(A_1i, A_2i, c = Ii, cmap=cmap)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.001)
            y_minor_locator = MultipleLocator(0.0002)
            # cbar.set_label("Intensity")

        elif log_flag == True and Label_mode == True:
            Ii_log = np.log10(Ii)
            map_0 = ax.scatter(A_1i, A_2i, c = Ii_log, cmap=cmap)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.0001)
            y_minor_locator = MultipleLocator(0.00005)
            # cbar.set_label("label")

        elif log_flag == False and Label_mode == True:
            map_0 = ax.scatter(A_1i, A_2i, c = Ii, cmap=cmap)
            ## cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.0001)
            y_minor_locator = MultipleLocator(0.00005)
            # cbar.set_label("label")
        # set details of the plot
        
        plt.tick_params(axis='both', which='major', length=11)
        plt.tick_params(axis='both', which='minor', length=5)

        if Normalize_flag == True:    
            cbar_ticks = np.arange(-10, 1, 2)  # Ticks from -10 to 0 with step of 2
        elif Normalize_flag == False:
            cbar_ticks = np.arange(0, levels.max() + 1, 2)   

        # cbar = plt.colorbar(map_0)
        # cbar.set_ticks(cbar_ticks)
        
        # if log_flag == True:
        #     cbar.set_label(r'$\log_{10}(\mathrm{Intensity})$ [a.u.]',rotation=-90, labelpad=55)
        # elif log_flag == False:
        #     cbar.set_label("Intensity [counts]",rotation=-90, labelpad=55)

        ax.xaxis.set_major_locator(x_major_locator)
        ax.xaxis.set_minor_locator(x_minor_locator)


        ax.yaxis.set_major_locator(y_major_locator)
        ax.yaxis.set_minor_locator(y_minor_locator)

        # add a guideline as specific position
        if x_guideline_flag == True:
            ax.axvline(x=X_0,  color=guideline_color, linestyle='--',  linewidth=15, label=f'X = {X_0}')
        if y_guideline_flag == True:
            ax.axhline(y=Y_0,  color=guideline_color, linestyle='--',  linewidth=15, label=f'Y = {Y_0}')

        # plt_text_dict['plt_text'] = 't'
        
        print(plt_text_dict['plt_text'])
        
        # add rotation
        for tick in ax.get_yticklabels():
                tick.set_rotation(270)
        # for tick in cbar.ax.get_yticklabels():
        #         tick.set_rotation(270)
        # ax.set_ylabel(rotation=270, labelpad=15)

        ### add some texts in the figure
        if plt_text_dict != None:
            plt.figtext(plt_text_dict['x'], plt_text_dict['y'], plt_text_dict['plt_text'], ha="center"
                        # ,rotation=-90
                        , fontsize=48, bbox={"facecolor":"white", "alpha":1.0, "pad":5})
            
        # Adjust padding
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # plt.tight_layout()
        if save_flag == True:
            plt.savefig(save_path, dpi = 300, bbox_inches='tight')

        return fig

class angular_position_to_plot_v_4():
    
    """
    Compared to the _v, remove the title and colorbar information
    Compared to the _v_2, change the order of  x and y for input
    Compared to the _v_3, directly produce the plot on ax

    """


    def __init__ (self, diffraction_info, set_info, image_clip_info
                  , ttheta_range = None
                  , omega_range  = None 
                  , phi_range    = None):
        self.diffraction_info = diffraction_info
        self.set_info = set_info
        self.image_clip_info = image_clip_info
        self.ttheta_range = ttheta_range
        self.omega_range  = omega_range
        self.phi_range    = phi_range

    def phi_refer_calculator(self, phi_base = 0, BSP = 0, phi_range = None):
        camera_L, _, _, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        phi_refer_deg = []
        phi_refer_deg = [phi_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(Y_start,Y_start+Y_range)] 
        if phi_range == None:
            phi_refer_deg = phi_refer_deg
        else:
            phi_refer_deg = phi_refer_deg[phi_range[0]:(phi_range[1] + 1)]
        return phi_refer_deg
    
    def ttheta_calculator(self,  tthe_range = None):
        camera_L, ttheta_base, BSP, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        ttheta_deg = []
        ttheta_deg = [ttheta_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(X_start,X_start+X_range)] 
        if tthe_range == None:
            ttheta_deg = ttheta_deg
        else:
            ttheta_deg = ttheta_deg[tthe_range[0]:(tthe_range[1] + 1)]

        return ttheta_deg


    def omega_calculator(self, if_need_first_omega_flag = False, omega_range = None):
        omega_s, omega_e, omega_step = self.set_info
        omega_deg = []
        omega_deg = np.arange(omega_s, omega_e + omega_step/2, omega_step)
        if if_need_first_omega_flag == False: # false means we do not need the first omega
            omega_deg = omega_deg[1:]

        if omega_range == None:
            omega_deg = omega_deg
        else:
            omega_deg = omega_deg[omega_range[0]:(omega_range[1] + 1)]

        return omega_deg

    def plot(self, column_flag = ['omega', 'phi'],
             Normalize_flag = True,
                gridplot_number = 500, log_flag = True, if_need_first_omega_flag = False
                , Threshold_flag_ratio = False, Threshold_ratio = 20
                , Threshold_flag_value = False, Threshold_value = 0.1
                , angular_array = None
                , Label_mode = False, n_c = 2
                , plt_text_dict = None
                , x_guideline_flag = False, X_0 = 0
                , y_guideline_flag = False, Y_0 = 0
                , guideline_color = 'white'
                , ax = None):

        if ax == None:
            print("The Axes object is not created")
            return

        # input the array with the 2D angular positioned Intensity
        # and transfer raw data into gridplot
        ref_deg_dict = {
            'omega':self.omega_calculator(if_need_first_omega_flag = if_need_first_omega_flag, omega_range=self.omega_range),
            '2theta':self.ttheta_calculator(tthe_range=self.ttheta_range),
            'phi':self.phi_refer_calculator(phi_range=self.phi_range)
        }

        rows_name, columns_name = column_flag
        
        print(rows_name, columns_name)
        
        rows_raw = ref_deg_dict[rows_name]
        columns_raw = ref_deg_dict[columns_name]
        
        print(np.shape(rows_raw), np.shape(columns_raw))

        # transforms the columns to Xs, rows to Ys
        Angular_2_raw, Angular_1_raw = np.meshgrid(columns_raw, rows_raw) 

        angular_1 = Angular_1_raw.ravel()
        angular_2 = Angular_2_raw.ravel()
        Intensity_map = angular_array.flatten()

        print(Angular_1_raw.shape, Angular_2_raw.shape, angular_array.shape)
    
        # calculate the gridplot [QXi, QZi] from the raw [qx, qz]
        angular_1i = np.linspace(angular_1.min(), angular_1.max(), gridplot_number)
        angular_2i = np.linspace(angular_2.min(), angular_2.max(), gridplot_number)
        A_1i, A_2i = np.meshgrid(angular_1i, angular_2i)

        ## calculate corresponing Ii based on the gridplot [QXi, QZi]
        Ii = griddata(points = (angular_1, angular_2)
                        , values = Intensity_map
                    ,xi = (A_1i, A_2i),  method = 'linear')

        Ii = np.nan_to_num(Ii)
        
        ## add threshold if nexessary, which is set as False for default
        if Threshold_flag_ratio == True:
            threshold = Ii.max()/Threshold_ratio ########## change

            binary_mask = Ii >= threshold

            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
            
        elif Threshold_flag_value == True:
            threshold = Threshold_value ########## change

            binary_mask = Ii >= threshold
            
            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
        
        ###########test#########
        # A_1i = np.array(Angular_1_raw)
        # A_2i = np.array(Angular_2_raw)
        # Ii = np.array(angular_array)
        ###########test#########

        ## add if normalize is necessary, which is set as True in default
        if Normalize_flag == True:
            Ii = (Ii - Ii.min())/ (Ii.max()- Ii.min()) + 1e-10 ## avoid the log10 problem
            # define the levels correspning to the 1e-10
            levels = np.arange(-10, 0.25, 0.125)
        elif Normalize_flag == False:
            Ii = Ii + 1
            levels = np.arange(0, 6.25, 0.125)

        ## map the results[QXi, QZi, Intensity]
        ## the color is based on log(Intensity)

        if Label_mode == True:
            cmap = cm.get_cmap('Set3', n_c)
        else:
            cmap = cm.get_cmap('gist_ncar', len(levels))

        # fig,ax = plt.subplots(1,1, figsize = (10,10))
        # ax = plt.gca()
        # levels_0 = np.linspace(Ii_log.min(),1,100)
        # ax.tick_params(bottom=True,
        #             left=True,
        #             right=False,
        #             top=False)

        if log_flag == True and column_flag == ['phi', 'omega']:
            Ii_log = np.log10(Ii)
            map_0 = ax.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.1)
            x_minor_locator = MultipleLocator(0.02)
            y_major_locator = MultipleLocator(0.06)
            y_minor_locator = MultipleLocator(0.012)
            # cbar.set_label("log(Intensity)")
            # plt.ylabel(r'$\mathit{\varphi}$ [degree]', rotation=270, labelpad=55)
            # plt.xlabel(r'$\mathit{\omega}$  [degree]')
        
        elif log_flag == True and column_flag == ['phi', '2theta']:
            Ii_log = np.log10(Ii)
            map_0 = ax.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.1)
            x_minor_locator = MultipleLocator(0.02)
            y_major_locator = MultipleLocator(0.15)
            y_minor_locator = MultipleLocator(0.03)
            # cbar.set_label("log(Intensity)")
            # plt.ylabel(r'$\mathit{\varphi}$ [degree]', rotation=270, labelpad=55)
            # plt.xlabel(r'2$\mathit{\theta}$ [degree]')
        
        elif log_flag == True and column_flag == ['2theta', 'omega']:
            Ii_log = np.log10(Ii)
            map_0 = ax.contourf(A_1i, A_2i, Ii_log, cmap=cmap, levels = levels)
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.15)
            x_minor_locator = MultipleLocator(0.03)
            y_major_locator = MultipleLocator(0.05)
            y_minor_locator = MultipleLocator(0.01)
            # cbar.set_label("log(Intensity)")
            # plt.ylabel(r'2$\mathit{\theta}$ [degree]', rotation=270, labelpad=55)
            # plt.xlabel(r'$\mathit{\omega}$ [degree]')
            
            
        elif log_flag == False and Label_mode == False:
            map_0 = ax.scatter(A_1i, A_2i, c = Ii, cmap=cmap)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.001)
            y_minor_locator = MultipleLocator(0.0002)
            # cbar.set_label("Intensity")

        elif log_flag == True and Label_mode == True:
            Ii_log = np.log10(Ii)
            map_0 = ax.scatter(A_1i, A_2i, c = Ii_log, cmap=cmap)
            # cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.0001)
            y_minor_locator = MultipleLocator(0.00005)
            # cbar.set_label("label")

        elif log_flag == False and Label_mode == True:
            map_0 = ax.scatter(A_1i, A_2i, c = Ii, cmap=cmap)
            ## cbar = fig.colorbar(map_0, ax = ax)
            y_major_locator = MultipleLocator(0.0001)
            y_minor_locator = MultipleLocator(0.00005)
            # cbar.set_label("label")
        # set details of the plot
        
        ax.tick_params(axis='both', which='major', length=11)
        ax.tick_params(axis='both', which='minor', length=5)

        if Normalize_flag == True:    
            cbar_ticks = np.arange(-10, 1, 2)  # Ticks from -10 to 0 with step of 2
        elif Normalize_flag == False:
            cbar_ticks = np.arange(0, levels.max() + 1, 2)   

        # cbar = plt.colorbar(map_0)
        # cbar.set_ticks(cbar_ticks)
        
        # if log_flag == True:
        #     cbar.set_label(r'$\log_{10}(\mathrm{Intensity})$ [a.u.]',rotation=-90, labelpad=55)
        # elif log_flag == False:
        #     cbar.set_label("Intensity [counts]",rotation=-90, labelpad=55)

        ax.xaxis.set_major_locator(x_major_locator)
        ax.xaxis.set_minor_locator(x_minor_locator)


        ax.yaxis.set_major_locator(y_major_locator)
        ax.yaxis.set_minor_locator(y_minor_locator)

        # add a guideline as specific position
        if x_guideline_flag == True:
            ax.axvline(x=X_0,  color=guideline_color, linestyle='--',  linewidth=15, label=f'X = {X_0}')
        if y_guideline_flag == True:
            ax.axhline(y=Y_0,  color=guideline_color, linestyle='--',  linewidth=15, label=f'Y = {Y_0}')

        # plt_text_dict['plt_text'] = 't'
        
        print(plt_text_dict['plt_text'])
        
        # add rotation
        for tick in ax.get_yticklabels():
                tick.set_rotation(270)
        # for tick in cbar.ax.get_yticklabels():
        #         tick.set_rotation(270)
        # ax.set_ylabel(rotation=270, labelpad=15)

        ### add some texts in the figure
        if plt_text_dict != None:
            ax.text(plt_text_dict['x'], plt_text_dict['y'], plt_text_dict['plt_text'], ha="center"
                        , transform=ax.transAxes# ,rotation=-90
                        , fontsize=48, bbox={"facecolor":"white", "alpha":1.0, "pad":5})
            
        # Adjust padding
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # plt.tight_layout()
        # if save_flag == True:
        #     plt.savefig(save_path, dpi = 300, bbox_inches='tight')

        return

    
class angular_position_to_1D_plot():
    """
    Compared to other function, the target is to 
    produce the 1d intensity profile of desired positions

    """


    def __init__ (self, diffraction_info, set_info, image_clip_info
                  , ttheta_range = None
                  , omega_range  = None 
                  , phi_range    = None
                  , angular_type = 'phi'):
        self.diffraction_info = diffraction_info
        self.set_info         = set_info
        self.image_clip_info  = image_clip_info
        self.ttheta_range = ttheta_range
        self.omega_range  = omega_range
        self.phi_range    = phi_range
        self.angular_type = angular_type

    def phi_refer_calculator(self, phi_base = 0, BSP = 0, phi_range = None):
        camera_L, _, _, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        phi_refer_deg = []
        phi_refer_deg = [phi_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(Y_start,Y_start+Y_range)] 
        if phi_range == None:
            phi_refer_deg = phi_refer_deg
        else:
            phi_refer_deg = phi_refer_deg[phi_range[0]:(phi_range[1] + 1)]
        return phi_refer_deg
    
    def ttheta_calculator(self,  tthe_range = None):
        camera_L, ttheta_base, BSP, wl = self.diffraction_info
        X_start, Y_start, X_range, Y_range = self.image_clip_info
        ttheta_deg = []
        ttheta_deg = [ttheta_base + (i-BSP)*100*(10 ** -3)\
            /camera_L*180/np.pi for i in range(X_start,X_start+X_range)] 
        if tthe_range == None:
            ttheta_deg = ttheta_deg
        else:
            ttheta_deg = ttheta_deg[tthe_range[0]:(tthe_range[1] + 1)]

        return ttheta_deg


    def omega_calculator(self, if_need_first_omega_flag = False, omega_range = None):
        omega_s, omega_e, omega_step = self.set_info
        omega_deg = []
        omega_deg = np.arange(omega_s, omega_e + omega_step/2, omega_step)
        if if_need_first_omega_flag == False: # false means we do not need the first omega
            omega_deg = omega_deg[1:]

        if omega_range == None:
            omega_deg = omega_deg
        else:
            omega_deg = omega_deg[omega_range[0]:(omega_range[1] + 1)]

        return omega_deg
    def plot(self
         , Normalize_flag = True, save_flag = False, save_path = None
         , log_flag = True, if_need_first_omega_flag = False
         , Threshold_flag_ratio = False, Threshold_ratio = 20
         , Threshold_flag_value = False, Threshold_value = 0.1
         , angular_array = None
         , plt_text_dict = None):

        """
        angular_array: input the array with the 1D angular positioned Intensity

        """

        # prepare the x and y
        ref_deg_dict = {
            'omega':self.omega_calculator(if_need_first_omega_flag = if_need_first_omega_flag, omega_range=self.omega_range),
            '2theta':self.ttheta_calculator(tthe_range=self.ttheta_range),
            'phi':self.phi_refer_calculator(phi_range=self.phi_range)
        }

        angular_type = self.angular_type
        x = ref_deg_dict[angular_type]
        Intensity_map = angular_array.flatten()
        Ii = np.nan_to_num(Intensity_map) # this is the y

        print(angular_type, len(x), Ii.shape)

        ## add threshold if nexessary, which is set as False for default
        if Threshold_flag_ratio == True:
            threshold = Ii.max()/Threshold_ratio ########## change

            binary_mask = Ii >= threshold

            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
            
        elif Threshold_flag_value == True:
            threshold = Threshold_value ########## change

            binary_mask = Ii >= threshold
            
            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)

        ## add if normalize is necessary, which is set as True in default
        if Normalize_flag == True:
            Ii = (Ii - Ii.min())/ (Ii.max()- Ii.min()) + 1e-10 ## avoid the log10 problem
            # define the levels correspning to the 1e-10
        elif Normalize_flag == False:
            Ii = Ii + 1

        fig, ax = plt.subplots(1,1, figsize=(30,7.5))
        ax = plt.gca()
        ax.tick_params(bottom=True,
                    left=True,
                    right=False,
                    top=False)

        if angular_type == 'phi':
            plt.scatter(x, Ii, c='black', marker='o')
            plt.plot(x, Ii, c='red', label=r'1D $\mathit{\varphi}$ profile')
            x_major_locator = MultipleLocator(0.1)
            x_minor_locator = MultipleLocator(0.02)

            plt.ylabel('Intensity [counts]')
            plt.xlabel(r'$\mathit{\varphi}$ [degree]')
        
        elif angular_type == '2theta':
            plt.scatter(x, Ii, c='black', marker='o')
            plt.plot(x, Ii, c='red', label=r'1D 2$\mathit{\theta}$ profile')
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.15)
            x_minor_locator = MultipleLocator(0.03)

            plt.ylabel('Intensity [counts]')
            plt.xlabel(r'2$\mathit{\theta}$ [degree]')
        
        elif angular_type == 'omega':
            plt.scatter(x, Ii, c='black', marker='o')
            plt.plot(x, Ii, c='red', label=r'1D $\mathit{\omega}$ profile')
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.05)
            x_minor_locator = MultipleLocator(0.01)

            plt.ylabel('Intensity [counts]')
            plt.xlabel(r'$\mathit{\omega}$ [degree]')

        if log_flag == True:    
            # set the y-axis
            plt.yscale('log')
            # Calculate orders of magnitude for y.min() and y.max()
            min_order = np.floor(np.log10(Ii.min()))
            max_order = np.ceil(np.log10(Ii.max()))

            # Generate y-ticks from y.min() to y.max()
            yticks = [10 ** i for i in range(int(min_order), int(max_order))]
            plt.yticks(yticks)

            # set the length of ticks
            plt.tick_params(axis='both', which='major', length=11)
            plt.tick_params(axis='both', which='minor', length=5)


        ax.xaxis.set_major_locator(x_major_locator)
        ax.xaxis.set_minor_locator(x_minor_locator)
        
        print(plt_text_dict['plt_text'])
        
        # add rotation
        # for tick in ax.get_yticklabels():
        #         tick.set_rotation(270)
        # for tick in cbar.ax.get_yticklabels():
        #         tick.set_rotation(270)
        # ax.set_ylabel(rotation=270, labelpad=15)

        ### add some texts in the figure
        if plt_text_dict != None:
            plt.figtext(plt_text_dict['x'], plt_text_dict['y'], plt_text_dict['plt_text'], ha="center"
                        # ,rotation=-90
                        , fontsize=48, bbox={"facecolor":"white", "alpha":1.0, "pad":5})
            
        # Adjust padding
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # Add background
        plt.grid(linestyle='--', linewidth=0.5, alpha = 0.5)
        plt.legend()
        # plt.tight_layout()
        if save_flag == True:
            plt.savefig(save_path, bbox_inches='tight')# , dpi = 300)

        else:
                return fig
        

    def plot_reverse(self
         , Normalize_flag = True, save_flag = False, save_path = None
         , log_flag = True, if_need_first_omega_flag = False
         , Threshold_flag_ratio = False, Threshold_ratio = 20
         , Threshold_flag_value = False, Threshold_value = 0.1
         , angular_array = None
         , plt_text_dict = None):

        """
        angular_array: input the array with the 1D angular positioned Intensity

        Note: 20240731 plot_reverse() is same as plot() but change the x cordinate into reverse direction

        """

        # prepare the x and y
        ref_deg_dict = {
            'omega':self.omega_calculator(if_need_first_omega_flag = if_need_first_omega_flag, omega_range=self.omega_range),
            '2theta':self.ttheta_calculator(tthe_range=self.ttheta_range),
            'phi':self.phi_refer_calculator(phi_range=self.phi_range)
        }

        angular_type = self.angular_type
        x = ref_deg_dict[angular_type]
        Intensity_map = angular_array.flatten()
        Ii = np.nan_to_num(Intensity_map) # this is the y

        print(angular_type, len(x), Ii.shape)

        ## add threshold if nexessary, which is set as False for default
        if Threshold_flag_ratio == True:
            threshold = Ii.max()/Threshold_ratio ########## change

            binary_mask = Ii >= threshold

            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)
            
        elif Threshold_flag_value == True:
            threshold = Threshold_value ########## change

            binary_mask = Ii >= threshold
            
            Ii = ma.array(Ii, mask= ~binary_mask)
            Ii = Ii.filled(0)

        ## add if normalize is necessary, which is set as True in default
        if Normalize_flag == True:
            Ii = (Ii - Ii.min())/ (Ii.max()- Ii.min()) + 1e-10 ## avoid the log10 problem
            # define the levels correspning to the 1e-10
        elif Normalize_flag == False:
            Ii = Ii + 1

        fig, ax = plt.subplots(1,1, figsize=(30,7.5))
        ax = plt.gca()
        ax.tick_params(bottom=True,
                    left=True,
                    right=False,
                    top=False)

        if angular_type == 'phi':
            plt.scatter(x, Ii, c='black', marker='o')
            plt.plot(x, Ii, c='red', label=r'1D $\mathit{\varphi}$ profile')
            x_major_locator = MultipleLocator(0.1)
            x_minor_locator = MultipleLocator(0.02)

            plt.ylabel('Intensity [counts]')
            plt.xlabel(r'$\mathit{\varphi}$ [degree]')
        
        elif angular_type == '2theta':
            plt.scatter(x, Ii, c='black', marker='o')
            plt.plot(x, Ii, c='red', label=r'1D 2$\mathit{\theta}$ profile')
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.15)
            x_minor_locator = MultipleLocator(0.03)

            plt.ylabel('Intensity [counts]')
            plt.xlabel(r'2$\mathit{\theta}$ [degree]')
        
        elif angular_type == 'omega':
            plt.scatter(x, Ii, c='black', marker='o')
            plt.plot(x, Ii, c='red', label=r'1D $\mathit{\omega}$ profile')
            # cbar = fig.colorbar(map_0, ax = ax)
            x_major_locator = MultipleLocator(0.05)
            x_minor_locator = MultipleLocator(0.01)

            plt.ylabel('Intensity [counts]')
            plt.xlabel(r'$\mathit{\omega}$ [degree]')

        if log_flag == True:    
            # set the y-axis
            plt.yscale('log')
            # Calculate orders of magnitude for y.min() and y.max()
            min_order = np.floor(np.log10(Ii.min()))
            max_order = np.ceil(np.log10(Ii.max()))

            # Generate y-ticks from y.min() to y.max()
            yticks = [10 ** i for i in range(int(min_order), int(max_order))]
            plt.yticks(yticks)

            # set the length of ticks
            plt.tick_params(axis='both', which='major', length=11)
            plt.tick_params(axis='both', which='minor', length=5)


        ax.xaxis.set_major_locator(x_major_locator)
        ax.xaxis.set_minor_locator(x_minor_locator)

        # change the direction into 
        ax.invert_xaxis()
        
        print(plt_text_dict['plt_text'])
        
        # add rotation
        # for tick in ax.get_yticklabels():
        #         tick.set_rotation(270)
        # for tick in cbar.ax.get_yticklabels():
        #         tick.set_rotation(270)
        # ax.set_ylabel(rotation=270, labelpad=15)

        ### add some texts in the figure
        if plt_text_dict != None:
            plt.figtext(plt_text_dict['x'], plt_text_dict['y'], plt_text_dict['plt_text'], ha="center"
                        # ,rotation=-90
                        , fontsize=48, bbox={"facecolor":"white", "alpha":1.0, "pad":5})
            
        # Adjust padding
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # Add background
        plt.grid(linestyle='--', linewidth=0.5, alpha = 0.5)
        plt.legend()
        # plt.tight_layout()
        if save_flag == True:
            plt.savefig(save_path, bbox_inches='tight')# , dpi = 300)


        return fig
    


    