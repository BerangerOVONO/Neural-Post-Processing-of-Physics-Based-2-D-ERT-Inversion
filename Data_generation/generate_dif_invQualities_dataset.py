#########################################################################################################################################
# This script  generate geological subsurfaces, simulate data collection and inverse the later based on conventional physics        #####
# approach with different inversion qualities (regularisation norms, mesh sizes) to enable effective evaluation of the impact       #####
# of the pre-inversion on the performance of the neural network.                                                                    ##### 
# The building functions are in the script make_dif_invQualities_dataset.py , you may visit it for more details and settings.       #####   
#                                                                                                                                   #####
# Author: Béranger OVONO EKORE                                                                                                      #####
#########################################################################################################################################


# ## the building functions and libraries
# Please make sure to set the parameters in the file make_dif_invQualities_dataset.py before running this script.
from make_dif_invQualities_dataset import *
 
# ## Generate multi-layers subsurfaces with and without anisotropy, simulate data and inversion with different qualities
create_data_multi_layers_invQual(left, right, depth, grid, nx, nz, config_files=config_files, config_names=['slm'], ratio=3, nb_elec=96
                                 , number_init=1000, inf_res=10, sup_res=600, area=5, cmap='jet', noiseAbs=5e-5, seed=1, noise_levels=noise_levels
                                 , rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names, rhoa_datFile_names=rhoa_datFile_names
                                 , rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=False, with_inv=True 
                                 , inv_rho_file_names=inv_rho_file_names, paraDepth=depth, qualities=[32, 34], paraMaxCellSizes=[20, 5], limits=None
                                 , zWeight=None, anisotropy=True, inject_fluid=True, cmin=None, cmax=None, tol_lim=0.1, fix_marker_area=0.1, norm_reguls=[False, True])


# note that norm_reguls=[False, True] means [L2, L1] regularisations
# Please find others parameters details in the script make_dif_invQualities_dataset.py
         