#########################################################################################################################################
# This script  generate complex multi-layers subsurfaces, simulate data collection and inverse the later based  on conventional     #####
# physics approach using PyGIMLi for Dipole-dipole and Wenner-Schlumberger configurations. The data will be used to efficiently     #####
# train neural networks on DC-current resistivity in 2D without topography.                                                         #####
# The building function are in the script make_dcr_dataset.py , you may need to visit it for more details.                          #####
#                                                                                                                                   #####
# Author: Béranger OVONO EKORE                                                                                                      #####
#########################################################################################################################################

# --- the building functions and libraries ---
# Please make sure to set the parameters in the file make_dcr_dataset.py before running this script.
from make_dcr_dataset import *
 
# --- Generate multi-layers subsurfaces with and without anisotropy ---
create_data_multi_layers(left, right, depth, grid, nx, nz, config_files=config_files, config_names=config_names, ratio=2, nb_elec=96
                         , number_init=1000, inf_res=10, sup_res=1000, area=5, cmap='jet', noiseAbs=5e-5, seed=1, noise_levels=noise_levels
                         , rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names, rhoa_datFile_names=rhoa_datFile_names
                         , rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names, save_img=False, with_inv=True 
                         , inv_rho_file_names=inv_rho_file_names, paraDepth=depth, quality=32, paraMaxCellSize=10, limits=None, zWeight=None
                         , anisotropy=True, inject_fluid=True, cmin=None, cmax=None, fix_marker_area=0.1)
                        
