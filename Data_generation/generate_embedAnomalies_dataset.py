#########################################################################################################################################
# This script  generate simple geological settings with embedded anomalies, simulate data collection and inverse the later based    #####
# on conventional physics approach using PyGIMLi for Dipole-dipole and Wenner-Schlumberger configurations. The data will be used    #####
# to efficiently train neural networks on DC-current resistivity in 2D without topography.                                          #####
# The building function are in the script make_dcr_dataset.py , you may need to visit it for more details.                          #####
#                                                                                                                                   #####
# Author: Béranger OVONO EKORE                                                                                                      #####
#########################################################################################################################################


# --- the building functions and libraries ---
# Please make sure to set the parameters in the file make_dcr_dataset.py before running this script.
from make_dcr_dataset import *
import faulthandler
faulthandler.enable()

# --- Generate homogeneous media with embed anomalies ---
anomaly_res=[10, 100, 500, 1000, 2000]
create_data_embed_anomaly_multi_size(left, right, depth, grid, nx, nz, config_files=config_files, config_names=['slm', 'dd']
                                     , nb_elecg=96, inf_res=10, sup_res=2000, area=10, anomaly_res=anomaly_res, anomaly_size=[6]
                                     , anomaly_pad=[3, 1], anomaly_stride=[12, 4], rho_file_names=rho_file_names, rho_file_mesh_names=rho_file_mesh_names
                                     , rhoa_datFile_names=rhoa_datFile_names, rhoa_pandaFile_names=rhoa_pandaFile_names, rhoa_imgFile_names=rhoa_imgFile_names
                                     , save_img=True, with_inv=True , inv_rho_file_names=inv_rho_file_names, paraDepth=depth, quality=32, paraMaxCellSize=10, limits=None
                                     , zWeight=None, seed=None, cmap='jet', cmin=None, cmax=None, ratio_multi_anomaly=3, fix_marker_area=0.1)


