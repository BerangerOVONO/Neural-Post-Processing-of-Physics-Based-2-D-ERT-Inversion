###################################################################################################################################################################
# In this script we will be enhancing direct current resistivity (dcr) 2D inversion result using deep convolutional neural networks based on Unet architecture.  ##
# Please check the file CNNs_module.py for model implementation details and The file post_inv_functions.py for dataset classes, transforms, train and            ##
# evaluation functions.                                                                                                                                          ##
#                                                                                                                                                                ##
# Author: Béranger OVONO EKORE                                                                                                                                   ##
###################################################################################################################################################################

# ## Libs, modul and functions
from CNNs_module import *
from post_inv_functions import *

# ## dataset path
directory = "/gpfs/projets/deepres/bdd_indexes"
#csv_file = 'index_multiLayers_paths_all.csv'
#csv_file = 'index_embedAnomaly_paths_all.csv'
csv_file = 'index_preInv_DCR_T0_WSnDD.csv'
#csv_file = 'index_multiLayers_InvQualities_WS_paths.csv'
csv_path = os.path.join(directory, csv_file)

## adding the directory path to the dataset name if they are not already there
#data_names = pd.read_csv(csv_path)
#data_directory = "/gpfs/projets/deepres/Data/preInv_DCR_T0_WSnDD_saphir"
#data_paths = add_directory_to_df_os(data_names, data_directory, slice=[0,1,2,3,4,5])
data_paths = pd.read_csv(csv_path)
data_paths = data_paths.reset_index()
#data_paths = data_paths.iloc[0:1000, :]
print("Total indexes in the dataset: ",data_paths.shape[0])
print(data_paths)
#data_paths = data_paths.reset_index()

# ## Initial dataset grid used to resize data images for training if necessary
# may match with the geological dimensions in your dataset
# geological dimensions (x_distance=from -left to right, and z_distance=from -depth to the surface 0), see bellow!
left = 50
right = 50
depth = 25
# Grid mesh dimension on it we will sample the image over the unstructured mesh
nz = 100 # number of vertical cells (pixels) in the grid (the image becoming)
nx = 401 #4*nz # number of horizontal cell (pixels) in the grid (the image becoming)
x, z = np.linspace(-left, right, nx), np.linspace(-depth, 0, nz) # the gris extend
grid = mt.createGrid(x, z, marker=1) # initial grid matching the images size in the dataset
# Grid mesh dimension on it we will downsample the image over the unstructured mesh if necessary
out_nz = 51 # number of vertical cells (pixels) in the grid (the image becoming)
out_nx = 201 #4*nz # number of horizontal cell (pixels) in the grid (the image becoming)
out_x, out_z = np.linspace(-left, right, out_nx), np.linspace(-depth, 0, out_nz) # the gris extend
outgrid = mt.createGrid(out_x, out_z, marker=1) # grid used for down sample the images in the dataset 

# seed the random functions for reproducibility and fair comparison
#set_seed(26)

# ## The transform objects
downSample_transform = downSampleIMG(grid, outgrid, out_nx, out_nz, is_tensor=True)
normalizedOverMaxRes = normRes(maxDatasetRes=2010.0, logScale=True)

# ## The dataset objets directly on the gpu device
pg.info("loading the dataset to the gpu device")
# Normalize and down sampling transforms or not on the inputs images
#downsampling_tensor_norm_transform = transforms.Compose([downSample_transform, normalizedOverMaxRes, transforms.Lambda(lambda x: x.unsqueeze(0))])
#resize = (out_nz-1, out_nx-1)  # resize to the downsampled size to avoid interpolation artifacts 
norm_transform = transforms.Compose([normalizedOverMaxRes, transforms.Lambda(lambda x: x.unsqueeze(0))])
#norm_transform = transforms.Compose([downSample_transform, normalizedOverMaxRes, transforms.Lambda(lambda x: x.unsqueeze(0))])
# for Wenner-Schlumberger
down_dataset = device_img_preInvData(data_paths, data_dir=None, transform_x=norm_transform, transform_y=norm_transform, decay_col=0, device=device)
# for Dipole-dipole
#down_dataset = device_img_preInvData(data_paths, data_dir=None, transform_x=norm_transform, transform_y=norm_transform, decay_col=2, device=device)
pg.info("downsimpled dataset[1][0] size: ", down_dataset[1][0].size())

# ## train, validation and tes size
N = len(down_dataset)
train_size = int(0.8 * N)
val_size = int(0.1 * N)
test_size = N - train_size - val_size
batch_size = 32

# ## save and load directory
saving_directory = "/gpfs/projets/deepres/Trained_PostInv_reloc_dir"
if not os.path.exists(saving_directory):
    os.makedirs(saving_directory)
os.chdir(saving_directory)

# ## Load dataset for pytorch
## load the indices for the train, val and test sets, if they are already created
#indices_dict = torch.load("train_val_test_split_Data_gpu_EA_b32_WSnDD_norm2k10.pth", weights_only=False)
#indices_dict = torch.load("train_val_test_split_Data_gpu_ML_b32_WSnDD_norm2k10.pth", weights_only=False)
#indices_dict = torch.load("train_val_test_split_Data_gpu_ALL_b32_WSnDD_norm2k10.pth", weights_only=False)
#indices_dict = torch.load("train_val_test_split_Data_gpu_ML_b32_WS_InvQual.pth", weights_only=False)
#train_indices, val_indices, test_indices = indices_dict['train_indices'], indices_dict['val_indices'], indices_dict['test_indices']
#train_set_down, val_set_down, test_set_down = Subset(down_dataset, train_indices), Subset(down_dataset, val_indices), Subset(down_dataset, test_indices)
## if the indices are not created, create them and save them for later reuse 
train_set_down, val_set_down, test_set_down = random_split(down_dataset, [train_size, val_size, test_size])
trainload = DataLoader(train_set_down, batch_size=batch_size, shuffle=False, pin_memory=False)#, num_workers=2)
valload = DataLoader(val_set_down, batch_size=batch_size, shuffle=False, pin_memory=False)#, num_workers=2)
testload = DataLoader(test_set_down, batch_size=batch_size, shuffle=False, pin_memory=False)#, num_workers=2)
# saving the split indices for later reuse
#save_id_file = "train_val_test_split_Data_gpu_EA_b32_WSnDD_norm2k10.pth"
#save_id_file = "train_val_test_split_Data_gpu_ML_b32_WSnDD_norm2k10.pth"
#save_id_file = "train_val_test_split_Data_gpu_ALL_b32_WSnDD_norm2k10.pth"
#save_id_file = "train_val_test_split_Data_gpu_ML_b32_WS_InvQual.pth"
#torch.save({'train_indices': train_set_down.indices, 'val_indices': val_set_down.indices, 'test_indices': test_set_down.indices}, save_id_file)
#print("Trian, valitation and test split indices for a later reuse have been saved in " + save_id_file)
print("Nomber of batches: ", len(trainload))


# ## Network and parameters
Net_inverse = UNetFlex(1, 64, 4, 2)
Net_inverse.to(device)
countNetParameters(Net_inverse, verbose=True)
#print(Net_inverse)
# The optimizer
lr=1e-5
optimizer = optim.Adam(Net_inverse.parameters(), lr=lr)
#optimizer = optim.SGD(Net_inverse.parameters(), lr=0.0001,momentum=0.9,weight_decay=1e-5)
# The cost function
criterion_train = nn.MSELoss()
lambda_tv = None
#criterion_train = MSE_n_TV_Loss(lambda_tv=lambda_tv)

# ## Initializing the neural network and saving directory
# save the initial parameters of the neural for the first run
is_first_run = True
save_init_params_NN = "init_parameters_UNetFlex_c64_nl4_b32.pth"

if is_first_run:
    # save the initial parameters of the neural for the first run
    torch.save(Net_inverse.state_dict(), save_init_params_NN)
    print("The initial parameters of the neural network have been saved in " + save_init_params_NN)
else:
    # load the initial parameters of the neural from the first run for a fair comparison
    Net_inverse.load_state_dict(torch.load(save_init_params_NN, weights_only=True))
    print("The initial parameters of the neural network have been loaded from " + save_init_params_NN)

# ## Train and validation
# saving model name
#save_model_name='model_postInv_c64_nl4_b32_ALL_DD_norm2010_log10_size200x50'
save_model_name='model_postInv_c64_nl4_b32_ALL_WS_norm2010_log10_size200x50'
if lambda_tv:
    save_model_name = save_model_name + '_TV_' + str(lambda_tv)
# loss plot title
#title='Comprehensive Dataset Unet on DD-Config with LogScale'
title='Comprehensive Dataset Unet on WS-Config with LogScale'
#title='Milti-Layered Dataset Unet on WS-Config with LogScale with TV Loss lambda= '+str(lambda_tv)

# train
train_n_validation(Net_inverse, trainload, valload, epochs=100, init_lr=lr, update_lr_epoch=[0.6], print_freq=400, n_batches_train=len(trainload)
                   , n_batches_val=len(valload), save_model_name=save_model_name, save_train_loss_name='loss_train'+save_model_name+'.txt'
                   , save_val_loss_name='loss_val'+save_model_name+'.txt', optimizer=optimizer, criterion_train=criterion_train, criterion_val=criterion_train
                   ,title=title, patience=10, counter=0, best_loss=float("inf"), tolerance=5e-5)

# ## rsme evalution table
# loading the best model (the last one saved) for evaluation. Note that the training has continued until early stopping criteria is met.
Net_inverse.load_state_dict(torch.load(save_model_name+'.pth', weights_only=True))
#table_eval = test_rsme(Net_inverse, testload, table_file='Evaluation_postInv_norm2K10_logScale.csv', dataset_type='Comprehensive', dataset_size=N
#                   , config='Dipole-dipole',  model='Unet_DD', drop_ex_row=None, lam=lambda_tv, device=None)
table_eval = test_rsme(Net_inverse, testload, table_file='Evaluation_postInv_norm2K10_logScale.csv', dataset_type='Comprehensive', dataset_size=N
                   , config='Wenner-Schlumberger',  model='Unet_WS', drop_ex_row=None, lam=lambda_tv, device=None)
