######################################################################################################################
# This is the data class, train, validaation and evaluation functions used to  enhance 2D DCR inversion result      ## 
# using deep convolutional networks.                                                                                ## 
#                                                                                                                   ## 
# Author: Béranger OVONO EKORE                                                                                      ## 
######################################################################################################################

# ### Libs
import torch
#import torchvision
from torchvision.transforms import transforms
#import torchvision.transforms.functional as TF
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
#from pygimli.viewer import showMesh
import random
import pandas as pd
import os


# ## Set the GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pg.info("divice: ", device)

# ## Adding your directory path to the dataset functions
def add_directory_to_df_os(data_names, directory, slice, verbose=True):
    """ add the directory to the dataset file names to access to data using os """
    data_paths = data_names.copy()
    for c in slice:
        current_col = data_paths.iloc[:, c]
        data_paths.iloc[:, c] = list(map(lambda file: os.path.join( directory, file), current_col))
    # adding the indexes
    data_paths = data_paths.reset_index()
    if verbose:
        pg.info("The directory path has been add to the dataset names.")
    return data_paths

# ### The dataset class
# ## Differents classes of dataset depending on your device capacity :
# - The class device_img_preInvData might be the efficient choice if you can store all the dataset on your gpu device. 
#   In this case please state the option device=your gpu device. (The training will be fast as all the dataset will be
#   load on the gpu at once).
# - If not you, you probably may use device_preInvData leaving the option device=None (default device=cpu). Here you the
#   batches can be sent to the gpu device one at the time in your training function. (but the training will also take more time)
# - The class img_info_preInvData is similar to the previous one but here the data are loaded one at the time
#   when requested in the __getitem__ function. This class might be useful if you have a large dataset that can't fit in your cpu
#   memory. (We recommand to use it after training when you just want to load some data for evaluation or visualize predictions results)
# - And the last class all_info_preInvData is the most expensive. This class might just be used for visualization. 
#   The training will be slow with unnecessary loading if you use this class.
class device_img_preInvData(Dataset):
    """ Preload all the dataset in the gpu to speed up training """
    def __init__(self, df_dataPaths, data_dir=None, transform_x=None, transform_y=None, decay_col=0, device=None):
        self.data_dir = data_dir
        self.df_dataPaths = df_dataPaths
        if data_dir:
            self.df_dataPaths = add_directory_to_df_os(df_dataPaths, data_dir)
            self.df_dataPaths = self.df_dataPaths.reset_index()
        self.len = self.df_dataPaths.shape[0]

        self.info_linked_id = torch.from_numpy(self.df_dataPaths.iloc[:, 0].values) # always stay on cpu
        self.x_images = torch.from_numpy(np.array(list(map(pd.read_csv, self.df_dataPaths.iloc[:, 1]))))
        self.y_images = torch.from_numpy(np.array(list(map(pd.read_csv, self.df_dataPaths.iloc[:, decay_col+4]))))
        
        self.transform_x = transform_x
        self.transform_y = transform_y
        if self.transform_x :
            self.x_images = torch.stack(list(map(self.transform_x, self.x_images))).to(dtype=torch.float)
        if self.transform_y :
            self.y_images = torch.stack(list(map(self.transform_y, self.y_images))).to(dtype=torch.float)
        
        self.device = device
        if self.device:
            self.x_images = self.x_images.to(self.device, non_blocking=True)
            self.y_images = self.y_images.to(self.device, non_blocking=True)
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x_image, y_image, info_linked_id = self.x_images[index], self.y_images[index], self.info_linked_id[index]
        return x_image, y_image, info_linked_id

   
class img_info_preInvData(Dataset):
    """ Single dataset row loading """
    def __init__(self, df_dataPaths, data_dir=None, transform_x=None, transform_y=None, decay_col=0):
        self.decay_col = decay_col
        self.data_dir = data_dir
        self.df_dataPaths = df_dataPaths
        if data_dir:
            self.df_dataPaths = add_directory_to_df_os(df_dataPaths, data_dir)
            self.df_dataPaths = self.df_dataPaths.reset_index()
        self.len = self.df_dataPaths.shape[0]
        self.info_linked_id = torch.from_numpy(self.df_dataPaths.iloc[:, 0].values) # always stay on cpu
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x_image = torch.from_numpy(np.loadtxt(self.df_dataPaths.iloc[index, 1], delimiter=','))#pd.read_csv(self.df_dataPaths.iloc[index, 1])
        y_image = torch.from_numpy(np.loadtxt(self.df_dataPaths.iloc[index, self.decay_col+4], delimiter=','))#pd.read_csv(self.df_dataPaths.iloc[index, 4])
        info_linked_id = self.info_linked_id[index] 
        
        if self.transform_x:
            x_image = self.transform_x(x_image)
        x_image = x_image.to(dtype=torch.float)
        if self.transform_y:
            y_image = self.transform_y(y_image)
        y_image = y_image.to(dtype=torch.float)
        return x_image, y_image, info_linked_id

    
class all_info_preInvData(Dataset):
    """ Single dataset row loading """
    def __init__(self, df_dataPaths, data_dir=None, transform_x=None, transform_y=None):
        self.data_dir = data_dir
        self.df_dataPaths = df_dataPaths
        if data_dir:
            self.df_dataPaths = add_directory_to_df_os(df_dataPaths, data_dir)
            self.df_dataPaths = self.df_dataPaths.reset_index()
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.len = self.df_dataPaths.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # True resistivity map
        x_image_path = self.df_dataPaths.iloc[index, 1]
        x_image = np.loadtxt(x_image_path, delimiter=',') 
        # Tri-mesh used for the simulation WS
        x_mesh_path = self.df_dataPaths.iloc[index, 2]
        x_mesh = pg.load(x_mesh_path)
        
        # data simulated from WS
        data_simu_path_ws = self.df_dataPaths.iloc[index, 3]
        data_simu_ws = ert.load(data_simu_path_ws, verbose=False)
        # pre inversion map for WS
        y_image_path_ws = self.df_dataPaths.iloc[index, 4]
        y_image_ws = np.loadtxt(y_image_path_ws, delimiter=',') 
        # inversion quality for WS: rmse 
        rmse_ws = self.df_dataPaths.iloc[index, 7]

        # data simulated from WS
        data_simu_path_dd = self.df_dataPaths.iloc[index, 5]
        data_simu_dd = ert.load(data_simu_path_dd, verbose=False)
        # pre inversion map for WS
        y_image_path_dd = self.df_dataPaths.iloc[index, 6]
        y_image_dd = np.loadtxt(y_image_path_dd, delimiter=',') 
        # inversion quality for WS: rmse 
        rmse_dd = self.df_dataPaths.iloc[index, 8]
        
        
        if self.transform_x :
            x_image = self.transform_x(x_image)
        if self.transform_y :
            y_image_ws = self.transform_y(y_image_ws)
            y_image_dd = self.transform_y(y_image_dd)
        
        return x_image, y_image_ws, y_image_dd, x_mesh, data_simu_ws, data_simu_dd, rmse_ws, rmse_dd

        
# ## Dawn sampling the image in order to accerate the training by reducing the image dimensions if necessary 
# This transform might be useful if you have a large image and want to reduce its size for the training and it is accerate
# but this  will also degradate a little bite the image resolution and some spatial informations (edges, shape and interface).
class downSampleIMG(object):
    """ Efficient resistivities image resizing without introduction new values by interpolation for numpy ndarray and tensor"""
    def __init__(self, inGrid=None, outGrid=None, out_nx=None, out_nz=None, nan=999999.9, is_tensor=False):
        self.inGrid = inGrid
        self.outGrid = outGrid
        self.out_nx = out_nx
        self.out_nz = out_nz
        self.nan = nan
        self.is_tensor = is_tensor
        
    def __call__(self, inIMG):
        indata = inIMG.flatten().tolist()
        outdata = []
        for pos in (self.outGrid).cellCenters():
            cell = (self.inGrid).findCell(pos)
            if cell:
                outdata.append(indata[cell.id()])
            else:
                outdata.append(self.nan)
        outIMG = np.array(outdata).reshape(self.out_nz-1, self.out_nx-1)
        if self.is_tensor:
            outIMG =torch.from_numpy(outIMG)
        return outIMG

# ## Normalized all the resistivities maps over the max resistivity in all the dataset
# the log scale might be usefull to reduce the resistivity range, enforce posistivity and highlight resistivity contrasts
# this will help the training and avoid the loss explosion by confinning the data in [0, 1]
class normRes(object):
    """ Normalize resistivities image by the max resistivity overall the dataset """
    def __init__(self, maxDatasetRes=1000.0, logScale=False, regulLog=0.0):
        self.logScale = logScale
        self.regulLog = regulLog  # to avoid log(0)
        if self.logScale:
            maxDatasetRes = np.log10(maxDatasetRes)
        self.maxDaatsetRes = torch.tensor([maxDatasetRes], dtype=torch.float32)

    def __call__(self, inIMG):
        if self.logScale:
            outIMG = torch.log10(inIMG+self.regulLog)/abs(self.maxDaatsetRes)
        else:
            outIMG = inIMG/self.maxDaatsetRes
        return outIMG 


## loss saving plot functions
def show_save_loss(train_los, val_loss, title, save_name):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(train_los, label="train")
    ax.plot(val_loss, label="validation")
    ax.set_ylim([0, max(max(train_los), max(val_loss))*0.5])
    #ax.set_ylim([0, 1])
    ax.set_xlabel("Epoch")
    ax.legend(loc='best')
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.minorticks_on()
    ax.grid(True, which='minor', axis='both', color='black', linestyle=':', linewidth=0.4)
    fig.savefig(save_name + '.png', bbox_inches='tight', pad_inches=0, transparent=False, dpi=300, facecolor='white')
    plt.close()

## seed the ramdom functions for reproducibility and fair comparison
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

## MSE loss + total variation constraint
class MSE_n_TV_Loss(nn.Module):
    def __init__(self, lambda_tv=1e-4):
        super().__init__()
        self.lambda_tv = lambda_tv
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_tv = self.total_variation(pred)
        return loss_mse + self.lambda_tv * loss_tv

    def total_variation(self, x):
        diff_h = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        diff_w = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        return torch.mean(diff_h) + torch.mean(diff_w)

# ## Train and validation
# the train function
def train_n_validation(Net, trainload, valload, epochs, init_lr, update_lr_epoch=[0.5, 0.9], print_freq=50, n_batches_train=1, n_batches_val=1
                       , save_model_name='model', save_train_loss_name='loss_train.txt', save_val_loss_name='loss_val.txt', optimizer=None
                       , criterion_train=None, criterion_val=None, title='Embed Anomaly Dataset', patience=10, counter=0, best_loss=float("inf"), tolerance=5e-5):
    Loss_train, Loss_val = [], []
    #pg.info('Strating train and validation')
    print('Strating train and validation')
    for epoch in range(epochs):
        ## update optimiser learning rate
        if update_lr_epoch!=[]:
            if epoch==int(update_lr_epoch[0]*epochs):
                update_lr_epoch.pop(0)
                init_lr/=2
                patience//=2
                for param_group in optimizer.param_groups:
                    param_group['lr']=init_lr
        ## train
        Net.train()
        running_loss = 0.0
        for i, (y, yhat, info_linked_id) in enumerate(trainload):
            #y, yhat = y.to(device, non_blocking=True), yhat.to(device, non_blocking=True) # use this if you can't store the all dataset in the gpu device
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward , backward, optimize
            outputs = Net(yhat)
            loss = criterion_train(outputs, y)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item() 
            if i % print_freq == 0:    # print every print_freq mini-batches
                print('[%d, %5d] train not normalized loss: %.4f' %
                      (epoch + 1, i + 1, running_loss))
        #running_loss/=n_batches_train
        Loss_train.append(running_loss/n_batches_train)
        pg.info('[%d] epoch train loss: %.5f' % (epoch + 1, running_loss))
        #print('[%d] epoch train loss: %.3f' % (epoch + 1, running_loss))
        ## validation
        running_val_loss = 0.0
        Net.eval()
        with torch.no_grad():
            for y, yhat, info_linked_id in valload:
                #y, yhat = y.to(device, non_blocking=True), yhat.to(device, non_blocking=True) # use this if you can't store the all dataset in the gpu device
                y_pred = Net(yhat)
                loss = criterion_val(y_pred, y)
                running_val_loss += loss.item()
            #running_val_loss/=n_batches_val
            Loss_val.append(running_val_loss/n_batches_val)
        pg.info('[%d] epoch val loss: %.5f' % (epoch + 1, running_val_loss))
        #print('[%d] epoch val loss: %.3f' % (epoch + 1, running_loss))
        ## saving and early stopping
        if running_val_loss <= (best_loss-tolerance):
            best_loss, counter = running_val_loss, 0
            # save the best model
            torch.save(Net.state_dict(), save_model_name + '.pth')
            np.savetxt(save_train_loss_name,np.array(Loss_train))
            np.savetxt(save_val_loss_name,np.array(Loss_val))
            #pg.info("Saved PyTorch Model State to " + save_model_name + ".pth")
            print("Saved PyTorch Model State to " + save_model_name + ".pth")
            show_save_loss(Loss_train, Loss_val, title=title, save_name='Loss_evol_on_'+save_model_name)
        else:
            counter += 1
            # don't save the model as it isn't the best
            #np.savetxt(save_train_loss_name,np.array(Loss_train))
            #np.savetxt(save_val_loss_name,np.array(Loss_val))
            #show_save_loss(Loss_train, Loss_val, title=title, save_name='Loss_evol_on_'+save_model_name)
            if counter >= patience:
                print("Early stopping")
                pg.info('Training break cause: early stopping encountered')
                break
    pg.info('Finished Training')
    #print('Finished Training')
    
# ## Count the number of parameters in the network
def countNetParameters(Net, verbose=True):
    """ Count the number of parameters in the network """
    total = sum(p.numel() for p in Net.parameters())
    trainable = sum(p.numel() for p in Net.parameters() if p.requires_grad)
    if verbose:
        print(f"Total number of parameters in the network   :  {total}")
        print(f"Number of trainable parameters in the network   : {trainable}")
    return total, trainable

# ## RSME evalution
def rmse_loss(y_true, y_pred):
    return torch.sqrt(torch.mean((y_pred - y_true)**2))

def test_rsme(Net, testload, table_file=None, dataset_type='Multi-Layers', dataset_size=1000, config='Wenner-Schlumberger'
              ,  model='Unet', drop_ex_row=None, lam=None, device=None):
    total_params, trainable_params = countNetParameters(Net, verbose=False)
    Net.eval()
    total_rmse = 0.0
    nb_batches = 0
    with torch.no_grad():
        for i, (y, yhat, info_linked_id) in enumerate(testload):
            if device:
                y, yhat = y.to(device, non_blocking=True), yhat.to(device, non_blocking=True)
            y_pred = Net(yhat)
            current_rmse = rmse_loss(y, y_pred)
            total_rmse += current_rmse.item()
            nb_batches += 1
    rmse_mean = total_rmse / nb_batches
    print('Average RMSE: %.4f' %  rmse_mean)
    if lam!=None:
        df_row = pd.DataFrame({'Data type': [dataset_type], 'Measurement arrays': [config], 'Dataset size': [dataset_size], 'model': [model]
                               , 'Nb parameters': [trainable_params], 'Lambda': [lam], 'RMSE': [round(rmse_mean, 4)] })
    else:
        df_row = pd.DataFrame({'Data type': [dataset_type], 'Measurement arrays': [config], 'Dataset size': [dataset_size], 'model': [model]
                            , 'Nb parameters': [trainable_params], 'RMSE': [round(rmse_mean, 4)] })
    if os.path.exists(table_file) == False:
        table_df = df_row
        table_df.to_csv(table_file, index=False)
    else:
        table_df = pd.read_csv(table_file)
        if drop_ex_row:
            table_df = table_df.drop(table_df.index[drop_ex_row])
        table_df = pd.concat([table_df, df_row])
        table_df.to_csv(table_file, index=False)
    return table_df