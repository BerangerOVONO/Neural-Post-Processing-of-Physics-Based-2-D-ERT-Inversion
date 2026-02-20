########################################################################
# CNNs modules and networks                                       ######
# Author: Julien MILLE (master) , Béranger OVONO EKORE (modif)    ######                 
########################################################################

# full assembly of the sub-parts to form the complete net

# sub-parts of the U-Net model

# Libs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys

# dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.nbChannelsIn = in_ch
        self.nbChannelsOut = out_ch
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.nbChannelsIn = in_ch
        self.nbChannelsOut = out_ch
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        #x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
        #               diffY // 2, int(diffY / 2)))
        #print('x2', x2.shape)
        #print('x1', x1.shape)
        if x2.shape != x1.shape:
            x1 = TF.resize(x1, size=x2.shape[2:])
        x= torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class lastconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(lastconv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1)
                                  , nn.BatchNorm2d(out_ch)
                                  , nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class fullConnec(nn.Module):
    def __init__(self, in_nbclasse, out_nbclasse):
        super(fullConnec, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_nbclasse, out_nbclasse)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(out_nbclasse, out_nbclasse))
        
    def forward(self, x):
        x = self.fc(x)
        return x

class fullConnecDropout(nn.Module):
    def __init__(self, in_nbclasse, out_nbclasse, drop_rate=0.3):
        super(fullConnecDropout, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_nbclasse, out_nbclasse)
                                , nn.ReLU(inplace=True)
                                , nn.Dropout(p=drop_rate)
                                , nn.Linear(out_nbclasse, out_nbclasse))
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
class fullConnecDeep(nn.Module):
    def __init__(self, in_nbclasse, out_nbclasse):
        super(fullConnecDeep, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_nbclasse, out_nbclasse)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(out_nbclasse, out_nbclasse)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(out_nbclasse, out_nbclasse)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(out_nbclasse, out_nbclasse))
        
    def forward(self, x):
        x = self.fc(x)
        return x
    

class outConvLinear(nn.Module):
    def __init__(self, in_ch, out_ch, in_flat, out_flat):
        super(outConvLinear, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1)
                                  , nn.BatchNorm2d(out_ch)
                                  , nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Linear(in_flat, out_flat)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(out_flat, out_flat))
                                                  
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1 
        for s in size:
            num_features *= s
        return num_features
    
class outConvLinearDeep(nn.Module):
    def __init__(self, in_ch, out_ch, in_flat, out_flat):
        super(outConvLinearDeep, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1)
                                  , nn.BatchNorm2d(out_ch)
                                  , nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Linear(in_flat, out_flat)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(out_flat, out_flat)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(out_flat, out_flat)
                                , nn.ReLU(inplace=True)
                                , nn.Linear(out_flat, out_flat))
                               
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1 
        for s in size:
            num_features *= s
        return num_features

# VGG-like CNN with flexible number of convolutional layers and fully connected layers (not used in the paper, but tested in preliminary experiments).
class CNNFlexFlat(nn.Module):
    def __init__(self, nbfeaturesbase, nblayers, ratiofeatures, nbclasses, input_shape=(1, 32, 32)):
        super(CNNFlexFlat, self).__init__()
        # couche d'entrée de la convolution
        self.inc = inconv(input_shape[0], nbfeaturesbase)
        # couche de sortie de la convolution 
        self.outconv = lastconv(nbfeaturesbase*ratiofeatures**(nblayers-1), input_shape[0])  
        # les couches de convolutions
        self.conv_layers = [self.inc]
        if nblayers>1:
            for l in range(1, nblayers):
                self.conv_layers.append(down(int(nbfeaturesbase*(ratiofeatures**(l-1))), int(nbfeaturesbase*(ratiofeatures**(l)))))
        self.conv_layers.append(self.outconv)
        # add to module
        for i in range(len(self.conv_layers)):
            self.add_module("conv_layers" + '{:01}'.format(i), self.conv_layers[i])
        # couches denses
        self.outconv_flat = self._get_conv_output(input_shape)
        self.outfc = fullConnec(self.outconv_flat, nbclasses)
    
    def forward(self, x):
        for l in range(len(self.conv_layers)):
            x = self.conv_layers[l](x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.outfc(x)
        return x
    
    def _get_conv_output(self, shape):
        x = torch.zeros(1, *shape)
        for l in range(len(self.conv_layers)):
            x = self.conv_layers[l](x)
        return x.view(1, -1).size(1)
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1 
        for s in size:
            num_features *= s
        return num_features

# U-Net with flexible number of convolutional layers and fully connected layers at the end (not used in the paper, but tested in preliminary experiments).
class UNetFlexFlat(nn.Module):
    def __init__(self, nbclasses, nbfeaturesbase, nblevels, ratiofeatures, in_flat, out_flat):
        super(UNetFlexFlat, self).__init__()
        self.inc = inconv(nbclasses, nbfeaturesbase)
        self.outc = outConvLinear(nbfeaturesbase, nbclasses, in_flat, out_flat)

        self.downLayers = []
        self.upLayers = []

        if nblevels>0:
            for l in range(nblevels-1):
                self.downLayers.append(down(int(nbfeaturesbase*(ratiofeatures**l)), int(nbfeaturesbase*(ratiofeatures**(l+1)))))
            self.downLayers.append(down(int(nbfeaturesbase*(ratiofeatures**(nblevels-1))), int(nbfeaturesbase*(ratiofeatures**(nblevels-1)))))

            # self.upLayers.append(up(nbfeaturesbase*(ratiofeatures**nblevels), nbfeaturesbase*(ratiofeatures**(nblevels-2))))
            for l in range(nblevels-1):
                in_ch = self.downLayers[nblevels-1-l].nbChannelsOut + self.downLayers[nblevels-2-l].nbChannelsOut 
                self.upLayers.append(up(in_ch, self.downLayers[nblevels-2-l].nbChannelsOut))
            self.upLayers.append(up(nbfeaturesbase + self.downLayers[0].nbChannelsOut , nbfeaturesbase))

            # print('Creation')
            for i in range(len(self.downLayers)):
                # print('down', i, ' : in=', self.downLayers[i].nbChannelsIn, ' out=', self.downLayers[i].nbChannelsOut)
                self.add_module("DownLayer" + '{:01}'.format(i), self.downLayers[i])
            for i in range(len(self.upLayers)):
                # print('up', i, ' : in=', self.upLayers[i].nbChannelsIn, ' out=', self.upLayers[i].nbChannelsOut)
                self.add_module("UpLayer" + '{:01}'.format(i), self.upLayers[i])

    def forward(self, x):
        nblevels = len(self.downLayers)
        if nblevels>0:
            xs = []
            xs.append(self.inc(x))
            for l in range(nblevels):
                # print('down', l, ' : in=', self.downLayers[l].nbChannelsIn, ' out=', self.downLayers[l].nbChannelsOut, ' shape=', xs[-1].shape)
                xs.append(self.downLayers[l](xs[-1]))

            x = self.upLayers[0](xs[nblevels], xs[nblevels-1])
            for l in range(1,nblevels):
                x = self.upLayers[l](x, xs[nblevels-1-l])
            x = self.outc(x)
        else:
            x = self.outc(self.inc(x))
        return x

# --- U-Net with flexible number of convolutional layers without fully connected layers at the end (the one used in the paper). ---
class UNetFlex(nn.Module):
    def __init__(self, nbclasses, nbfeaturesbase, nblevels, ratiofeatures):
        super(UNetFlex, self).__init__()
        self.inc = inconv(1, nbfeaturesbase)
        self.outc = outconv(nbfeaturesbase, nbclasses)

        self.downLayers = []
        self.upLayers = []

        if nblevels>0:
            for l in range(nblevels-1):
                self.downLayers.append(down(int(nbfeaturesbase*(ratiofeatures**l)), int(nbfeaturesbase*(ratiofeatures**(l+1)))))
            self.downLayers.append(down(int(nbfeaturesbase*(ratiofeatures**(nblevels-1))), int(nbfeaturesbase*(ratiofeatures**(nblevels-1)))))

            # self.upLayers.append(up(nbfeaturesbase*(ratiofeatures**nblevels), nbfeaturesbase*(ratiofeatures**(nblevels-2))))
            for l in range(nblevels-1):
                in_ch = self.downLayers[nblevels-1-l].nbChannelsOut + self.downLayers[nblevels-2-l].nbChannelsOut 
                self.upLayers.append(up(in_ch, self.downLayers[nblevels-2-l].nbChannelsOut))
            self.upLayers.append(up(nbfeaturesbase + self.downLayers[0].nbChannelsOut , nbfeaturesbase))

            # print('Creation')
            for i in range(len(self.downLayers)):
                # print('down', i, ' : in=', self.downLayers[i].nbChannelsIn, ' out=', self.downLayers[i].nbChannelsOut)
                self.add_module("DownLayer" + '{:01}'.format(i), self.downLayers[i])
            for i in range(len(self.upLayers)):
                # print('up', i, ' : in=', self.upLayers[i].nbChannelsIn, ' out=', self.upLayers[i].nbChannelsOut)
                self.add_module("UpLayer" + '{:01}'.format(i), self.upLayers[i])

    def forward(self, x):
        nblevels = len(self.downLayers)
        if nblevels>0:
            xs = []
            xs.append(self.inc(x))
            for l in range(nblevels):
                # print('down', l, ' : in=', self.downLayers[l].nbChannelsIn, ' out=', self.downLayers[l].nbChannelsOut, ' shape=', xs[-1].shape)
                xs.append(self.downLayers[l](xs[-1]))

            x = self.upLayers[0](xs[nblevels], xs[nblevels-1])
            for l in range(1,nblevels):
                x = self.upLayers[l](x, xs[nblevels-1-l])
            x = self.outc(x)
        else:
            x = self.outc(self.inc(x))
        return x