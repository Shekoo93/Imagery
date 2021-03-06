
colornames = ["red", "blue", "green", "purple", "yellow", "cyan", "orange", "brown", "pink", "teal"]
# MNIST VAE from http://
# Modified by Brad Wyble and Shekoo Hedayati
# Modifications:
#   Colorize transform that changes the colors of a grayscale image
# colors are chosen from 10 options:
#colornames = ["red", "blue", "green", "purple", "yellow", "cyan", "orange", "brown", "pink", "teal"]
# specified in "colorvals" variable below

# also there is a skip connection from the first layer to the last layer to enable reconstructions of new stimuli
# and the VAE bottleneck is split, having two different maps
# one is trained with a loss function for color only (eliminating all shape info, reserving only the brightest color)
# the other is trained with a loss function for shape only
#
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from config import numcolors,args

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load


colorlabels = [0,1]
colorrange = .1
colorvals = [
    [1 - colorrange, colorrange * 1, colorrange * 1],
    [colorrange * 1, 1 - colorrange, colorrange * 1],
]



try:
    import accimage
except ImportError:
    accimage = None

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def Colorize_func(img):
    global numcolors,colorlabels  # necessary because we will be modifying this counter variable

    #if not _is_pil_image(img):
    #    img = tensor_to_PIL(img)

    thiscolor = colorlabels[numcolors]  # what base color is this?

    rgb = colorvals[thiscolor];  # grab the rgb for this base color
    numcolors += 1  # increment the index

    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange

    #img = img.convert('L')

    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img, 'RGB')

    return img

def Colorize_func_grey(img):
    # global numcolors  #necessary because we will be modifying this counter variable

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    rgb=[1,1,1]  #for grey scale images
    r_color = rgb[0]  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1]
    b_color = rgb[2]
    av=(r_color+g_color+b_color)/3
    img = img.convert('L')
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * av, np_img * av, np_img * av])
    np_img = np_img.astype(np.uint8)

    img = Image.fromarray(np_img, 'RGB')

    return img
def Colorize_func_red(img):
    # global numcolors  #necessary because we will be modifying this counter variable

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    colornum = 0
    thiscolor = colorlabels[colornum]  # what base color is this?
    rgb = colorvals[thiscolor];  # grab the rgb for this base color
    # numcolors += 1  #increment the index

    r_color = rgb[
                  0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange

    img = img.convert('L')
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    np_img = np_img.astype(np.uint8)

    img = Image.fromarray(np_img, 'RGB')

    return img


def Colorize_func_green(img):
    # global numcolors  #necessary because we will be modifying this counter variable

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    colornum = 1
    thiscolor = colorlabels[colornum]  # what base color is this?
    rgb = colorvals[thiscolor];  # grab the rgb for this base color
    # numcolors += 1  #increment the index

    r_color = rgb[
                  0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange

    img = img.convert('L')
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    np_img = np_img.astype(np.uint8)

    img = Image.fromarray(np_img, 'RGB')

    return img
def Colorize_func_secret(img,npflag = 0):
    global numcolors,colorlabels  # necessary because we will be modifying this counter variable

    #if not _is_pil_image(img):
    #    img = tensor_to_PIL(img)

    thiscolor = colorlabels[numcolors]  # what base color is this?
    thiscolor = np.random.randint(10)

    rgb = colorvals[thiscolor];  # grab the rgb for this base color
      # increment the index

    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange

    #img = img.convert('L')

    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    np_img[0,0,0] = thiscolor   #secretely embed the color label inside
        #this is a temporary fix
    #print(np_img[0,0,0])
    #print(numcolors)
    img = Image.fromarray(np_img, 'RGB')
    if npflag ==1:
        img = backup

    return img
# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import imageio
import os

from config import  args
from dataloader import notMNIST

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

bs = 100
nw = 8
mini_bs=25

def data_filter (data_type, selected_labels):
  data_trans= copy.deepcopy(data_type)
  data_type_labels= data_type.targets
  idx_selected= np.isin(data_type_labels, selected_labels)
  idx_selected=torch.tensor(idx_selected)
  data_trans.targets= data_type_labels[idx_selected]
  data_trans.data = data_type.data[idx_selected]
  return data_trans
#MNIST Dataset
train_dataset= datasets.MNIST(root='./mnist_data/', train=True,
                               transform=transforms.Compose([Colorize_func, transforms.ToTensor()]), download=True)
test_dataset = datasets.MNIST(root='./mnist_data/', train=False,
                              transform=transforms.Compose([Colorize_func, transforms.ToTensor()]), download=False)


#train_dataset_labels=datasets.MNIST(root='./mnist_data/', train=True,
                              # transform=transforms.Compose([Colorize_func_secret, transforms.ToTensor()]), download=True)

#train_loader_labels=torch.utils.data.DataLoader(dataset=train_dataset_labels, batch_size=bs, shuffle=True,  drop_last= True,num_workers=nw)

#train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,  drop_last= True,num_workers=nw)
#train_loader_labels=torch.utils.data.DataLoader(dataset=input_oneHot, batch_size=bs, shuffle=True,  drop_last= True,num_workers=nw)

#test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True,num_workers=nw)

#test_loader_smaller = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=mini_bs, shuffle=False, drop_last=True,num_workers=nw)
train_dataset_red = datasets.MNIST(root='./mnist_data/', train=True , transform= transforms.Compose([Colorize_func_red,transforms.ToTensor()]), download=True)
train_dataset_g = datasets.MNIST(root='./mnist_data/', train=True ,transform=transforms.Compose([Colorize_func_green,transforms.ToTensor()]), download=True)
test_dataset_red= datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.Compose([Colorize_func_red, transforms.ToTensor()]), download=False)
test_dataset_g= datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.Compose([Colorize_func_green, transforms.ToTensor()]), download=False)


dataset_train_filt1= data_filter(train_dataset_red,[0,1,2,3,4])
dataset_train_filt2= data_filter(train_dataset_g,[5,6,7,8,9])
test_filtered1= data_filter(test_dataset_red,[7])
test_filtered2= data_filter(test_dataset_g,[0])
dataset_train_combined= torch.utils.data.ConcatDataset((dataset_train_filt1,dataset_train_filt2))

dataset_test_combined= torch.utils.data.ConcatDataset((test_filtered1,test_filtered2))

#combined_imgs= torch.cat(images_red_tr, images_green_tr )

train_loader = torch.utils.data.DataLoader(dataset_train_combined, batch_size=bs, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset_test_combined, batch_size=bs, shuffle=False, drop_last=True)



#train and test the classifiers on MNIST and f-MNIST
bs_tr=60000
bs_te=10000

#train_loader_class= torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs_tr, shuffle=True,num_workers=nw)
#test_loader_class= torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_te, shuffle=False,num_workers=nw)


matching=0
if matching==1:
    dataiter = iter(train_loader)
    images,labels=dataiter.next()
    input_oneHot = F.one_hot(labels)
    print(input_oneHot)


def thecolorlabels(datatype):
    # colorlabels = np.random.randint(0,10,100000)
    colornumstart = 0
    coloridx = range(colornumstart, len(datatype))
    labelscolor = colorlabels[coloridx]
    return torch.tensor(labelscolor)

testdataset = 0
if (testdataset == 1):
    dataiter = iter(train_loader)
    # get some random training iglobal numcolorsmages
    images, labels = dataiter.next()
    labels_color=thecolorlabels(images)

    print(labels_color)
    #print(images[0][0][0])
    save_image(images, 'sample5.png')
    img = mpimg.imread('sample5.png')
    # print(img[1:20,1:20,2])
    plt.imshow(img)
    plt.show()

# Creating a dataloader



class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)  # shape
        self.fc32 = nn.Linear(h_dim2, z_dim)

        self.fc33 = nn.Linear(h_dim2, z_dim)  # color
        self.fc34 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # shape
        self.fc4c = nn.Linear(z_dim, h_dim2)  # color
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)


    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h), self.fc33(h), self.fc34(h)  # mu, log_var


    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder_noskip(self, z_shape, z_color):
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def decoder_color(self, z_shape, z_color):
        h = F.relu(self.fc4c(z_color))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def decoder_shape(self, z_shape, z_color):
        h = F.relu(self.fc4s(z_shape))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))




    def forward(self, x, whichdecode, detatchgrad='none'):
        mu_shape, log_var_shape, mu_color, log_var_color = self.encoder(x.view(-1, 784 * 3))
        if (detatchgrad == 'shape'):
            z_shape = self.sampling(mu_shape, log_var_shape).detach()
        else:
            z_shape = self.sampling(mu_shape, log_var_shape)

        if (detatchgrad == 'color'):
            z_color = self.sampling(mu_color, log_var_color).detach()
        else:
            z_color = self.sampling(mu_color, log_var_color)

        if (whichdecode == 'noskip'):
            output = self.decoder_noskip(z_shape, z_color)
        elif (whichdecode == 'color'):
            output = self.decoder_color(0, z_color)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape, 0)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

"""
class VAElabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim,  zlabel_dim):
        super(VAElabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label= nn.Linear(hlabel_dim,  zlabel_dim) #mu shape
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim) #log-var shape

        self.fc23label = nn.Linear(hlabel_dim, zlabel_dim)  # mu color
        self.fc24label = nn.Linear(hlabel_dim, zlabel_dim) #log_var color

    def sampling_labels (self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_labels,  detatchgrad='none'):

        h = F.relu(self.fc1label(x_labels))

        mu_shape_label = self.fc21label(h)
        log_var_shape_label=self.fc22label(h)

        mu_color_label=self.fc23label(h)
        log_var_color_label = self.fc24label(h)

        if (detatchgrad == 'shape'):
            z_shape_label = self.sampling_labels(mu_shape_label, log_var_shape_label).detach()
        else:
            z_shape_label = self.sampling_labels(mu_shape_label, log_var_shape_label)

        if (detatchgrad == 'color'):
            z_color_label = self.sampling_labels(mu_color_label, log_var_color_label).detach()
        else:
            z_color_label = self.sampling_labels(mu_color_label, log_var_color_label)

        return  z_shape_label, z_color_label
"""

class VAEshapelabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim,  zlabel_dim):
        super(VAEshapelabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label= nn.Linear(hlabel_dim,  zlabel_dim) #mu shape
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim) #log-var shape


    def sampling_labels (self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_labels):

        h = F.relu(self.fc1label(x_labels))

        mu_shape_label = self.fc21label(h)
        log_var_shape_label=self.fc22label(h)

        z_shape_label = self.sampling_labels(mu_shape_label, log_var_shape_label)


        return  z_shape_label

# reload a saved file
class VAEcolorlabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim, zlabel_dim):
        super(VAEcolorlabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        self.fc21label = nn.Linear(hlabel_dim, zlabel_dim)  # mu color
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim)  # log-var color

    def sampling_labels(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_labels):
        h = F.relu(self.fc1label(x_labels))

        mu_shape_label = self.fc21label(h)
        log_var_shape_label = self.fc22label(h)

        z_color_label = self.sampling_labels(mu_shape_label, log_var_shape_label)

        return  z_color_label

def image_activations(imgs):
    with torch.no_grad():
        vae.eval()
        mu_shape, log_var_shape, mu_color, log_var_color = vae.encoder(imgs.view(-1, 784 * 3))
        z_shape= vae.sampling(mu_shape, log_var_shape)
        z_color=vae.sampling(mu_color, log_var_color)
    return z_shape, z_color

def image_recon(z_labels):
    with torch.no_grad():
        vae.eval()
        output=vae.decoder_noskip(z_labels)
    return output

# build model
vae = VAE(x_dim=784 * 3, h_dim1=256, h_dim2=128, z_dim=4)
# vae = torch.nn.DataParallel(vae)
vae_shape_labels= VAEshapelabels(xlabel_dim=10, hlabel_dim=7,  zlabel_dim=4)

vae_color_labels= VAEcolorlabels(xlabel_dim=2, hlabel_dim=7,  zlabel_dim=4)

if torch.cuda.is_available():
    vae.cuda()
    vae_shape_labels.cuda()
    vae_color_labels.cuda()
    print('CUDA')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    vae.load_state_dict(checkpoint['state_dict'])

    for parameter in vae.parameters():
        parameter.requires_grad = False
    vae.eval()
    return vae


def load_checkpoint_shapelabels(filepath):
    checkpoint = torch.load(filepath)
    vae_shape_labels.load_state_dict(checkpoint['state_dict_shape_labels'])
    for parameter in vae_shape_labels.parameters():
        parameter.requires_grad = False
    vae_shape_labels.eval()
    return vae_shape_labels

def load_checkpoint_colorlabels(filepath):
    checkpoint = torch.load(filepath)
    vae_color_labels.load_state_dict(checkpoint['state_dict_color_labels'])
    for parameter in vae_color_labels.parameters():
        parameter.requires_grad = False
    vae_color_labels.eval()
    return vae_color_labels

optimizer = optim.Adam(vae.parameters())

optimizer_shapelabels= optim.Adam(vae_shape_labels.parameters())
optimizer_colorlabels= optim.Adam(vae_color_labels.parameters())

def loss_label(label_act,image_act):

    criterion=nn.MSELoss(reduction='sum')

    #Loss=nn.CrossEntropyLoss()(z_label,Variable(labels))

    #z_label=torch.tensor(z_label,requires_grad=True)
    #e=nn.CrossEntropyLoss()(z_label,z_shape)
    e=criterion(label_act,image_act)
   # print('the error is')
    #print(e)
    return e



# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):

    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784 * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD

def loss_function_shape(recon_x, x, mu, log_var):
    # make grayscale reconstruction
    grayrecon = recon_x.view(bs, 3, 28, 28).mean(1)
    # print(grayrecon.shape)
    grayrecon = torch.stack([grayrecon, grayrecon, grayrecon], dim=1)
    # print(grayrecon.shape)
    # plot grayscale images to prove it's working
    # COMMENTED OUT
    # img = grayrecon[:,:,:,:].cpu().detach()
    # save_image(img, 'sample.png')
    # img = mpimg.imread('sample.png')
    # plt.imshow(img)
    # plt.show()
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784*3), reduction='sum')
    # here's a loss BCE based only on the grayscale reconstruction.  Use this in the return statement to kill color learning
    BCEGray = F.binary_cross_entropy(grayrecon.view(-1, 784 * 3), x.view(-1, 784 * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCEGray + KLD


def loss_function_color(recon_x, x, mu, log_var):
    # make color-only (no shape) reconstruction and use that as the loss function
    recon = recon_x.clone().view(bs, 3, 784)
    # compute the maximum color for the r,g and b channels for each digit separately
    maxr, maxi = torch.max(recon[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(recon[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(recon[:, 2, :], -1, keepdim=True)

    # now build a new reconsutrction that has only the max color, and no shape information at all
    # dividing by 4 is very helpful here, without it, the reconstructions are very dim
    # Not entrely sure
    recon[:, 0, :] = maxr
    recon[:, 1, :] = maxg
    recon[:, 2, :] = maxb
    recon = recon.view(-1, 784 * 3)
    maxr, maxi = torch.max(x[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(x[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(x[:, 2, :], -1, keepdim=True)
    newx = x.clone()
    newx[:, 0, :] = maxr
    newx[:, 1, :] = maxg
    newx[:, 2, :] = maxb
    newx = newx.view(-1, 784 * 3)
    BCE = F.binary_cross_entropy(recon, newx, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch,whichdecode):
   
    #colorlabels = np.random.randint(0, 10,100000)  # regenerate the list of color labels at the start of each test epoch
    
    vae.train()
    train_loss = 0
    loader = tqdm(train_loader)


    for batch_idx, (data, _) in enumerate(loader):  # get the next batch
        # pull the color labels out of the colorlabels list
       # colornumstart = 0 + batch_idx * bs  # figure out which numbers these are from the batch_idx
      #  coloridx = range(colornumstart, colornumstart + bs)
        # batch_colornames = list()   #this can be enabled if we want the color names
        #batch_colorlabels = list()
       # for color in coloridx:  # get the list of color labels for this batch
           # thiscolor = colorlabels[color]
          #  batch_colorlabels.append(thiscolor)
        #          batch_colornames.append(colornames[thiscolor])   #this can be enabled if we want the color names
        # used to just verify that color labels are correct
        # save_image(data, 'sample.png')
        # img = mpimg.imread('sample.png')
        # plt.imshow(img)
        # plt.show()
        # print(batch_colornames)
        data = data.cuda()
        detachgrad = 'none'
        optimizer.zero_grad()
        if (whichdecode == 'iterated'):
            if batch_idx % 2 == 0:  #
                whichdecode_use = 'noskip'
                detachgrad = 'color'

            else:
                whichdecode_use = 'noskip'
                detachgrad = 'shape'

            recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use, detachgrad)

            if (whichdecode == 'iterated'):
                if batch_idx % 2 == 0:  # one of out 3 times, let's use the skip connection
                    loss = loss_function_shape(recon_batch, data, mu_shape, log_var_shape)
                    loss.backward()

                else:
                    loss = loss_function_color(recon_batch, data, mu_color, log_var_color)
                    loss.backward()

                train_loss += loss.item()
                optimizer.step()
                loader.set_description(
                    (
                        f'epoch: {epoch + 1}; mse: {loss.item():.5f};'
                    )
                )
                sample_size = 25
                if batch_idx % 1000 == 0:
                    vae.eval()
                    sample = data[:sample_size]
                    with torch.no_grad():
                        reconb, mu_color, log_var_color, mu_shape, log_var_shape = vae(sample, 'noskip')
                        reconc, mu_color, log_var_color, mu_shape, log_var_shape = vae(sample, 'color')
                        recons, mu_color, log_var_color, mu_shape, log_var_shape = vae(sample, 'shape')
                    # print(sample.shape)
                    # print(reconb.shape)
                    utils.save_image(
                        torch.cat([sample, reconb.view(sample_size, 3, 28, 28),
                                   reconc.view(sample_size, 3, 28, 28), recons.view(sample_size, 3, 28, 28)], 0),
                        f'sample/{str(epoch + 1).zfill(5)}_{str(batch_idx).zfill(5)}.png',
                        nrow=sample_size,
                        normalize=False,
                        range=(-1, 1),
                    )
                # if batch_idx % 100 == 0:
                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(data), len(train_loader.dataset),
                #         100. * batch_idx / len(train_loader), ( train_loss) /( (batch_idx+1) *len(data))))
                #
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(whichdecode):
    vae.eval()
    
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):  # get the next batch
            #colornumstart = 0 + batch_idx * bs  # figure out which numbers these are from the batch_idx
            #coloridx = range(colornumstart, colornumstart + bs)
            # batch_colornames = list()   #this can be enabled if we want the color names
           # batch_colorlabels = list()
            #for color in coloridx:  # get the list of color labels for this batch
                #thiscolor = colorlabels[color]
               # batch_colorlabels.append(thiscolor)
            data = data.cuda()
            recon, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode)

            # sum up batch loss
            test_loss += loss_function_shape(recon, data, mu_shape, log_var_shape).item()
            test_loss += loss_function_color(recon, data, mu_color, log_var_color).item()


    print('Example reconstruction')
    data = data.cpu()
    data=data.view(bs, 3, 28, 28)
    save_image(data[0:8], f'{args.dir}/recon.png')
    #    img = mpimg.imread('recon.png')
    #    plt.imshow(img)
    #    plt.show()

    print('Imagining a shape')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, 4).cuda() * 0
        zs = torch.randn(64, 4).cuda() * 1
        sample = vae.decoder_noskip(zs, zc).cuda()
        sample=sample.view(64,3,28,28)
        save_image(sample[0:8], f'{args.dir}/sampleshape.png')
    #      img = mpimg.imread('sampleshape.png')
    #      plt.imshow(img)
    #      plt.show()

    print('Imagining a color')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, 4).cuda() * 1
        zs = torch.randn(64, 4).cuda() * 0
        sample = vae.decoder_noskip(zs, zc).cuda()
        sample=sample.view(64, 3, 28, 28)
        save_image(sample[0:8], f'{args.dir}/samplecolor.png')
    #      img = mpimg.imread('samplecolor.png')
    #      plt.imshow(img)
    #      plt.show()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))



def train_labels(epoch):
    global colorlabels, numcolors
    #colorlabels = np.random.randint(0,2, 100000)
    #print(colorlabels)
    #colors= torch.tensor([0,1])
    # colors=colors.clone().detach()

   # colorlabels = torch.cat(100000* [colors])
    

    numcolors = 0
    train_loss_shapelabel = 0
    train_loss_colorlabel = 0

    vae_shape_labels.train()
    vae_color_labels.train()

    dataiter = iter(train_loader)
    red_labels=torch.tensor([0,1,2,3,4])
    green_labels=torch.tensor([5,6,7,8,9])
    # labels_color=0

    for i in range(len(train_loader)):

        # print(len(train_loader))
        # z_color_one=torch.tensor([ 0.9379, -4.8616, -0.2427, -1.3959]).cuda()
        # z_color_two=torch.tensor([ -0.2391,  6.2767, -0.8730, -5.5947]).cuda()
        # z_color_all=torch.stack([z_color_one,z_color_two])
        # z_color_all=z_color_all.repeat(50,1)
        # print(z_color_all.shape)

        optimizer_shapelabels.zero_grad()
        optimizer_colorlabels.zero_grad()


        image, labels = dataiter.next()
       
        
        labels_forcolor=labels.clone()
        for col in red_labels:
            labels_forcolor[labels_forcolor==col]=0
        for col in green_labels:
            labels_forcolor[labels_forcolor==col]=1
            
        
        image = image.cuda()
        labels = labels.cuda()
        
        input_oneHot = F.one_hot(labels, num_classes=10)
        input_oneHot = input_oneHot.float()
        input_oneHot = input_oneHot.cuda()

        labels_color = labels_forcolor  # get the color labels
        
        labels_color = labels_color.cuda()
        
        #print(labels_color)
        
        color_oneHot = F.one_hot(labels_color, num_classes=2)
        color_oneHot = color_oneHot.float()
        color_oneHot = color_oneHot.cuda()



        z_shape_label = vae_shape_labels(input_oneHot)
        z_color_label = vae_color_labels(color_oneHot)

        z_shape, z_color = image_activations(image)


        loss_of_shapelabels = loss_label(z_shape_label, z_shape)
        loss_of_shapelabels.backward()
        train_loss_shapelabel += loss_of_shapelabels.item()
            # print(train_loss_label)
        optimizer_shapelabels.step()

        loss_of_colorlabels = loss_label(z_color_label, z_color)
        loss_of_colorlabels.backward()
        train_loss_colorlabel += loss_of_colorlabels.item()
            # print(train_loss_label)
        optimizer_colorlabels.step()

        if i % 1000 == 0:
            vae_shape_labels.eval()
            vae_color_labels.eval()
            vae.eval()
            # print(labels_color)

            with torch.no_grad():
                # print(color_oneHot[:5])
                # print(z_color_label-z_color)
                # print(loss_of_labels)
                # print('color map for labels')
                # print(z_color_label)

                # print('color map for images')
                # print(z_color)

                recon_imgs = vae.decoder_noskip(z_shape, z_color)
                recon_imgs_shape = vae.decoder_shape(z_shape, z_color)
                recon_imgs_color = vae.decoder_color(z_shape, z_color)

                recon_labels = vae.decoder_noskip(z_shape_label, z_color_label)
                recon_shapeOnly = vae.decoder_shape(z_shape_label, 0)
                recon_colorOnly = vae.decoder_color(0, z_color_label)
                # recon_imgs=vae.decoder_noskip(z_shape)

                sample_size = 20
                orig_imgs = image[:sample_size]
                recon_labels = recon_labels[:sample_size]
                recon_imgs = recon_imgs[:sample_size]
                recon_imgs_shape = recon_imgs_shape[:sample_size]
                recon_imgs_color = recon_imgs_color[:sample_size]
                recon_shapeOnly = recon_shapeOnly[:sample_size]
                recon_colorOnly = recon_colorOnly[:sample_size]

            utils.save_image(
                torch.cat(
                    [orig_imgs, recon_imgs.view(sample_size, 3, 28, 28), recon_imgs_shape.view(sample_size, 3, 28, 28),
                     recon_imgs_color.view(sample_size, 3, 28, 28), recon_labels.view(sample_size, 3, 28, 28),
                     recon_shapeOnly.view(sample_size, 3, 28, 28), recon_colorOnly.view(sample_size, 3, 28, 28)], 0),
                f'sample_training_labels/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )

    print(
        '====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss_shapelabel / (len(train_loader.dataset) / bs)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch,train_loss_colorlabel / (len(train_loader.dataset) / bs)))
 

def test_outputs():

    vae_shape_labels.eval()
    vae_color_labels.eval()
    recons=list()

    with torch.no_grad():

        # get some random training iglobal numcolorsmages
        labels_shape=[2]
        labels_shape=torch.Tensor(labels_shape).cuda()

        labels_color = [1]
        labels_color=torch.tensor(labels_color).cuda()

        shape_oneHot = F.one_hot(labels_shape.long(), num_classes=10)
        shape_oneHot = shape_oneHot.float()
        shape_oneHot = shape_oneHot.cuda()

        color_oneHot = F.one_hot(labels_color.long(), num_classes=2)
        color_oneHot = color_oneHot.float()
        color_oneHot = color_oneHot.cuda()


        z_label_shape= vae_shape_labels(shape_oneHot)

        z_label_color = vae_color_labels( color_oneHot)
        #noise_levels=torch.randn_like(z_label)
        noise_sd_c=torch.tensor(np.arange(0,25,.5))
        noise_sd_s=torch.tensor(np.arange(0,25,.5))
        
        #noise_levels=torch.tensor(np.arange(-3,3.5,.5))
        for step in range(len(noise_sd_s)):
           z_label_shape= vae_shape_labels(shape_oneHot)
           z_label_color = vae_color_labels( color_oneHot)
          
           if step==0:
             noise_s=0
             noise_c=0
           else: 
 
             noise_s=torch.normal(mean=0, std=noise_sd_s[2] ,size=z_label_color.shape).cuda()
             noise_c=torch.normal(mean=0, std=noise_sd_c[2] ,size=z_label_color.shape).cuda()
             
              #noise_s=1
              #z_label_shape=z_label_shape*noise_s
           z_label_color= z_label_color
              
           z_label_shape=z_label_shape+noise_s
             
           recons_label = vae.decoder_noskip(z_label_shape,z_label_color ).cuda()

           recons.append(recons_label)
           recons_new=torch.stack(recons)

        utils.save_image(recons_new.view(-1, 3, 28, 28),'label_recon.png',nrow=10,
                normalize=False,
                range=(-1, 1),
            )
        
