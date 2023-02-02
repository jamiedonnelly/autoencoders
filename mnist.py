# Misc. Imports
import argparse
import logging
from typing import Union, Dict
import matplotlib.pyplot as plt
import os

# Backend imports
import numpy as np  
import torch 
from torch import nn
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader

# Local Imports
from models import model_dict
import parsing
import autoencoder
import build

parser = argparse.ArgumentParser(description='MNIST Autoencoder Training for Torchvision models')
parser.add_argument('--arch','-a',type=str,choices=model_dict.keys(),help='Choice of architecture based on \
    existing torchvision convolutional architectures, e.g., VGG11, VGG19, ResNet50, etc...')
parser.add_argument('--latent_dim',default=4,type=int,help='Number of dimensions for the latent variables produced\
    by the encoding subnetwork. (default: 4)')
parser.add_argument('--epochs',default=10,type=int,help='Number of epochs to train the model for')
parser.add_argument('--b','--batch_size',default=32,type=int,help='mini-batch size (default: 32)')
parser.add_argument('--lr','--learning-rate',default=1e-3,type=float,help='Initial learning rate')
parser.add_argument('--betas', default=(0.9,0.999), type=tuple, help='(B0,B1) params for ADAM optimizer.')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--plot-freq', '-p', default=500, type=int,
    metavar='N', help='plot frequency in batches (default: 500) i.e., plot sample of predictions\
    every 500 batches.')

# Create logger  
def get_logger(dir,filename):

    # Remove existing log if exists 
    filename = os.path.join(dir,filename)
    if filename in os.listdir(dir):
        os.system("rm '{filename}'")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Plot predictions 
def plot_samples(pred,sample,loss,epoch,batch,dir) -> None:
    true_images = [sample[i].detach().numpy().reshape(64,64) for i in range(3)]
    pred_images = [pred[i].detach().numpy().reshape(64,64) for i in range(3)]
    fig, axes = plt.subplots(3,2,figsize=(6,6),dpi=200,tight_layout=True,gridspec_kw={'wspace': -0.20, 'hspace': 0.08})
    for ix, ax in enumerate(axes):
        minv, maxv = np.min(true_images[ix]), np.max(true_images[ix])
        pred_plot = ax[0].imshow(pred_images[ix])
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.colorbar(pred_plot,ax=ax[0])
        pred_plot.set_clim(minv,maxv)
        true_plot = ax[1].imshow(true_images[ix])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.colorbar(true_plot,ax=ax[1])
        true_plot.set_clim(minv,maxv)
    plt.suptitle(f"=== Epoch: {epoch} === Batch: {batch} === Loss: {loss}")
    plt.savefig(os.path.join(dir,f"epoch_{epoch}_batch_{batch}_loss_{loss}.png"))

def dataloader(input_shape,batch_size) -> torch.utils.data.DataLoader:

    # MNIST DATA
    traindata = MNIST('./digits/',download=True,transform=transforms.Compose([transforms.Resize((input_shape[-2],input_shape[-1])),\
                                                                transforms.ToTensor()]))
    trainloader = DataLoader(traindata,batch_size,True)

    return trainloader
    
def configure_model(key,input_shape,latent_dim) -> autoencoder.Autoencoder:
    
    # Initialise base encoder
    base = model_dict[key]()

    # Skip 'downsample' layers in Resnet architecture
    if 'res' in key:
        encoder = parsing._parse_torchvision_model(base,['downsample'])
    else:
        encoder = parsing._parse_torchvision_model(base)

    # Modify encoder
        # Modify input layer 
    input_params = parsing._extract_params(encoder[0])
    input_params['in_channels'] = input_shape[0]
    encoder[0] = nn.Conv2d(**input_params) 

        # Modify output layer 
    output_layer_params = parsing._extract_params(encoder[-1])
    output_layer_params['out_features'] = latent_dim
    encoder[-1] = nn.Linear(**output_layer_params)
    
    # Create decoder 
    decoder = build.build_decoder(encoder,input_shape)

    # Create autoencoder
    auto = autoencoder.Autoencoder(encoder,decoder)

    return auto 

def config_optim(model,lr,betas,decay):
    return torch.optim.Adam(model.parameters(),lr=lr,betas=betas,weight_decay=decay)

def train(model,loader,device_ids,logger,plot_freq,epochs,lr,betas,decay):
    # Use MSE for reconstruction loss by default
    criterion = nn.MSELoss()

    # Configure optimizer based 
    optim = config_optim(model,lr,betas,decay)

    # Training loop
    print("beginning training...")
    for epoch in range(epochs):
        print(f"Epoch: {epoch}...")
        for ix, (x,y) in enumerate(loader):

            # Zero-out gradients
            optim.zero_grad()
            
            # Send input to GPU:0 for parallelisation 
            xgpu = x.to(device_ids[0])
            out = model.forward(xgpu)

            # backward pass - will automatically occurr on GPU because out and xgpu exist on GPU:0
            loss = criterion(out,xgpu)
            loss.backward()
            optim.step()

            # logging/tracking 
            if ix % plot_freq == 0:
                logger.info(f"\tEpoch: {epoch} == Batch: {ix} == MSE: {loss.item():.4f}")
                # Plot samples - must send data back to cpu 
                plot_samples(out.cpu(),xgpu.cpu(),loss.item(),epoch,ix,'images')

    return model

def main():

    global args
    args = parser.parse_args()

    # Initialise logger 
    logger = get_logger('./logs',f'train_{args.arch}_z_{args.latent_dim}.log')

    # Initialise device_ids 
    device_ids = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

    # Input shape 
    input_shape = (1,64,64)

    # data and preprocessing
    loader = dataloader(input_shape,args.b)
    logger.info("\ndata loaded...")

    # load model
    auto = configure_model(args.arch,input_shape,args.latent_dim)
    if 'res' in args.arch:
        # For some as-yet unknown, but not undiagnosed, reason, resnet models return a reconstruction 
        # that is of shape (H-1 x W-1) where the target should be (H x W)
        auto.decoder.append(nn.ConvTranspose2d(1,1,5,padding=2))

    # wrap model in an DataParallel object
    auto_parallel = nn.DataParallel(auto,device_ids=device_ids)
    
    # Send model to GPU:0 - this acts as a master and co-ordinates the data parellelism 
    # The forward pass is still distributed across the full list of devices - `device_ids`
    auto_parallel.to(device_ids[0])

    # Train the model 
    logger.info("\nmodel loaded...")
    trained_model = train(auto_parallel,loader,device_ids,logger,args.plot_freq,\
        args.epochs,args.lr,args.betas,args.weight_decay)   
    return trained_model

if __name__=="__main__":

    """
        Demonstrates an example of data-parellel autoencoder training on MNIST dataset for an arbitrary 
        Torchvision architecture, e.g., VGG, Resnet, Alexnet,. 

        The script will automatically use the maximum number of GPUs available as obtained through `device_ids`. 

        Argparser is utilised to modify architectures and parameters. 

        A simple model can be run using `python mnist.py --arch=vgg11` with the remaining parameters set to default.

    """
    model = main()

