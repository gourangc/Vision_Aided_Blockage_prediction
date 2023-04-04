"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Evaluation script for vision aided blockage prediction task
Author: Gouranga Charan
Date: 10/29/2021
"""
import torch
import sys 

from build_net import RecNet
from model_train_eval import modelTrain
from data_feed import DataFeed
import torchvision.transforms as trf
from torch.utils.data import DataLoader
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage import io, transform
import sklearn


options_dict = {
    'tag': 'Exp1_beam_seq_pred_no_images',
    'operation_mode': 'beams',

    # Data:
    'train_ratio': 1,
    'test_ratio': 1,
    'img_mean': (0.4905,0.4938,0.5285),
    'img_std':(0.05922,0.06468,0.06174),
    'val_data_file': 'scenario19_dev_series_test_final.csv',

    # Net:
    'net_type':'gru',
    'cb_size': 2,  # Beam codebook size
    'out_seq': 1,  # Length of the predicted sequence
    'inp_seq': 8, # Length of inp beam and image sequence
    'embed_dim': 30,  # Dimension of the embedding space (same for images and beam indices)
    'hid_dim': 128,  # Dimension of the hidden state of the RNN
    'out_dim': 2,  # Dimensions of the softmax layers
    'num_rec_lay': 3,  # Depth of the recurrent network
    'drop_prob': 0.5,

    # Train param
    'gpu_idx': 0,
    'solver': 'Adam',
    'shf_per_epoch': True,
    'num_epochs': 150,
    'batch_size':128,
    'val_batch_size':1,
    'lr': 1e-3,
    'lr_sch': [160],
    'lr_drop_factor':0.1,
    'wd': 0,
    'display_freq': 10,
    'coll_cycle': 10,
    'val_freq': 20,
    'prog_plot': False,
    'fig_c': 0,
    'val': True,
    'resume_train': False
}


#for i in range(1006):

# Fetch training data


transf = trf.Compose([
    trf.ToTensor(),
])

val_feed = DataFeed(root_dir=options_dict['val_data_file'],
                     n=options_dict['inp_seq']+options_dict['out_seq'],
                     transform=transf)
val_loader = DataLoader(val_feed,batch_size=1)
options_dict['test_size'] = val_feed.__len__()

dataset_sizes = {
    'valid': len(val_loader.dataset)
}
print(dataset_sizes)

with torch.cuda.device(options_dict['gpu_idx']):

    # Build net:
    # ----------
    if options_dict['net_type'] == 'gru':
        net = RecNet(options_dict['embed_dim'],
                     options_dict['hid_dim'],
                     options_dict['out_dim'],
                     options_dict['out_seq'],
                     options_dict['num_rec_lay'],
                     options_dict['drop_prob'],
                     )
        net = net.cuda()
        


    # Train and test:
    # ---------------
    net, options_dict = modelTrain(net,
                                   val_loader,
                                   options_dict)



