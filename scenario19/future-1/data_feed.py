"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Evaluation script for vision aided blockage prediction task
Author: Gouranga Charan
Date: 10/29/2021
"""

import os
import numpy as np
import pandas as pd
import torch
import random
from skimage import io
from torch.utils.data import Dataset
import ast

############### Create data sample list #################
def create_samples(root, shuffle=False, nat_sort=False):
	f = pd.read_csv(root)
	data_samples = []
	beam_samples = []
	for idx, row in f.iterrows():
		bboxes = row.values[:8]
		data_samples.append(bboxes)
		beams = row.values[8:9]
		beam_samples.append(beams)

	print('list is ready')
	return data_samples, beam_samples


#########################################################

class DataFeed(Dataset):
    """
    A class fetching a PyTorch tensor of beam indices.
    """
    
    def __init__(self, root_dir,
    			n,
    			transform=None,
    			init_shuflle=True):
    
        self.root = root_dir
        self.samples, self.pred_val = create_samples(self.root, shuffle=init_shuflle)
        self.transform = transform
        self.seq_len = n
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx] # Read one data sample
        beam_val = self.pred_val[idx]
        sample = sample[:self.seq_len]
        bbox_val = torch.zeros((self.seq_len,30))
        beam_val = beam_val[:1] # Read a sequence of tuples from a sample
        beams = torch.zeros((1,))
        for i,s in enumerate( beam_val ):
            x = s # Read only beams
            beams[i] = torch.tensor(x, requires_grad=False)
        	
        for i,s in enumerate(sample):
            data = s 
            bbox_data = ast.literal_eval(data)           
            bbox_val[i] = torch.tensor(bbox_data, requires_grad=False)		
        
        return (bbox_val, beams)
