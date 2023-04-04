"""
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Evaluation script for vision aided blockage prediction task
Author: Gouranga Charan
Date: 10/29/2021
"""
import torch
import torch.nn as nn

# Beam prediction model relying on input beam sequences alone
class RecNet(nn.Module):
    def __init__(self,
                 inp_dim,
                 hid_dim,
                 out_dim,
                 out_seq,
                 num_layers,
                 drop_prob=0.2):
        super(RecNet, self).__init__()
        self.hid_dim = hid_dim
        self.out_seq = out_seq
        self.num_layers = num_layers

        # Define layers
        self.gru = nn.GRU(inp_dim,hid_dim,num_layers,batch_first=True,dropout=drop_prob)
        self.classifier = nn.Linear(hid_dim,out_dim)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)--> Softmax is implicitly implemented into the cross entropy loss

    def forward(self,x,h):
        out, h = self.gru(x,h)
        out = self.relu(out[:,-1*self.out_seq:,:])
        y = self.classifier(out)
        return [y, h]

    def initHidden(self,batch_size):
        return torch.zeros( (self.num_layers,batch_size,self.hid_dim))


