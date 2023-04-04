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
import torch.optim as optimizer
# from torch.utils.data import DataLoader
import numpy as np
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
#from pandas_ml import ConfusionMatrix

import pandas as pd

def modelTrain(net,val_loader,options_dict):
    """

    :param net:
    :param data_samples:
    :param options_dict:
    :return:
    """
    # Optimizer:
    # ----------
    if options_dict['solver'] == 'Adam':
        opt = optimizer.Adam(net.parameters(),
                             lr=options_dict['lr'],
                             weight_decay=options_dict['wd'],
                             amsgrad=True)
    else:
        ValueError('Not recognized solver')

    scheduler = optimizer.lr_scheduler.MultiStepLR(opt,
                                                   milestones=options_dict['lr_sch'],
                                                   gamma=options_dict['lr_drop_factor'])

    # Define training loss:
    # ---------------------
    criterion = nn.CrossEntropyLoss()
    
    start_epoch = 0
    
    if options_dict['val']:
        print("------------------------------- Loading trained parameters -------------------------------")
        checkpoint = torch.load('./checkpoint/scenario17_vision_blockage-prediction_future-1.pth.tar')
        start_epoch = checkpoint['epoch']
        embed = checkpoint['embedding']
        net.load_state_dict(checkpoint['gru_state_dict'])    
    else:
        None

    # Initialize training hyper-parameters:
    # -------------------------------------
    itr = 0
    running_train_loss = []
    running_trn_top_1 = []
    running_val_top_1 = []
    train_loss_ind = []
    val_acc_ind = []


    print('------------------------------- Commence Testing ---------------------------------')
    t_start = time.time()

    # Validation:
    # -----------
    test_list = []
    pred_list = []
    
    pred_seq_correct = []
    pred_seq_wrong = []
    out_lst = []

    net.eval()
    batch_score = 0
    _acc_score = 0
    _recall_score = 0
    _precision_score = 0
    _f1_score = 0

    with torch.no_grad():
        for v_batch, (bbox, label) in enumerate(val_loader):
            blockage_seq = []
            init_beams = label.type(torch.LongTensor)
            lst_init_beams = (init_beams.float()).detach().cpu().numpy()
            blockage_seq.append(lst_init_beams.tolist())
            bbox = bbox[:, :options_dict['inp_seq']].float()            
            bbox = bbox.cuda()
            batch_size = label.shape[0]
            targ = label.type(torch.LongTensor)
            targ = targ.view(batch_size,options_dict['out_seq'])
            targ = targ.cuda()
            h_val = net.initHidden(batch_size).cuda()

            out, h_val = net.forward(bbox, h_val)

            pred_beams = torch.argmax(out, dim=2)
            test_list.append(targ.detach().cpu().numpy()[0][0])
            pred_list.append(pred_beams.detach().cpu().numpy()[0][0])
            _targ = targ.float()
            _targ = _targ.detach().cpu().numpy()
            _pred_beams = pred_beams.float()
            _pred_beams = _pred_beams.detach().cpu().numpy()
            print(_pred_beams)
            batch_score += torch.sum( torch.prod( pred_beams == targ, dim=1, dtype=torch.float ) )
            tmp_list = [targ.detach().cpu().numpy()[0][0], pred_beams.detach().cpu().numpy()[0][0] ]
            out_lst.append(tmp_list)
   
        running_val_top_1.append(batch_score.cpu().numpy() / options_dict['test_size'])
        val_acc_ind.append(itr)
        print('Validation-- Top-1 accuracy = {0:5.4f}'.format(
            running_val_top_1[-1])
        )    
        print(confusion_matrix(test_list, pred_list))
        print("F1-score:",f1_score(test_list, pred_list))
        
        df = pd.DataFrame()
        df['gt_val'] = test_list
        df['pred_val'] = pred_list
        
        df.to_csv(r'../test_pred.csv', index=False)
        




    
    
    
    t_end = time.time()
    train_time = (t_end - t_start)/60
    print('Validation lasted {0:6.3f} minutes'.format(train_time))
    print('------------------------ Testing Done ------------------------')

    return [net, options_dict]
