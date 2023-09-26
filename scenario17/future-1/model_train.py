'''
The Wireless Intelligence Lab

DeepSense 6G
http://www.deepsense6g.net/

Description: Training script for vision-aided blockage prediction task
Author: Gouranga Charan
Date: 10/29/2021
'''

import os
import time

import torch
import torch.nn as nn
import torch.optim as optimizer

import csv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

def modelTrain(net, trn_loader, val_loader, options_dict):
    """
    Training function for the blockage prediction model.
    
    Args:
    - net (nn.Module): The neural network model to train.
    - trn_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    - options_dict (dict): Dictionary of training options.

    Returns:
    - Tuple: A tuple containing the trained model and the updated options dictionary.
    """
    # Optimizer:
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
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0

    # Initialize training variables:
    itr = 0
    embed = nn.Embedding(options_dict['cb_size'], options_dict['embed_dim'])
    running_train_loss = []
    running_trn_top_1 = []
    running_val_top_1 = []
    train_loss_ind = []
    val_acc_ind = []
    train_acc = []
    val_acc = []
    best_acc = 0

    def save_checkpoint(state, filename='checkpoint/bbox_blockage_pred_img.pth.tar'):
        torch.save(state, filename)

    if not os.path.exists('checkpoint'):
      os.makedirs('checkpoint')

    print('------------------------------- Commence Training ---------------------------------')
    t_start = time.time()
    
    for epoch in range(start_epoch, start_epoch + options_dict['num_epochs']):

        net.train()
        h = net.initHidden(options_dict['batch_size'])
        h = h.cuda()

        # Training:
        for batch, (bbox, label) in enumerate(trn_loader):
            itr += 1   
            bbox = bbox[:, :options_dict['inp_seq']].float()           
            bbox = bbox.cuda()
            batch_size = label.shape[0]
            targ = label.type(torch.LongTensor)
            targ = targ.view(-1)
            targ = targ.cuda()
         
            h = h.data[:,:batch_size,:].contiguous().cuda()
            opt.zero_grad()
            out, h = net.forward(bbox, h)
            out = out.view(-1,out.shape[-1])
            train_loss = criterion(out, targ)  # (pred, target)
            train_loss.backward()
            opt.step()        
            out = out.view(batch_size,options_dict['out_seq'],options_dict['out_dim'])
            pred_beams = torch.argmax(out,dim=2)
            targ = targ.view(batch_size,options_dict['out_seq'])
            top_1_acc = torch.sum( torch.prod(pred_beams == targ, dim=1, dtype=torch.float) ) / targ.shape[0]
            
            if np.mod(itr, options_dict['coll_cycle']) == 0:
                running_train_loss.append(train_loss.item())
                running_trn_top_1.append(top_1_acc.cpu().numpy())
                train_loss_ind.append(itr)
                train_acc.append(top_1_acc.item())
                
            if np.mod(itr, options_dict['display_freq']) == 0:
                print(
                    'Epoch No. {0}--Iteration No. {1}-- Mini-batch loss = {2:10.9f} and Top-1 accuracy = {3:5.4f}'.format(
                    epoch + 1,
                    itr,
                    train_loss.item(),
                    top_1_acc.item())
                    )

            # Validation:
            # -----------
            test_list = []
            pred_list = []
            
            out_lst = []
            
            net.eval()
            batch_score = 0
            
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
                    batch_score += torch.sum( torch.prod( pred_beams == targ, dim=1, dtype=torch.float ) )
                    tmp_list = [targ.detach().cpu().numpy()[0][0], pred_beams.detach().cpu().numpy()[0][0] ]
                    out_lst.append(tmp_list)
                
                with open("./analysis/pred_blk_val_%s.csv"%epoch, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(out_lst)
                
                   
                running_val_top_1.append(batch_score.cpu().numpy() / options_dict['test_size'])
                val_acc_ind.append(itr)
                print('Validation-- Top-1 accuracy = {0:5.4f}'.format(
                    running_val_top_1[-1])
                )
                val_acc.append(running_val_top_1)                  
                
                print(confusion_matrix(test_list, pred_list))
                print("F1-score:",f1_score(test_list, pred_list))

                _f1_score = f1_score(test_list, pred_list)
                if _f1_score > best_acc:
                    best_acc = _f1_score
                    save_checkpoint({
                         'epoch': epoch + 1,
                         'gru_state_dict': net.state_dict(),
                         'optimizer': opt.state_dict(),
                         'embedding': embed,
                    })
                else:
                    best_acc = best_acc
            net.train()

        current_lr = scheduler.get_lr()[-1]
        scheduler.step()
        new_lr = scheduler.get_lr()[-1]
        if new_lr != current_lr:
            print('Learning rate reduced to {}'.format(new_lr))
    
    print('------------------------ Training Done ------------------------')

    return [net, options_dict]
