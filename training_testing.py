import torch
import torch.nn.functional as F

import numpy as np

from losses import *

def training_loop(model, optimizer, source_dl, target_dl, e, epochs,device, adaptation = 'fasttikuda'):
    
    train_loss = []

    p = e/epochs
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    # Training loop
    for idx, ((input_source, label_source), (input_target, label_target)) in (enumerate(zip(source_dl,target_dl))):

        input_source = input_source.float().to(device)
        input_target = input_target.float().to(device)

        label_source = label_source.float().squeeze().to(device)[:,-1]

        optimizer.zero_grad()

        out_s, _, feat_s = model(input_source,device)
        out_t, _, feat_t = model(input_target,device)


        feat_s = feat_s.flatten(start_dim = 1)
        feat_t = feat_t.flatten(start_dim = 1)
        
        if(adaptation == 'tik'):
            loss = F.mse_loss(out_s.squeeze(), label_source) + alpha* TIK_LOSS(feat_s,feat_t,device=device)
        elif(adaptation == 'gram'):
            loss = F.mse_loss(out_s.squeeze(), label_source) + alpha* DARE_GRAM(feat_s,feat_t,device=device)
        elif(adaptation == 'fasttikuda'):
            loss = F.mse_loss(out_s.squeeze(), label_source) + alpha* Fast_TikUDA(feat_s,feat_t,device=device)
        elif(adaptation == 'nothing'):
            loss = F.mse_loss(out_s.squeeze(), label_source)
        else:
            print('Adaptation loss not implemented')
        
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    return  np.array(train_loss).mean()

def testing_loop(model,test_dl,device):
    
    model.eval()
    
    with torch.inference_mode():
        pred = []
        gt = []

        for batch_idx, (x,y) in enumerate((test_dl)):
            x = x.to(device).float()
            y = y.to(device).float()

            recons = model(x, device)[0]

            pred.append(recons.detach().cpu().numpy())
            gt.append(y.detach().cpu().numpy())

            recon_loss = F.mse_loss(y.squeeze()[-1], recons.squeeze())

        pred = np.concatenate(pred, axis=0).squeeze()
        
        gt = np.concatenate(gt, axis=0)
        gt= np.array(gt[:,-1])
    return pred,gt
