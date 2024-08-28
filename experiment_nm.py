import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F

import numpy as np
import torchvision
import progress.bar
import utils
import math
import time
from torchvision import transforms
# import model
import gc

import os
import time
import ipdb
import hist_loss
import cv2
import NMcontrol

def set_mode(mode, model):
    """ Set the network to train/eval mode. Affects the dropout and batchnorm. """
    if mode == 'train':
        model.train()
        print('\n{:-^50}'.format(' Network Mode '))
        print("Network now in '{}' mode.".format(mode))
        print('-' * 50 + '\n')
    elif mode == 'eval':
        model.eval()
        print('\n{:-^50}'.format(' Network Mode '))
        print("Network now in '{}' mode.".format(mode))
        print('-' * 50 + '\n')
    else:
        raise ValueError(
            "Invalid mode '{}'. Valid options are 'train' and 'eval'.".format(mode))
   

def NM_optimization(hdrs, EVs, loss_function):
    EV_nm = []
    for (k,EV) in enumerate(EVs):
        with torch.no_grad():
            hdr_clamp = (torch.clamp(hdrs[k] * (2**EV), min = 0.0, max = 2**12-1)/(2**EV)).unsqueeze(0)
            x0 = NMcontrol.initial(hdr_clamp, EV)   
            ev_nm, _ = NMcontrol.nelder_mead_AE(x0, loss_function, hdr_clamp, 5) 
            EV_nm.append(ev_nm)
    EV_nm = torch.stack(EV_nm)
    return EV_nm

def set_data(hdr0, hdr, EV0, EV, EV_nm0, EV_nm, opts):
    """ Set the input tensors. """
    B, _, W, H = hdr0.size()

    
    ldr0 = hist_loss.gamma(hdr0, EV0)
    gray0 = hist_loss.rgb2gray(ldr0)
    hist0 = hist_loss.multiScaleHist(gray0)   
    
    ldr = hist_loss.gamma(hdr, EV)
    gray = hist_loss.rgb2gray(ldr)
    hist = hist_loss.multiScaleHist(gray)   
   
    B, _, n_bins = hist.size()
    adjusted_expt0 = ((EV0-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))
    adjusted_expt = ((EV-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))
    exp_t0 = adjusted_expt0.view(-1, 1, 1).expand((B, 1, n_bins))
    exp_t = adjusted_expt.view(-1, 1, 1).expand((B, 1, n_bins))
    
    adjusted_expt_nm0 = ((EV_nm0-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))
    adjusted_expt_nm = ((EV_nm-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))
    expt_nm0 = adjusted_expt_nm0.view(-1, 1, 1).expand((B, 1, n_bins))
    expt_nm = adjusted_expt_nm.view(-1, 1, 1).expand((B, 1, n_bins))
    
        
    new_img0 = torch.cat((hist0, exp_t0, expt_nm0), 1)
    new_img = torch.cat((hist, exp_t, expt_nm), 1)
    images = torch.cat((new_img0, new_img),1)
    return images
 
def forward_hist_nm(opts, model, data, test_flag = False):
    """ Evaluate the forward pass of the parameter estimator model. """
    loss_sum_EV = torch.FloatTensor([0]).to(opts.device)
    # loss_sum_hist = torch.FloatTensor([0]).to(opts.device)
    hist_loss_function = hist_loss.hist_loss()

    if opts.gru:
        h = model.init_hidden(opts.batch_size)
    else:
        h = None

    EV0 = data['target'][:,0].to(opts.device) # B
    EV1 =  data['target'][:,1].to(opts.device)
    EV_nm0 = data['target'][:,0].to(opts.device) # B
    EV_nm1 =  data['target'][:,1].to(opts.device)
    
    # EV_seq = [EV0, EV]
    EV_seq = []
    # loss_seq = []
    EV_nm_seq = []
    hdr_seq = data['img_seq'][2:] # a list
    hdr0 = data['img_seq'][0].to(opts.device) # [B, C, W, H]
    hdr1 = data['img_seq'][1].to(opts.device)
    image_data = set_data(hdr0, hdr1, EV0, EV1, EV_nm0, EV_nm1, opts)

    len_seq = len(hdr_seq)
    EV_loss_function = hist_loss.hist_loss()

    for (k, hdr2) in enumerate(hdr_seq): #k in range(len_seq):      
        EV2, hdr_hist, h = model(image_data, h = h)
        hdr2 = hdr2.to(opts.device)

        if opts.loss_function == 'hist':
            loss_EV  = EV_loss_function(hdr2, EV2)
        elif opts.loss_function == 'l2':
            loss_function = nn.MSELoss()
            target = torch.transpose(data['target'][:,2:], 0, 1).to(opts.device)
            loss_EV = loss_function(EV2, target)
        loss_sum_EV += loss_EV
       
        EV_seq.append(EV2)
        # loss_seq.append(loss)
              
        EV_nm2 = NM_optimization(hdr2, EV2.clone().detach(), hist_loss_function)
        EV_nm_seq.append(EV_nm2)

        EV0 = EV1
        EV1 = EV2
        EV_nm0 = EV_nm1
        EV_nm1 = EV_nm2
        
        hdr0 = hdr1 # should normalize the image 
        hdr1 = hdr2
        image_data = set_data(hdr0, hdr1, EV0, EV1, EV_nm0, EV_nm1, opts)
    
    EV_nm_seq = torch.cat(EV_nm_seq)
    EV_seq = torch.cat(EV_seq)
    # return (loss_sum_EV + loss_sum_hist)/len_seq, EV_seq, EV_nm_seq, loss_sum_EV/len_seq, loss_sum_hist/len_seq
    return (loss_sum_EV)/len_seq, EV_seq, EV_nm_seq


def forward_ldr_nm(opts, model, data, test_flag = False):
    """ Evaluate the forward pass of the parameter estimator model. """
    loss_sum_hist = torch.FloatTensor([0]).to(opts.device)
    hist_loss_function = hist_loss.hist_loss()

    if opts.gru:
        h = model.init_hidden(opts.batch_size)
    else:
        h = None

    EV0 = data['target'][:,0].to(opts.device) # B
    EV1 =  data['target'][:,1].to(opts.device)
    EV_nm0 = data['target'][:,0].to(opts.device) # B
    EV_nm1 =  data['target'][:,1].to(opts.device)
    
    # EV_seq = [EV0, EV]
    EV_seq = []
    EV_nm_seq = []
    hdr_seq = data['img_seq'][2:] # a list
    hdr0 = data['img_seq'][0].to(opts.device) # [B, C, W, H]
    hdr1 = data['img_seq'][1].to(opts.device)
    image_data = set_data(hdr0, hdr1, EV0, EV1, EV_nm0, EV_nm1, opts)

    len_seq = len(hdr_seq)
    EV_loss_function = hist_loss.hist_loss()
    loss_hist_seq = []
    for (k, hdr2) in enumerate(hdr_seq): #k in range(len_seq):      
        EV2, ldr_hist, h = model(image_data, h = h)

        hdr2 = hdr2.to(opts.device)
        target = data['target'][:,k+2].to(opts.device)
        # hdr_log2 = torch.log2(hdr2)
        ldr_hist_gt = []
        for h2 in hdr2:
            l2 = hist_loss.rgb2gray(hist_loss.gamma(h2.unsqueeze(0), target))[0]
            ldr_hist_gt.append(torch.cat(EV_loss_function.hist_region(l2, block_size = 1)))
        ldr_hist_gt = torch.cat(ldr_hist_gt)
        loss_hist = torch.mean(1 - F.cosine_similarity(ldr_hist, ldr_hist_gt, dim=1))
        # loss_sum_hist += loss_hist
        loss_hist_seq.append(loss_hist)
        EV_seq.append(EV2)
        # loss_seq.append(loss)
              
        EV_nm2 = NM_optimization(hdr2, EV2.clone().detach(), hist_loss_function)
        EV_nm_seq.append(EV_nm2)

        EV0 = EV1
        EV1 = EV2
        EV_nm0 = EV_nm1
        EV_nm1 = EV_nm2
        
        hdr0 = hdr1 # should normalize the image 
        hdr1 = hdr2
        image_data = set_data(hdr0, hdr1, EV0, EV1, EV_nm0, EV_nm1, opts)
    
    EV_nm_seq = torch.cat(EV_nm_seq)
    EV_seq = torch.cat(EV_seq)
    loss_hist_seq = torch.stack(loss_hist_seq)
    return torch.mean(loss_hist_seq), EV_seq, EV_nm_seq, loss_hist_seq


def set_data_seq(img0, img, expt0, expt, EV_nm0, EV_nm, opts): #what about batch
    """ Set the input tensors. """
    B = img0.size(0)

    adjusted_expt0 = ((expt0-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))
    adjusted_expt = ((expt-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))
    exp_t0 = adjusted_expt0.view(-1, 1, 1, 1).expand((B, 1, 224, 224))
    exp_t = adjusted_expt.view(-1, 1, 1, 1).expand((B, 1, 224, 224))
    
    adjusted_expt_nm0 = ((EV_nm0-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))
    adjusted_expt_nm = ((EV_nm-opts.min_exposure)/(opts.max_exposure-opts.min_exposure))
    expt_nm0 = adjusted_expt_nm0.view(-1, 1, 1, 1).expand((B, 1, 224, 224))
    expt_nm = adjusted_expt_nm.view(-1, 1, 1, 1).expand((B, 1, 224, 224))
    
    new_img0 = torch.cat((img0, exp_t0, expt_nm0), 1)
    new_img = torch.cat((img, exp_t, expt_nm), 1)
    images = torch.cat((new_img0, new_img),1)

    return images


def forward_cnn(opts, model, data):
    """ Evaluate the forward pass of the parameter estimator model. """
    loss = torch.FloatTensor([0]).to(opts.device)
    hist_loss_function = hist_loss.hist_loss()

    EV0 = data['target'][:,0].to(opts.device) # B
    EV1 =  data['target'][:,1].to(opts.device)
    hdr0 = data['img_seq'][0].to(opts.device) # [B, C, W, H]
    hdr1 = data['img_seq'][1].to(opts.device)
    img0 = hist_loss.gamma(hdr0, EV0)
    img1 = hist_loss.gamma(hdr1, EV1) # [B, C, W, H]

    EV_nm0 = EV0
    EV_nm1 = EV1
    images = set_data_seq(img0, img1, EV0, EV1, EV_nm0, EV_nm1, opts)

    EV_seq = []#[EV0, EV]
    # loss_seq = []
    EV_nm_seq = []
    hdr_seq = data['img_seq'][2:] # a list
    len_seq = len(hdr_seq)
    for (k, hdr2) in enumerate(hdr_seq): #k in range(len_seq):
        hdr2 = hdr2.to(opts.device)        
        EV2 = model.forward(images) # [B, 1] 
        if opts.loss_function == 'hist': 
            loss_function = hist_loss.hist_loss()
            loss_p, _ = loss_function(hdr2, EV2)

        elif opts.loss_function == 'l2':
            loss_function = nn.MSELoss()
            target = torch.transpose(data['target'][:,2:], 0, 1).to(opts.device)
            loss_p = loss_function(EV2, target) #??
            
        loss += loss_p 
        EV_seq.append(EV2)
        # loss_seq.append(loss_p)
       
        if opts.gtNM:
            EV_nm2 = data['target'][:,k+2].unsqueeze(1).to(opts.device)
        else:
            EV_nm2 = NM_optimization(hdr2, EV2.clone().detach(), hist_loss_function)

        EV_nm_seq.append(EV_nm2)
        EV0 = EV1
        EV1 = EV2
        EV_nm0 = EV_nm1
        EV_nm1 = EV_nm2
        
        img0 = img1 
        img1 = hist_loss.gamma(hdr2, EV2)
        images = set_data_seq(img0, img1, EV0, EV1, EV_nm0, EV_nm1, opts)
    
    EV_seq = torch.cat(EV_seq)
    EV_nm_seq = torch.cat(EV_nm_seq)

    return loss/len_seq, EV_seq, EV_nm_seq


def orb_loss(img_cv2, hdr, param):
    
    img1 = img_cv2
    img2 = hist_loss.gamma(hdr, param)
    img2_norm = img2[0].cpu().detach().numpy().transpose(1, 2, 0)
    img2 = (img2_norm*255).astype(np.uint8)
    
    orb = cv2.ORB_create(nfeatures=5000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    mask = []
    if len(kp2)>0:
        orb_matches = bf.match(des1, des2)
        kp1_coords = np.asarray([kp1[m.queryIdx].pt for m in orb_matches]).reshape(-1,1,2)
        kp2_coords = np.asarray([kp2[m.trainIdx].pt for m in orb_matches]).reshape(-1,1,2)
        _, mask = cv2.findFundamentalMat(kp1_coords, kp2_coords, method=cv2.FM_RANSAC, ransacReprojThreshold=3.0)
    if mask is None:
        matches=[]
        score = 0
    else:
        score = len(mask)
        matches = [orb_matches[i] for (i, good) in enumerate(mask) if good >0]
    matched_image = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2) 
    return score, matched_image, img2
    

def model_test_orb(opts, data, EVs):
    """ Evaluate the forward pass of the parameter estimator model. """
    with torch.no_grad():
        score_seq = []
        matched_image_seq = []
        img_seq = data['img_seq'][2:]
        hdr = img_seq[0].to(opts.device)
        ldr = hist_loss.gamma(hdr, EVs[0])
        ldr = ldr[0].cpu().detach().numpy().transpose(1, 2, 0)
        ldr = (ldr*255).astype(np.uint8)       
        
        for k in range(1, len(img_seq)):      
            hdr = img_seq[k].to(opts.device)
            score, matched_image, ldr = orb_loss(ldr, hdr, EVs[k])
            matched_image_seq.append(matched_image)
            score_seq.append(score)
            
    return score_seq, matched_image_seq


def model_test(opts, model, loss_function, data):
    """ Evaluate the model and test loss without optimizing. """
    with torch.no_grad():
        loss_hist = torch.FloatTensor([0])
        loss_EV, EVs, EVs_nm = forward_hist_nm(opts, model, data, True)
        if opts.joint_learning:
            loss_hist, EVs, EVs_nm, _ = forward_ldr_nm(opts, model, data, True)

    return EVs, EVs_nm, loss_EV, loss_hist

def get_errors(loss_p):
    """ Return a dictionary of the current errors. """
    error_dict = {'Loss_p': loss_p.item()}

    return error_dict


def save_checkpoint(epoch, label, opts, model):
    """ Save model to file. """
    cur_dir = os.getcwd()
    model_dir = os.path.join(cur_dir, opts.results_dir, opts.experiment_name, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

    model_dict = {'epoch': epoch,
                    'label': label,
                    'state_dict': model.state_dict()}

    print("Saving model to {}".format(model_file))
    torch.save(model_dict, model_file)

def load_checkpoint(opts, model, label):
    """ Load a model from a file. """
    cur_dir = os.getcwd()
    model_dir = os.path.join(cur_dir, opts.results_dir, opts.experiment_name, 'checkpoints')
    model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

    print("Loading model from {}".format(model_file))
    model_dict = torch.load(model_file, map_location=opts.device)

    model.to(opts.device)
    model.load_state_dict(model_dict['state_dict'])

    return model

def load_checkpoint2(opts, model, model_dir, label):
    """ Load a model from a file. """
    model_dir = os.path.join(model_dir, 'checkpoints')
    model_file = os.path.join(model_dir, '{}_net.pth.tar'.format(label))

    print("Loading model from {}".format(model_file))
    model_dict = torch.load(model_file, map_location=opts.device)

    model.to(opts.device)
    model.load_state_dict(model_dict['state_dict'])

    return model


def train(opts, model, train_data, val_data, num_epochs, resume_from_epoch=None):
    train_loader = DataLoader(train_data,
                              batch_size=opts.batch_size,
                              shuffle=True,
                              num_workers=opts.dataloader_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=opts.dataloader_workers,
                            pin_memory=True)

    if os.path.exists(os.path.join(opts.results_dir, opts.experiment_name, 'training')):
        previous_runs = os.listdir(os.path.join(opts.results_dir, opts.experiment_name, 'training'))
        if len(previous_runs) == 0:
            run_number = 1    
        else:
            run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1
    else:
        run_number = 1
    
    log_dir_num = 'run_%02d' % run_number
    print("Currently on run #: ", run_number)
    log_learning_rate = 'lr_{}'.format(opts.lr)
    log_batch_size = 'batch_{}'.format(opts.batch_size)

    train_log_dir = os.path.join(opts.results_dir, opts.experiment_name, 'training', log_dir_num, log_learning_rate, log_batch_size)#, log_loss_type, log_loss_formulation, log_method, log_normalization, log_compensating)
    val_log_dir = os.path.join(opts.results_dir, opts.experiment_name, 'validation', log_dir_num, log_learning_rate, log_batch_size)#, log_loss_type, log_loss_formulation, log_method, log_normalization, log_compensating)
    train_writer = SummaryWriter(train_log_dir)
    val_writer = SummaryWriter(val_log_dir)

    

    opts.save_txt('config.txt', log_dir_num)

    ### Load from Checkpoint
    if resume_from_epoch is not None:
        try:
            initial_epoch = model.load_checkpoint(resume_from_epoch) + 1
            iterations = (initial_epoch -1)*opts.batch_size
        except FileNotFoundError:
            print('No model available for epoch {}, starting fresh'.format(resume_from_epoch))
            initial_epoch = 1
            iterations = 0

    else:
        initial_epoch = 1
        iterations = 0

    ### !! TRAIN AND VALIDATE ###
    if opts.jobs == 1:
        opts.best_model = 1e12

    # MODEL PARAMETERS
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    loss_function = hist_loss.hist_loss()

    for epoch in range(initial_epoch, num_epochs + 1):
        epoch_start = time.perf_counter()
        
      
        # TRAIN
        epoch_train_loss = None
        set_mode('train', model)

        bar = progress.bar.Bar('Epoch {} train'.format(epoch), max=len(train_loader))

        for i,data in enumerate(train_loader):
            iters = len(train_loader) * (epoch-1) + i + 1
            max_iters = opts.train_epochs * len(train_loader)

            poly_learning_rate(optimizer, opts.lr, iters, max_iters, index_split=100000)
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            # if not opts.cnn: 
            if opts.joint_learning:
                if epoch %2 ==0:
                    loss, EVs, _ = forward_hist_nm(opts, model, data)
                else:
                    loss, EVs, _, _ = forward_ldr_nm(opts, model, data)

            else:
                loss, EVs, _ = forward_hist_nm(opts, model, data)            # else:
                # loss, EVs, _, _ = forward_cnn(opts, model, data)
            loss.backward()
            optimizer.step()
                       
            if epoch_train_loss is None:
                epoch_train_loss = get_errors(loss)
            else:
                epoch_train_loss = utils.concatenate_dicts(epoch_train_loss, get_errors(loss))
            
            gc.collect()
            iterations += 1
            bar.next()
        bar.finish()

        train_end = time.perf_counter()

        # VALIDATE
        epoch_val_loss = None
        set_mode('eval', model)

        bar = progress.bar.Bar('Epoch {} val'.format(epoch), max=len(val_loader))

        MSELoss = nn.MSELoss()

        for data in val_loader:

            print('\n')
            target = data['target'][:,2:]
            print('target', target)
            EVs, EV_nms, loss_EV, loss_hist = model_test(opts, model, loss_function, data)
            target = torch.transpose(target, 0, 1).to(opts.device)

            print('EV', torch.transpose(EVs, 0, 1))
            print('EV_nm', torch.transpose(EV_nms, 0, 1))
            
            
            loss_mse = MSELoss(EVs, target)   
            
            if epoch_val_loss is None:
                epoch_val_loss = get_errors(loss_EV)
                epoch_val_mse = get_errors(loss_mse)
                epoch_val_hist = get_errors(loss_hist)
            else:
                epoch_val_loss = utils.concatenate_dicts(epoch_val_loss, get_errors(loss_EV))
                epoch_val_mse = utils.concatenate_dicts(epoch_val_mse, get_errors(loss_mse))
                epoch_val_hist = utils.concatenate_dicts(epoch_val_hist, get_errors(loss_hist))

            bar.next()
        bar.finish()
        
        epoch_end = time.perf_counter()
        epoch_avg_val_loss = utils.compute_dict_avg(epoch_val_loss)
        epoch_avg_train_loss = utils.compute_dict_avg(epoch_train_loss)
        epoch_avg_val_mse = utils.compute_dict_avg(epoch_val_mse)
        epoch_avg_val_hist = utils.compute_dict_avg(epoch_val_hist)
        
        train_fps = len(train_data)/(train_end-epoch_start)
        val_fps = len(val_data)/(epoch_end-train_end)

        print('End of epoch {}/{} | iter: {} | time: {:.3f} s | train: {:.3f} fps | val: {:.3f} fps'.format(epoch, num_epochs, iterations, epoch_end - epoch_start, train_fps, val_fps))

        # LOG ERRORS
        train_errors = utils.tag_dict_keys(epoch_avg_train_loss, 'train')
        val_errors = utils.tag_dict_keys(epoch_avg_val_loss, 'val')
        val_mse = utils.tag_dict_keys(epoch_avg_val_mse, 'val')
        val_hist = utils.tag_dict_keys(epoch_avg_val_hist, 'val')

        print('Train errors: ', train_errors)
        print('Val errors: ', val_errors)
        print('Val mse: ', val_mse)
        print('Val hist loss: ', val_hist)
        
        for key, value in sorted(train_errors.items()):
            # print('Key: ', key, 'Value: ', value)
            train_writer.add_scalar(key, value, epoch)
            print('{:20}: {:.3e}'.format(key, value))

        for key, value in sorted(val_errors.items()):
            # print('Key: ', key, 'Value: ', value)
            val_writer.add_scalar(key, value, epoch)
            print('{:20}: {:.3e}'.format(key, value))

        # SAVE CHECKPOINT
        save_checkpoint(epoch, 'latest', opts, model)

        if epoch % opts.checkpoint_interval == 0:
            save_checkpoint(epoch, epoch, opts, model)

        curr_total_val_loss = 0
        for key, val in epoch_avg_val_loss.items():
            try:
                curr_total_val_loss += val[-1]
            except IndexError:
                curr_total_val_loss += val
        
        if curr_total_val_loss < opts.best_model:
            save_checkpoint(epoch, 'best', opts, model)
            opts.best_model = curr_total_val_loss

            # save the config of the best performing model
            opts.save_txt('best_model_config.txt')

    
def test(opts, model, test_data, which_epoch='best', batch_size=1, save_loss=False):
    tmpT = transforms.ToPILImage()

    test_loader = DataLoader(test_data,
                             batch_size = batch_size,
                             shuffle=False,
                             num_workers=opts.dataloader_workers,
                             pin_memory=True)

    model = load_checkpoint(opts, model, which_epoch)
    set_mode('eval', model)

    output_dir = os.path.join(opts.results_dir, opts.experiment_name, 'test_{}'.format(which_epoch))

    os.makedirs(output_dir, exist_ok=True)

    test_start = time.perf_counter()


    bar = progress.bar.Bar('Test', max=len(test_loader))


    for idx, data in enumerate(test_loader):    
        seq_dir = os.path.join(opts.results_dir, idx)
        if not os.path.exists(seq_dir):
            os.mkdir(seq_dir)            
        fout = open(os.path.join(seq_dir, 'score.txt'),'a')

        score_seq, matched_image_seq, EV_seq = model_test_orb(opts, model, data)
        for i in range(len(matched_image_seq)):        
            output = tmpT(matched_image_seq[i])
            output.save(os.path.join(seq_dir, str(i)+'_' + str(EV_seq[i].cpu().detach().numpy()) + '.jpg'))
            fout.write(str(EV_seq[i].cpu().detach().numpy()) + ' ' + str(score_seq[i].cpu().detach().numpy()) +'\n')
        bar.next()
    bar.finish()
    fout.close()

    test_end = time.perf_counter()
    test_fps = len(test_data)/(test_end-test_start)
    print('Processed {} images | time: {:.3f} s | test: {:.3f} fps'.format(len(test_data), test_end-test_start, test_fps))

  

def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=4, scale_lr=10.0):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr
