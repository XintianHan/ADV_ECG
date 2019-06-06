import scipy.io as sio
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
from model.CNN import CNN
from utils.DataLoader import ECGDataset, ecg_collate_func
import sys
import argparse
import os

data_dirc = 'data/'
RAW_LABELS = np.load(data_dirc+'raw_labels.npy')
PERMUTATION = np.load(data_dirc+'random_permutation.npy')
BATCH_SIZE = 8
MAX_SENTENCE_LENGTH = 18000
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
num_of_trials = 1000
LEARNING_RATE = 0.001
NUM_EPOCHS = 200  # number epoch to train
data = np.load(data_dirc+'raw_data.npy')
data = data[PERMUTATION]
RAW_LABELS = RAW_LABELS[PERMUTATION]
mid = int(len(data)*0.9)
val_data = data[mid:]
val_label = RAW_LABELS[mid:]
val_dataset = ECGDataset(val_data, val_label)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=ecg_collate_func,
                                           shuffle=False)
model = CNN(num_classes=4)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load('saved_model/best_model.pth', map_location=lambda storage, loc: storage))
for param in model.parameters():
    param.requires_grad = False
def pgd_conv(inputs, lengths, targets, model, criterion, eps = None, step_alpha = None, num_steps = None, sizes = None, weights = None):
    """
    :param inputs: Clean samples (Batch X Size)
    :param targets: True labels
    :param model: Model
    :param criterion: Loss function
    :param gamma:
    :return:
    """

    crafting_input = torch.autograd.Variable(inputs.clone(), requires_grad=True)
    crafting_target = torch.autograd.Variable(targets.clone())
    for i in range(num_steps):
        output = model(crafting_input)
        loss = criterion(output, crafting_target)
        if crafting_input.grad is not None:
            crafting_input.grad.data.zero_()
        loss.backward()
        added = torch.sign(crafting_input.grad.data)
        step_output = crafting_input + step_alpha * added
        total_adv = step_output - inputs
        total_adv = torch.clamp(total_adv, -eps, eps)
        crafting_output = inputs + total_adv
        crafting_input = torch.autograd.Variable(crafting_output.detach().clone(), requires_grad=True)
    added = crafting_output - inputs
    added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
    for i in range(num_steps*2):
        temp = F.conv1d(added, weights[0], padding = sizes[0]//2)
        for j in range(len(sizes)-1):
            temp = temp + F.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
        temp = temp/float(len(sizes))
        output = model(inputs + temp)
        loss = criterion(output, targets)
        loss.backward()
        added = added + step_alpha * torch.sign(added.grad.data)
        added = torch.clamp(added, -eps, eps)
        added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
    temp = F.conv1d(added, weights[0], padding = sizes[0]//2)
    for j in range(len(sizes)-1):
        temp = temp + F.conv1d(added, weights[j+1], padding = sizes[j+1]//2)
    temp = temp/float(len(sizes))
    crafting_output = inputs + temp.detach()
    crafting_output_clamp = crafting_output.clone()
    for i in range(crafting_output_clamp.size(0)):
        remainder = MAX_SENTENCE_LENGTH - lengths[i]
        if remainder > 0:
            crafting_output_clamp[i][0][:int(remainder / 2)] = 0
            crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
    sys.stdout.flush()
    return  crafting_output_clamp

def success_rate(data_loader, model, eps = 1,  step_alpha = None, num_steps = None, sizes = None, weights = None):
    model.eval()
    correct_clamp = 0.0
    adv_exps = []
    adv_probs = []
    adv_classes = []
    pred_classes = []
    pred_probs = []
    pred_exps = []
    for bi, (inputs, lengths, targets) in enumerate(data_loader):
        # Choose only one example
        if bi >=1:
            break
        print('*******start batch ', bi, '*********')
        inputs_batch, lengths_batch, targets_batch = inputs.to(device), lengths.to(device), targets.to(device)
        crafted_clamp = pgd_conv(inputs_batch, lengths_batch, targets_batch, model, F.cross_entropy, eps, step_alpha, num_steps, sizes, weights)
        output = model(inputs_batch)
        output_clamp = model(crafted_clamp)
        pred = output.data.max(1, keepdim=True)[1].view_as(targets_batch)  # get the index of the max log-probability
        pred_clamp = output_clamp.data.max(1, keepdim=True)[1].view_as(targets_batch)
        idx1 = (pred == targets_batch)
        idx2 = (pred != pred_clamp)
        idx = idx1 & idx2
        pred = pred[idx]
        pred_clamp = pred_clamp[idx]
        print('pred:', pred)
        print('pred_clamp', pred_clamp)
        crafted_clamp = crafted_clamp[idx]
        print('length', lengths_batch[idx])
        total1 = []
        total2 = []
        total3 = []
        total4 = []
        crafted = torch.zeros(2,crafted_clamp.shape[1], crafted_clamp.shape[2])
        for k in range(num_of_trials):
            # if k >= 1:
            #     break
            # corresponding real signals of adversarial examples
            real_inputs = inputs_batch[idx].clone()
            length_inputs = lengths_batch[idx].clone()
            crafted_inputs = crafted_clamp.detach().clone()
            # add noise
            crafted_inputs = crafted_inputs + torch.randn(crafted_inputs.size()).to(device)*5
            added = crafted_inputs - real_inputs
            # added = torch.clamp(added, -eps, eps)
            temp = F.conv1d(added, weights[0], padding=sizes[0] // 2)
            for j in range(len(sizes) - 1):
                temp = temp + F.conv1d(added, weights[j + 1], padding=sizes[j + 1] // 2)
            temp = temp / float(len(sizes))
            temp = torch.clamp(temp, -eps, eps)
            crafting_output = real_inputs + temp.detach()
            # crafting_output = real_inputs + added
            crafting_output_clamp = crafting_output.clone()
            for i in range(crafting_output_clamp.size(0)):
                remainder = MAX_SENTENCE_LENGTH - length_inputs[i]
                if remainder > 0:
                    crafting_output_clamp[i][0][:int(remainder / 2)] = 0
                    crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
            output_craft = model(crafting_output_clamp)
            pred_craft = output_craft.data.max(1, keepdim=True)[1].view_as(pred)
            # Choose samples that can create more adversarial examples
            if pred_craft[0] == pred_clamp[0]:
                crafting_output_clamp = crafting_output_clamp.squeeze()
                total1.append(crafting_output_clamp[0].clone().cpu().numpy())
            if pred_craft[4] == pred_clamp[4]:
                crafting_output_clamp = crafting_output_clamp.squeeze()
                total2.append(crafting_output_clamp[4].clone().cpu().numpy())
            if pred_craft[2] == pred_clamp[2]:
                crafting_output_clamp = crafting_output_clamp.squeeze()
                total3.append(crafting_output_clamp[2].clone().cpu().numpy())
            if pred_craft[3] == pred_clamp[3]:
                crafting_output_clamp = crafting_output_clamp.squeeze()
                total4.append(crafting_output_clamp[3].clone().cpu().numpy())

        total1  = np.stack(np.array(total1), axis=0)
        total2  = np.stack(np.array(total2), axis=0)
        total3  = np.stack(np.array(total3), axis=0)
        total4  = np.stack(np.array(total4), axis=0)
        path = 'adv_exp'
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        np.save(path+'/total1_5.npy', total1)
        np.save(path+'/total2_5.npy', total2)
        np.save(path+'/total3_5.npy', total3)
        np.save(path+'/total4_5.npy', total4)
        print('************* end batch **********')
    sys.stdout.flush()
print('*************')
sizes = [5, 7, 11, 15, 19]
sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
print('sizes:',sizes)
print('sigmas:', sigmas)
crafting_sizes = []
crafting_weights = []
for size in sizes:
    for sigma in sigmas:
        crafting_sizes.append(size)
        weight = np.arange(size) - size//2
        weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
        weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
        crafting_weights.append(weight)

success_rate(val_loader, model, eps = 10, step_alpha = 1,
                        num_steps = 20, sizes = crafting_sizes, weights = crafting_weights)
sys.stdout.flush()