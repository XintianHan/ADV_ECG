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
BATCH_SIZE = 16
MAX_SENTENCE_LENGTH = 18000
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

def pgd(inputs, lengths, targets, model, criterion, eps = None, step_alpha = None, num_steps = None):
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
        crafting_input = torch.autograd.Variable(crafting_output.clone(), requires_grad=True)
    added = crafting_output - inputs
    crafting_output = inputs+ added
    crafting_output_clamp = crafting_output.clone()
    # remove pertubations on the padding
    for i in range(crafting_output_clamp.size(0)):
        remainder = MAX_SENTENCE_LENGTH - lengths[i]
        if remainder > 0:
            crafting_output_clamp[i][0][:int(remainder / 2)] = 0
            crafting_output_clamp[i][0][-(remainder - int(remainder / 2)):] = 0
    return crafting_output_clamp

def success_rate(data_loader, model, eps = 1, step_alpha = None, num_steps = None):
    model.eval()
    correct = 0.0
    correct_clamp = 0.0
    adv_exps = []
    adv_probs = []
    adv_classes = []
    pred_classes = []
    pred_probs = []
    pred_exps = []

    for bi, (inputs, lengths, targets) in enumerate(data_loader):
        inputs_batch, lengths_batch, targets_batch = inputs.to(device), lengths.to(device), targets.to(device)
        crafted_clamp = pgd(inputs_batch, lengths_batch, targets_batch, model, F.cross_entropy, eps, step_alpha, num_steps)
        output = model(inputs_batch)
        output_clamp = model(crafted_clamp)
        pred = output.data.max(1, keepdim=True)[1].view_as(
            targets_batch)  # get the index of the max log-probability
        pred_clamp = output_clamp.data.max(1, keepdim=True)[1].view_as(targets_batch)
        idx1 = (pred == targets_batch)
        idx2 = (pred != pred_clamp)
        idx = idx1 & idx2
        correct_clamp += pred_clamp.eq(targets_batch.view_as(pred_clamp)).cpu().numpy().sum()
        pred_exps.append(inputs_batch[idx].detach().cpu().numpy())
        adv_classes.append(pred_clamp[idx].detach().cpu().numpy())
        pred_classes.append(pred[idx].cpu().numpy())
        adv_probs.append(F.softmax(output_clamp)[idx].detach().cpu().numpy())
        pred_probs.append(F.softmax(output)[idx].detach().cpu().numpy())
        adv_exps.append(crafted_clamp[idx].detach().cpu().numpy())

    adv_exps = np.concatenate(adv_exps)
    adv_probs = np.concatenate(adv_probs)
    adv_classes = np.concatenate(adv_classes)
    pred_classes = np.concatenate(pred_classes)
    pred_probs = np.concatenate(pred_probs)
    pred_exps = np.concatenate(pred_exps)
    path = 'adv_exp/pgd'
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    np.save(path+'/adv_exps.npy', adv_exps)
    np.save(path+'/adv_probs.npy', adv_probs)
    np.save(path+'/adv_classes.npy', adv_classes)
    np.save(path+'/pred_classes.npy', pred_classes)
    np.save(path+'/pred_probs.npy', pred_probs)
    np.save(path+'/pred_exps.npy', pred_exps)
    correct_clamp /= len(data_loader.sampler)
    return correct_clamp

srpgd = success_rate(val_loader, model, eps = 10, step_alpha = 1, num_steps = 20)
print('success rate PGD 10,1,20:', srpgd)
sys.stdout.flush()