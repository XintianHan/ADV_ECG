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

print('Check uniform')
######## Check adv exp after concatenation
MAX_SENTENCE_LENGTH = 18000
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# make a list of data and labels
# Load data manually
total1 = np.load('adv_exp/total1_5.npy')
total2 = np.load('adv_exp/total2_5.npy')
total3 = np.load('adv_exp/total3_5.npy')
total4 = np.load('adv_exp/total4_5.npy')
real_inputs = np.load('adv_exp/real_inputs.npy')
real1 = torch.from_numpy(real_inputs[0][0]).to(device)
real2 = torch.from_numpy(real_inputs[4][0]).to(device)
real3 = torch.from_numpy(real_inputs[2][0]).to(device)
real4 = torch.from_numpy(real_inputs[3][0]).to(device)
reals = [real1, real2, real3, real4]
max1 = torch.from_numpy(np.max(total1, axis = 0)).to(device)
max2 = torch.from_numpy(np.max(total2, axis = 0)).to(device)
max3 = torch.from_numpy(np.max(total3, axis = 0)).to(device)
max4 = torch.from_numpy(np.max(total4, axis = 0)).to(device)
min1 = torch.from_numpy(np.min(total1, axis = 0)).to(device)
min2 = torch.from_numpy(np.min(total2, axis = 0)).to(device)
min3 = torch.from_numpy(np.min(total3, axis = 0)).to(device)
min4 = torch.from_numpy(np.min(total4, axis = 0)).to(device)
maxs = [max1, max2, max3, max4]
mins = [min1, min2, min3, min4]

# Load length of data manually
lengths = [9000,9000,9000,9000]

# Set the label manually (truth)
True_labels = [3, 2, 0, 0]

# Threshold, under which we concat
Threshold = 0.001

# Load the model 
# preprocess = 'zero'
# filtered = 'F'
# ratio = 19
# file_name = '_'.join([NETWORK, str(ratio), filtered, preprocess])

model = CNN(num_classes=4)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load('saved_model/best_model.pth', map_location=lambda storage, loc: storage))
for param in model.parameters():
    param.requires_grad = False

total_numbers = [0.0, 0.0, 0.0, 0.0]
adv_numbers = [0.0, 0.0, 0.0, 0.0]

# smooth convolution
sizes = [5, 7, 11, 15, 19]
sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
print('sizes:',sizes)
print('sigmas:', sigmas)
crafting_sizes = []
weights = []
for size in sizes:
    for sigma in sigmas:
        crafting_sizes.append(size)
        weight = np.arange(size) - size//2
        weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
        weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device)
        weights.append(weight)

# Concat and test
model.eval()
N = 100
# Start create new signals
for i in range(len(total_numbers)):
    length = lengths[i]
    real = reals[i]
    # true label
    true_label = torch.tensor([True_labels[i]]).to(device)
    for k in range(N):
        # sample uniform
        sample = torch.rand(real.shape).to(device)*(maxs[i]-mins[i])+mins[i]
        sample = sample.unsqueeze(0).unsqueeze(0)
        # smooth 
        added = sample - real.unsqueeze(0).unsqueeze(0)
        temp = F.conv1d(added, weights[0], padding=crafting_sizes[0] // 2)
        for j in range(len(sizes) - 1):
            temp = temp + F.conv1d(added, weights[j + 1], padding=crafting_sizes[j + 1] // 2)
        temp = temp / float(len(sizes))
        crafting_output = real.unsqueeze(0).unsqueeze(0) + temp.detach()
        # project back to the band
        max_unsq = maxs[i].unsqueeze(0).unsqueeze(0)
        min_unsq = mins[i].unsqueeze(0).unsqueeze(0)
        crafting_output = torch.min(crafting_output, max_unsq)
        crafting_output = torch.max(crafting_output, min_unsq)
        # remove noise on padding MAYBE redundant since we already project to min/max
        crafting_output_clamp = crafting_output.clone()
        remainder = MAX_SENTENCE_LENGTH - length
        if remainder > 0:
            crafting_output_clamp[0][0][:int(remainder / 2)] = 0
            crafting_output_clamp[0][0][-(remainder - int(remainder / 2)):] = 0
        # predict 
        output_craft = model(crafting_output_clamp)
        pred_craft = output_craft.data.max(1, keepdim=True)[1].view_as(true_label)
        total_numbers[i] += 1.0
        correct = pred_craft.eq(true_label)
        if not correct:
            adv_numbers[i] += 1
print('total_numbers:', total_numbers)
print('adv_numbers', adv_numbers)
for i in range(len(total_numbers)):
    print(i, 'ratio', adv_numbers[i]*1.0/total_numbers[i])