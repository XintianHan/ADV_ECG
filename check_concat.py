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

print('Check concatenation')
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
# cut number of data to 10
NUMBER = 10
total1 = total1[:NUMBER]
total2 = total2[:NUMBER]
total3 = total3[:NUMBER]
total4 = total4[:NUMBER]
totals = [total1, total2, total3, total4]

# Load length of data manually
lengths = [9000,9000,9000,9000]

# Set the label manually (truth)
True_labels = [3, 2, 0, 0]

# Threshold, under which we concat
Threshold = 0.001

model = CNN(num_classes=4)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load('saved_model/best_model.pth', map_location=lambda storage, loc: storage))
for param in model.parameters():
    param.requires_grad = False

total_numbers = [0.0, 0.0, 0.0, 0.0]
adv_numbers = [0.0, 0.0, 0.0, 0.0]

# Concat and test
model.eval()

for i in range(len(totals)):
    total = torch.from_numpy(totals[i]).float().to(device)
    length = lengths[i]
    # kj and jk so labels doubled
    true_label = torch.tensor([True_labels[i], True_labels[i]]).to(device)
    # Loop through all combinations of examples
    num_total= len(total)
    for j in range(num_total):
        for k in range(j+1, num_total):
            # Find indices to concat
            # within [Length//2, MAX_SENTENCE_LENGTH-Length//2] 
            sample_j = total[j]
            sample_k = total[k]
            diff = torch.abs(sample_j-sample_k)[length//2:MAX_SENTENCE_LENGTH-length//2]
            indices = (diff==0).nonzero() + length//2
            # if non of the differences under threshold, skip
            if len(indices) == 0:
                continue
            # start concat
            print('length of indices', len(indices))
            for l in range(len(indices)):
                # cannot concat first or last position
                if l == length//2 or l == MAX_SENTENCE_LENGTH-length//2-1:
                    continue
                # start concat
                # first part j; second part k
                sample_jk = torch.zeros(sample_j.shape).to(device)
                sample_jk[:l] = sample_j[:l]
                sample_jk[l:] = sample_k[l:]
                # first part k; second part j
                sample_kj = torch.zeros(sample_j.shape).to(device)
                sample_kj[:l] = sample_k[:l]
                sample_kj[l:] = sample_j[l:]
                # make jk and kj into a batch and shape of the input of the network 
                sample_jk = sample_jk.unsqueeze(0).unsqueeze(0)
                sample_kj = sample_kj.unsqueeze(0).unsqueeze(0)
                samples_new = torch.cat((sample_jk, sample_kj), dim = 0)
                output_craft = model(samples_new)
                pred_craft = output_craft.data.max(1, keepdim=True)[1].view_as(true_label)
                correct = pred_craft.eq(true_label)
                total_numbers[i] = total_numbers[i] + 2
                adv_numbers[i] = adv_numbers[i] + 2 - correct.cpu().numpy().sum()
                # print('correct', correct)
            print(i, 'adv numbers', adv_numbers[i])
            print(i, 'total numbers', total_numbers[i])
            sys.stdout.flush()
print('total_numbers:', total_numbers)
print('adv_numbers', adv_numbers)
for i in range(len(totals)):
    print(i, 'ratio', adv_numbers[i]*1.0/totals_numbers[i])