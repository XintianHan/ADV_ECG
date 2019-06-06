import numpy as np
import torch
import sys
import global_variables
import torch.nn.functional as F
device = global_variables.device

# Function for testing the model
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, labels in loader:
        data_batch, label_batch = data.to(device), labels.to(device)
        outputs = F.softmax(model(data_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += label_batch.size(0)
        correct += predicted.eq(label_batch.view_as(predicted)).sum().item()
    return (100 * correct / total)
# Function for calculating the F1 score
def cal_F1(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    cof_mat = np.zeros ((4,4))
    Ns = np.zeros(4)
    ns = np.zeros(4)
    for data, labels in loader:
        data_batch, label_batch = data.to(device), labels.to(device)
        outputs = F.softmax(model(data_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        total += label_batch.size(0)
        correct += predicted.eq(label_batch.view_as(predicted)).sum().item()
        acc = label_batch.view_as(predicted)
        for (a,p) in zip(acc, predicted):
            cof_mat[a][p] += 1
            Ns[a] += 1
            ns[p] += 1
    F1 = 0.0
    for i in range(len(Ns)):
        tempF = cof_mat[i][i]*2.0 /(Ns[i] + ns[i])
        F1 = F1+ tempF
        print('F1'+str(i)+':',tempF)
    F1 = F1/4.0
    print('cofmat',cof_mat)
    return 100 * correct / total, F1