import os
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy
import mne



class EEGDataset(Dataset):
    '''
    Create the dataset for EEGs
    '''
    def __init__(self, X, y):
        super(EEGDataset, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_X_y(path, data):
    X_data = []
    y_data = []
    # path = r"data/spectrogram_data"
    for data in sorted(os.listdir(path)):
        file_path = os.path.join(path, data)
        if 'X' in data:
            X = np.load(file_path)
            X_data.append(X)
        elif 'y' in data:
            y = np.load(file_path)
            y_data.append(y)

    X = np.concatenate(X_data, axis = 0)
    y = np.concatenate(y_data, axis = 0)


