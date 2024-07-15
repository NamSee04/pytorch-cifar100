""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy
import pandas as pd
from PIL import Image

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


def get_network(args):
    """ return given network
    """

    if args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'squeezenetlight':
        from models.squeezenetLight import squeezenetlight
        net = squeezenetlight()
    elif args.net == 'squeezenetv1':
        from models.squeezenetv1 import squeezenetv1
        net = squeezenetv1()
    elif args.net == 'squeezenetv2':
        from models.squeezenetv2 import squeezenetv2
        net = squeezenetv2()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_val = pd.read_csv('./data/val.csv')

class CustomDataset():
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_link']
        image = Image.open(img_path).convert('L')  # Open image and convert to grayscale
        label = self.df.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label

def get_training_dataloader(batch_size=16, shuffle=True, num_workers=2):
    """ return training dataloader
    Args:
        path: path to 952cangjie training dataset
        batch_size: dataloader batchsize
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is in grayscale
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(df_train, transform=transform_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_data_loader


def get_test_dataloader(batch_size=16, shuffle=True, num_workers=2):
    """ return training dataloader
    Args:
        path: path to 952cangjie testing dataset
        batch_size: dataloader batchsize
        shuffle: whether to shuffle
    Returns: test_data_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is in grayscale
        transforms.ToTensor()
    ])

    test_dataset = CustomDataset(df_test, transform=transform_test)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return test_data_loader

def get_valid_dataloader(batch_size=16, shuffle=True, num_workers=2):
    """ return training dataloader
    Args:
        path: path to 952cangjie valid dataset
        batch_size: dataloader batchsize
        shuffle: whether to shuffle
    Returns: valid_data_loader:torch dataloader object
    """
    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is in grayscale
        transforms.ToTensor()
    ])

    val_dataset = CustomDataset(df_val, transform=transform_val)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return val_data_loader


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]