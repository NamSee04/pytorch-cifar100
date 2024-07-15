import string
import pandas as pd
import numpy
from PIL import Image


import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

CHARS = 'abcdefghijklmnopqrstuvwxyz'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

df_train = pd.read_csv('./data/c3_train.csv')
df_test = pd.read_csv('./data/c3_test.csv')
df_val = pd.read_csv('./data/c3_val.csv')

transform_dataset = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is in grayscale
    transforms.ToTensor()
])

class CustomDataset(Dataset):
    def __init__(self, transform=None, mode=None):
        self.transform = transform
        self.mode = mode

        if mode == 'train': 
            self.df = df_train
        elif mode == 'valid':
            self.df = df_val
        elif mode == 'test':
            self.df = df_test

        if self.df is None:
            raise ValueError("paths and texts cannot be None")
    
        self.paths = [path for path in self.df['image_link']]
        self.texts = [path for path in self.df['Cangjie']]

    def __len__(self):
        return len(self.df)

    def  __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert('L')
        image = self.transform(image)

        text = self.texts[index]
        target = [CHAR2LABEL[c] for c in text]
        target_length = [len(target)]
        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)
        
        return image, target, target_length

def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths

train_dataset = CustomDataset(mode='train', transform=transform_dataset)
valid_dataset = CustomDataset(mode='valid', transform=transform_dataset)
test_dataset = CustomDataset(mode='test', transform=transform_dataset)


train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 4,
    collate_fn = collate_fn
)

valid_loader = DataLoader(
    dataset = valid_dataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 4,
    collate_fn = collate_fn
)

test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = 32,
    shuffle = True,
    num_workers = 4,
    collate_fn = collate_fn
)