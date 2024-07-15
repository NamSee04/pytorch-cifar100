import os
import argparse

import numpy
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torch.nn import CTCLoss


from models.c3_squeezenet import SqueezeNetLight as model
from zc3.preprocess import test_loader
from zc3.evaluate import evaluate
#from evaluate import evaluate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    args = parser.parse_args()

    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    criterion.to('cuda')

    net = model()
    net.load_state_dict(torch.load(args.weights))

    print(net)
    net.eval()
    evaluation = evaluate(net, test_loader, criterion)
    print('Test_result: loss={loss}, acc={acc}'.format(**evaluation))
