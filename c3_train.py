import os
import sys
import re
import datetime
import argparse
import time

import numpy
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.nn import CTCLoss

from conf import settings


from models.c3_squeezenet import SqueezeNetLight as model
from zc3.preprocess import train_loader, valid_loader, test_loader
from zc3.evaluate import evaluate

MILESTONES = [3, 8, 13]

def train_batch(model, data, optimizer, criterion, device):
    model.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = model(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # gradient clipping with 5
    optimizer.step()
    return loss.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    args = parser.parse_args()

    net = model()
    net = net.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones= MILESTONES, gamma=0.2)
    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    criterion.to('cuda')

    # TensorBoard setup
    best_acc = 0.0
    device = 'cpu'
    if args.gpu:
        device = 'cuda'
    i = 1
    for epoch in range(1, settings.EPOCH + 1):
        start = time.time()
        train_scheduler.step(epoch)
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(net, train_data, optimizer, criterion, device = 'cuda')
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i == 1 or i % 100 == 0:
                print('Training Epoch[', epoch, ']: ', loss / train_size, f'{optimizer.param_groups[0]["lr"]:.6f}')

            i += 1

        evaluation = evaluate(net, valid_loader, criterion)
        print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))
        prefix = 'squeezenet'
        loss = evaluation['loss']
        best = evaluation['acc']
        if best > best_acc:
            best_acc = best
            save_model_path = os.path.join('./checkpoint/c3', f'{prefix}_{i:06}.pt')
            torch.save(net.state_dict(), save_model_path)
            print('save model at ', save_model_path)
            
        print('train_loss: ', tot_train_loss / tot_train_count)
        print(f'epoch {epoch} training time consumed: {time.time() - start:.2f}s')