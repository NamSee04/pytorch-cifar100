#!/usr/bin/env python3

""" Train network using PyTorch

author: baiyu
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, get_valid_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

def train(epoch, net, training_loader, optimizer, loss_function, writer, warmup_scheduler, args):
    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        if args.gpu:
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1
        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print(f'Training Epoch: {epoch} [{batch_index * args.b + len(images)}/{len(training_loader.dataset)}]\tLoss: {loss.item():.4f}\tLR: {optimizer.param_groups[0]["lr"]:.6f}')

        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        writer.add_histogram(f"{layer}/{attr[1:]}", param, epoch)

    print(f'epoch {epoch} training time consumed: {time.time() - start:.2f}s')

@torch.no_grad()
def eval_training(epoch, net, data_loader, loss_function, writer, args, mode='Test'):
    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0

    for images, labels in data_loader:
        if args.gpu:
            images, labels = images.cuda(), labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print(f'Evaluating Network.....\n{mode} set: Epoch: {epoch}, Average loss: {test_loss / len(data_loader.dataset):.4f}, Accuracy: {correct.float() / len(data_loader.dataset):.4f}, Time consumed: {time.time() - start:.2f}s\n')

    writer.add_scalar(f'{mode}/Average loss', test_loss / len(data_loader.dataset), epoch)
    writer.add_scalar(f'{mode}/Accuracy', correct.float() / len(data_loader.dataset), epoch)

    return correct.float() / len(data_loader.dataset), test_loss / len(data_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)

    # Data loaders
    training_loader = get_training_dataloader(num_workers=4, batch_size=args.b, shuffle=True)
    valid_loader = get_valid_dataloader(num_workers=4, batch_size=args.b, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    valid_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # TensorBoard setup
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 1, 64, 64)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print(f'found best acc weights file: {weights_path}')
            net.load_state_dict(torch.load(weights_path))
            best_acc, _ = eval_training(0, net, valid_loader, loss_function, writer, args)
            print(f'best acc is {best_acc:.2f}')

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        net.load_state_dict(torch.load(weights_path))
        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume and epoch <= resume_epoch:
            continue

        train(epoch, net, training_loader, optimizer, loss_function, writer, warmup_scheduler, args)
        valid_acc, valid_loss = eval_training(epoch, net, valid_loader, loss_function, writer, args, mode='Valid')
        valid_scheduler.step(valid_loss)

        if epoch > settings.MILESTONES[0] and best_acc < valid_acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print(f'saving weights file to {weights_path}')
            torch.save(net.state_dict(), weights_path)
            best_acc = valid_acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print(f'saving weights file to {weights_path}')
            torch.save(net.state_dict(), weights_path)

    writer.close()
