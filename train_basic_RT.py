# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib
from test_embedded import Get_test_results_single
#matplotlib.use('agg')
#import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import ft_net, ft_net_scratch, ft_net_option_feature
from tensorboard_logger import configure, log_value
import json
#import visdom
import copy


######################################################################
# Options

# data_dir = '/home/ro/FG/STCAR_RT/pytorch'
data_dir = '/home/ro/FG/CUB_RT/pytorch'

# data_dir = '/home/jhlee/CUB_200_2011/pytorch'


dir_name = '/data/ymro/AAAI2021/base_CUB/res50_lr3_fclr2'

e_drop = 10
e_drop2 = 0
e_end = 20

train_batchsize = 40
test_batchsize = 30
configure(dir_name)
print(dir_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


use_gpu = True
gpu_id = 2
gpu_ids = []
gpu_ids.append(gpu_id)
# set gpu id2
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
print(gpu_ids[0])

lr1 = 0.1
lr2 = 0.01
lr3 = 0.001
lr4 = 0.0001

######################################################################
# Load Data
# ---------
#
init_resize = (256, 256)
resize = (224, 224)

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize(resize, interpolation=3),
    #transforms.RandomCrop(resize),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(init_resize, interpolation=3),  # Image.BICUBIC
    transforms.CenterCrop(resize),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]


print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'test': transforms.Compose(transform_val_list),
}


image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                               data_transforms['train'])
image_datasets['test'] = datasets.ImageFolder(os.path.join(data_dir, 'test'),
                                               data_transforms['test'])

dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=train_batchsize, shuffle=True, num_workers=16)

dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size= test_batchsize, shuffle=False, num_workers=8)



dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))




######################################################################



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    save_network(model, -1)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                _, outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # results = Get_test_results_single(image_datasets['test'], dataloaders['test'], model, f_size=2048)
                # # print(results)
                # print(
                #     'test accuracy : top-1 {:.4f} top-2 {:.4f} top-4 {:.4f} top-8 {:.4f}'.format(results[0], results[1],
                #                                                                                  results[2],
                #                                                                                  results[3]))

                # statistics
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data)

            results = Get_test_results_single(image_datasets['test'], dataloaders['test'], model, f_size=512)
            # print(results)
            print('test accuracy : top-1 {:.4f} top-2 {:.4f} top-4 {:.4f} top-8 {:.4f}'.format(results[0],results[1],results[2],results[3]))
            running_corrects = running_corrects.float()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.8f} Acc: {:.8f}'.format(phase, epoch_loss, epoch_acc))

            log_value('train_loss', epoch_loss, epoch)
            log_value('train_acc', epoch_acc, epoch)

            last_model_wts = model.state_dict()
            save_network(model, epoch)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


####################################################
# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(dir_name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])
        # nn.DataParallel(network, device_ids=[2,3]).cuda()


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#
def load_network_path(network, save_path):
    network.load_state_dict(torch.load(save_path))
    return network

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

model = ft_net_option_feature(int(len(class_names)), 512, 1, 2)
print('ok')
if use_gpu:
    model = model.cuda()
    # nn.DataParallel(model, device_ids=[2,3]).cuda()
criterion = nn.CrossEntropyLoss()


params_ft = []
params_ft.append({'params': model.model.conv1.parameters(), 'lr': lr3})
params_ft.append({'params': model.model.bn1.parameters(), 'lr': lr3})
params_ft.append({'params': model.model.layer1.parameters(), 'lr': lr3})
params_ft.append({'params': model.model.layer2.parameters(), 'lr': lr3})
params_ft.append({'params': model.model.layer3.parameters(), 'lr': lr3})
params_ft.append({'params': model.model.layer4.parameters(), 'lr': lr3})
params_ft.append({'params': model.classifier.parameters(), 'lr': lr2})

optimizer_ft = optim.SGD(params_ft, momentum=0.9, weight_decay=5e-4, nesterov=True)


if e_drop2 == 0:
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=e_drop, gamma=0.1)
else:
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[e_drop,e_drop2], gamma=0.1)


model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=e_end)



