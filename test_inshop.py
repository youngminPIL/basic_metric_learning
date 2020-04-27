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
import time
import os
import scipy.io
from model import ft_net
import random
######################################################################
# Options
# --------


test_dir = '/home/ro/FG/Inshop/pytorch'
num_class = 3997
gallery_eq_query = False

target = '/data/ymro/CVPR2020/RT/AUTO_inshop4_1_sqrtG2/thr94_nonstrict_min2_max04_ep3040_l1conv1binding_05_try1'
target = '/data/ymro/CVPR2020/RT/inshop_basline/3040_baseline4_1_try3'


test_batchsize = 60
feature_size = 2048 #resnet 50
num_bottle = feature_size
efrom = 39
euntil = 39

gpu_id = 0
gpu_ids.append(gpu_id)
# set gpu id2
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
print(gpu_ids[0])
##################################t##################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
init_resize = (256,256)
resize = (224,224)
data_transforms = transforms.Compose([
    transforms.Resize(resize, interpolation=3),
    transforms.CenterCrop(resize),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ############### Ten Crop
    # transforms.TenCrop(224),
    # transforms.Lambda(lambda crops: torch.stack(
    #   [transforms.ToTensor()(crop)
    #      for crop in crops]
    # )),
    # transforms.Lambda(lambda crops: torch.stack(
    #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
    #       for crop in crops]
    # ))
])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=test_batchsize,
                                              shuffle=False, num_workers=16) for x in ['gallery', 'query']}
use_gpu = torch.cuda.is_available()



######################################################################
# Load model
# ---------------------------

def load_network_path(network, save_path):
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    n_param = 5000
    feature_cat = torch.FloatTensor()
    features = []
    for i in range(len(dataloaders.dataset)//n_param+1):
        features.append(torch.FloatTensor())

    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)

        ff = torch.FloatTensor(n, feature_size).zero_()
        for i in range(2):
            if (i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        ii = count // n_param
        features[ii] = torch.cat((features[ii], ff), 0)
    for j in range(len(features)):
        feature_cat = torch.cat((feature_cat, features[j]), 0)
    return feature_cat

def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x


def pairwise_similarity(x, y=None):
    if y is None:
        y = x
    # normalization
    y = normalize(y)
    x = normalize(x)
    # similarity
    similarity = torch.mm(x, y.t())
    return similarity


def get_id(img_path):
    labels = []
    for path, v in img_path:
        label = path.split('/')[-2]
        # label = filename[0:4]
        # camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        # camera_id.append(int(camera[0]))
    return labels

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def Recall_at_ks(sim_mat, data='cub', query_ids=None, gallery_ids=None):
    # start_time = time.time()
    # print(start_time)
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids
    :param data

    Compute  [R@1, R@2, R@4, R@8]
    """

    ks_dict = dict()
    ks_dict['cub'] = [1, 2, 4, 8, 16, 32]
    ks_dict['car'] = [1, 2, 4, 8, 16, 32]
    ks_dict['jd'] = [1, 2, 4, 8]
    ks_dict['product'] = [1, 10, 100, 1000]
    ks_dict['shop'] = [1, 10, 20, 30, 40, 50]

    if data is None:
        data = 'cub'
    k_s = ks_dict[data]

    sim_mat = to_numpy(sim_mat)
    m, n = sim_mat.shape
    gallery_ids = np.asarray(gallery_ids)
    if query_ids is None:
        query_ids = gallery_ids
    else:
        query_ids = np.asarray(query_ids)

    num_max = int(1e6)

    if m > num_max:
        samples = list(range(m))
        random.shuffle(samples)
        samples = samples[:num_max]
        sim_mat = sim_mat[samples, :]
        query_ids = [query_ids[k] for k in samples]
        m = num_max

    # Hope to be much faster  yes!!
    num_valid = np.zeros(len(k_s))
    neg_nums = np.zeros(m)
    for i in range(m):
        x = sim_mat[i]

        pos_max = np.max(x[gallery_ids == query_ids[i]])
        neg_num = np.sum(x > pos_max)
        neg_nums[i] = neg_num

    for i, k in enumerate(k_s):
        if i == 0:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp
        else:
            temp = np.sum(neg_nums < k)
            num_valid[i:] += temp - num_valid[i-1]
    # t = time.time() - start_time
    # print(t)
    return num_valid / float(m)

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

query_label = get_id(query_path)
gallery_label = get_id(gallery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')

search = target + '/ft_ResNet50'


file_list = os.listdir(search)

print(search)
for tepoch in range(euntil+1):
    if tepoch < efrom:
        continue

    file_name = 'net_%d.pth' % tepoch
    path = search + '/' + file_name
    model = ft_net(num_class) #market
    model = load_network_path(model, path)



    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    query_feature = extract_feature(model, dataloaders['query'])
    gallery_feature = extract_feature(model, dataloaders['gallery'])

    # Save to Matlab for check

    query_label = np.asarray(query_label)
    gallery_label = np.asarray(gallery_label)

    sim_mat = pairwise_similarity(query_feature, gallery_feature)

    recall_ks = Recall_at_ks(sim_mat, query_ids=query_label, gallery_ids=gallery_label, data='shop')

    result = '  '.join(['%.4f' % k for k in recall_ks])
    #epoch = 1
    print('%s: %s'% (tepoch, result))
    #


