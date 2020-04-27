from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data.sampler import Sampler

import torch
from collections import defaultdict
import numpy as np
# import matplotlib.pyplot as plt


class RandomIdentitySampler(Sampler):
    def __init__(self, dataset, num_sample_per_class=1):
        self.dataset = dataset
        self.num_sample_per_class = num_sample_per_class
        self.index_dic = defaultdict(list)
        for index, (_, label) in enumerate(dataset):
            self.index_dic[label].append(index)
        self.labels = list(self.index_dic.keys())
        self.num_sample = len(self.labels)

    def __len__(self):
        return self.num_sample * self.num_sample_per_class

    def __iter__(self):
        indices = torch.randperm(self.num_sample)
        ret = []
        for i in indices:
            label = self.labels[i]
            t = self.index_dic[label]
            if len(t) >= self.num_sample_per_class:
                t = np.random.choice(t, size=self.num_sample_per_class, replace=False)
            else:
                t = np.random.choice(t, size=self.num_sample_per_class, replace=True)
            ret.extend(t)
        return iter(ret)

