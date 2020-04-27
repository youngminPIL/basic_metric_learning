from __future__ import absolute_import
from collections import defaultdict, OrderedDict

import numpy as np
import random
import os.path as osp
import os
import glob
from sklearn.metrics.pairwise import pairwise_distances
import torch
from torch.utils.data.sampler import Sampler









class RandomIdentitySampler_ori(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)












class RandomIdentitySampler_simple(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.
    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def random_members(self, pid, num_instances=-1):
        num_instances = self.num_instances if num_instances==-1 else num_instances
        t = self.index_dic[pid]
        if len(t) >= num_instances:
            t = np.random.choice(t, size=num_instances, replace=False)
        else:
            t = np.random.choice(t, size=num_instances, replace=True)
        return t

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.random_members(pid)
            ret.extend(t)
        return iter(ret)

class SeqSampler(Sampler):
    def __init__(self, order):
        self.order = order
    def __len__(self):
        return len(self.order)
    def __iter__(self):
        return iter(self.order)


class NeighborSampler(RandomIdentitySampler):
    def __init__(self, data_source, cls_proxy, batch_size, num_instances=1, distmat=None, feats=None, labels=None, feat_class_sorted_dist=None, feat_class_sorted_idx=None):
        super(NeighborSampler, self).__init__(data_source, num_instances=num_instances)
        self.batch_size = batch_size
        assert(batch_size % num_instances == 0)
        self.num_neigh = int(batch_size / num_instances) -1
        self.mults = [3,4,5] # [7,8,9] # [5,6,7] # [3,4,5] #[50]
        self.feats = feats
        self.labels = labels
        self.idx_to_pid = {idx:pid for idx,(pid,_) in enumerate(cls_proxy.items())}
        self.feat_class_sorted_dist = feat_class_sorted_dist
        self.feat_class_sorted_idx = feat_class_sorted_idx

    def hard_members_ktimes(self, anch_idx, neg_pids, mult=1):
        neg_cand_idx = np.where(np.isin(self.labels, neg_pids))[0]
        distmat_clspair = pairwise_distances(self.feats[anch_idx,:], self.feats[neg_cand_idx,:])
        neg_ranked = np.argsort(distmat_clspair, axis=1)
        d_ranked = distmat_clspair[[[i]*distmat_clspair.shape[1] for i in range(distmat_clspair.shape[0])], neg_ranked]
        neg_ranked = neg_cand_idx[neg_ranked]
        d, neg_idx = zip(*sorted(zip(d_ranked.reshape(-1), neg_ranked.reshape(-1))))
        neg_idx = list(OrderedDict.fromkeys(list(neg_idx)))
        print('Size of the neg idx pool: {}'.format(len(neg_idx)))
        print('nearest neighbor from top-{} instance pool'.format(len(neg_idx)))
        neg_idx = neg_idx[:self.num_neigh*self.num_instances]
        if len(neg_idx) < self.num_neigh * self.num_instances:
            print('WARINING: inefficient sampling!')
            neg_idx += [neg_idx[-1]]*(self.num_neigh*self.num_instances - len(neg_idx))

        return neg_idx

    def gen_neigh_from_aidx(self, anchor_idx, cls_mult):
        upper_bound = self.num_neigh * cls_mult
        d_ranked = self.feat_class_sorted_dist[anchor_idx, :upper_bound]
        neg_ranked = self.feat_class_sorted_idx[anchor_idx, :upper_bound]
        d, neg_idx = zip(*sorted(zip(d_ranked.reshape(-1), neg_ranked.reshape(-1))))
        neg_idx = list(OrderedDict.fromkeys(list(neg_idx)))[:upper_bound]
        assert(len(neg_idx)==cls_mult*self.num_neigh)
        print('num neg class: {}'.format(len(neg_idx)))
        neg_pids = [self.idx_to_pid[n] for n in neg_idx]
        print('anchor_idx: {}'.format(anchor_idx))
        print('neg pids: {}'.format(neg_pids))
        return neg_pids

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        cnt = 0
        for i in indices:
            mult_idx = np.random.choice(len(self.mults))
            mult = self.mults[mult_idx]

            # Anchor pids & idx
            anchor_pid = self.pids[i]
            anchor_idx = self.random_members(anchor_pid)

            # Negative pids
            print('Mine negative classes in a instance-level')
            print('anchor pid: {}'.format(anchor_pid))
            print('gen_neigh_from_aidx')
            pids = self.gen_neigh_from_aidx(anchor_idx, mult)
            print('Size of the candidate neighbor class: {}'.format(len(pids)))

            print('mult={}'.format(mult))
            neg_idx = self.hard_members_ktimes(anchor_idx, pids, mult=5)

            print('len(neg_idx)={}'.format(len(neg_idx)))
            print('neg idx: {}'.format(neg_idx))
            ret.extend(anchor_idx)
            ret.extend(neg_idx)

            if(len(ret) > self.__len__()):
                ret = ret[:self.__len__()]
                break
            cnt += 1
        return iter(ret)
