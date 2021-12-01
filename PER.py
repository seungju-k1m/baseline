from collections import deque
import numpy as np
from numpy.lib.arraysetops import isin
from baseline.utils import CompressedDeque
from copy import deepcopy
import itertools
import _pickle as pickle
import time
import torch


class Tree:

    def __init__(self, value=1.0, maxlen=10000):
        self.max_value = value
        self.prior = np.ones(0)
        self.prior_torch = torch.tensor([])
        self.idx = []
        self.maxlen = maxlen
        self.alpha = 1.0
    
    def push(self, priorities):
        # self.prior = np.append(
        #     self.prior, priorities
        # )
        self.prior_torch = torch.cat(
            (self.prior_torch, torch.tensor(priorities)), dim=0
        )

        # if len(self.prior) > self.maxlen:
        #     np.delete(self.prior , 0)
    
    def update_max_value(self, value):
        self.max_value = value
    
    def update(self, idx:list, vals:np.ndarray):

        # max_value = max(self.prior)
        # self.max_value = max_value
        # idx = np.array(idx)
        # self.prior[idx] = vals
        self.prior_torch[idx] = torch.tensor(vals).float()
    
    def __len__(self):
        return len(self.prior)
        

class PER:
    def __init__(
        self,
        maxlen=1000,
        max_value=1.0,
        beta=0.4):
        self.beta = beta
        self.length=0
        # self.memory = CompressedDeque(maxlen=maxlen)
        # self.memory = deque(maxlen=maxlen)
        # self.memory = np.empty(0)
        # self.memory = {}
        self.memory = []
        self.priority = Tree(maxlen=maxlen)
        self.maxlen = maxlen
        self.max_value = max_value

        self.bias = 0
        self.switch = False
        # self.alpha = alpha

    def push(self, d): 
        priorities = []
        for i, j in enumerate(d):
            data = pickle.loads(j)
            priorities.append(data[-1])
            self.memory.append(j)
        self.priority.push(priorities)
   
    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)
    
    def update(self, idx:list, vals:np.ndarray):
        """
        alpha !!
        """
        assert isinstance(vals, np.ndarray)
        assert isinstance(idx, list)

        self.priority.update(idx, vals)
        
    def sample(self, batch_size):
        """
        binary data, probability, idx -> data:key
        """

        # prob = self.priority.prior / (np.sum(self.priority.prior))
        prob = self.priority.prior_torch / torch.sum(self.priority.prior_torch)

        a = [i for i in range(len(prob))]
        idx = torch.multinomial(prob, batch_size)
        # try:
        #     idx = np.random.choice(a, batch_size, p=prob)
        # except ValueError:
        #     # print("Probability is WRONG?")
        #     # d = 1 - sum(prob)
        #     # if d > 0:
        #     #     prob[-1] += d
        #     # else:
        #     #     prob[-1] -= d
        #     prob = prob / np.sum(prob)
        #     idx = np.random.choice(a, batch_size, p=prob)
        bin_data = deepcopy([self.memory[id] for id in idx])
        s_prob = prob[idx]
        
        return bin_data, s_prob, idx
    
    def remove_to_fit(self):
        len_memory = len(self.memory)
        if len_memory <= self.maxlen:
            return 
        else:
            delta = len(self.memory) - self.maxlen
            del self.memory[:delta]
            ix = [i for i in range(delta)]

            self.priority.prior_torch = self.priority.prior_torch[delta:].contiguous()

    @property
    def max_weight(self):
        min_value = self.priority.prior_torch.min()
        min_prob = min_value / torch.sum(self.priority.prior_torch)
        max_weight = (len(self.priority) * min_prob) ** -self.beta
        return float(max_weight)