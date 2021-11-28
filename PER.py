from collections import deque
import numpy as np
from numpy.lib.arraysetops import isin
from baseline.utils import CompressedDeque
from copy import deepcopy
import itertools


class Tree:

    def __init__(self, value=1.0, maxlen=10000):
        self.max_value = value
        self.prior = np.ones(0)
        self.idx = []
        self.maxlen = maxlen
        self.alpha = 1.0
    
    def push(self, n=1):
        a = [self.max_value for i in range(n)]
        self.prior = np.append(
            self.prior, a
        )
        # if len(self.prior) > self.maxlen:
        #     np.delete(self.prior , 0)
    
    def update_max_value(self, value):
        self.max_value = value
    
    def update(self, idx:list, vals:np.ndarray, bias=0):

        max_value = max(self.prior)
        self.max_value = max_value
        idx = np.array(idx) - bias
        if max(idx) > self.maxlen:
            print(bias)
        assert max(idx) <= self.maxlen
        idx_check = idx >= 0
        idx = idx[idx_check]
        self.prior[idx] = vals[idx_check]
    
    def __len__(self):
        return len(self.prior)
        

class PER:
    def __init__(
        self,
        maxlen=1000,
        max_value=1.0):
        self.length=0
        # self.memory = CompressedDeque(maxlen=maxlen)
        # self.memory = deque(maxlen=maxlen)
        # self.memory = np.empty(0)
        self.memory = {}
        self.priority = Tree()
        self.maxlen = maxlen
        self.max_value = max_value

        self.bias = 0
        self.switch = False
        # self.alpha = alpha

    def push(self, d, offset=0): 
        n = len(d)
        for i, j in enumerate(d):
            self.memory[i+offset] = j
        self.priority.push(n)

        len_prioriy = len(self.priority)
        # 10001
        # offset: 9999
        if len_prioriy > self.maxlen:
            delta = len_prioriy - self.maxlen
            x = [i for i in range(delta)]
            self.priority.prior = np.delete(self.priority.prior, x)
            min_key = min(list(self.memory.keys()))
            for m in range(delta):
                del self.memory[min_key+m]
            if len(self.memory) != self.maxlen or len(self.priority) != self.maxlen:
                print(offset)
                print(delta)
                print(len(self.memory))
                print(len(self.priority))
                print('----------------')
   
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

        keys = list(self.memory.keys())
        min_id = min(keys)
        self.priority.update(idx, vals, min_id)
        
    def sample(self, batch_size):
        """
        binary data, probability, idx -> data:key
        """
        assert len(self.priority.prior) < (self.maxlen + 1)
        keys = list(self.memory.keys())

        prob = self.priority.prior / np.sum(self.priority.prior)

        a = [i for i in range(len(prob))]
        try:
            idx = np.random.choice(a, batch_size, p=prob)
        except ValueError:
            # print("Probability is WRONG?")
            d = 1 - sum(prob)
            prob[-1] += d
            idx = np.random.choice(a, batch_size, p=prob)
        key_idx = [keys[id] for id in idx]
        bin_data =[self.memory[id] for id in key_idx]
        s_prob = prob[idx]
        
        return list(bin_data), s_prob, key_idx
    
    @property
    def max_weight(self):
        return self.priority.max_value