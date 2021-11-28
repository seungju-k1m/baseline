from collections import deque
import numpy as np
from numpy.lib.arraysetops import isin
from baseline.utils import CompressedDeque
from copy import deepcopy
import itertools
import _pickle as pickle


class Tree:

    def __init__(self, value=1.0, maxlen=10000):
        self.max_value = value
        self.prior = np.ones(0)
        self.idx = []
        self.maxlen = maxlen
        self.alpha = 1.0
    
    def push(self, priorities):
        self.prior = np.append(
            self.prior, priorities
        )

        # if len(self.prior) > self.maxlen:
        #     np.delete(self.prior , 0)
    
    def update_max_value(self, value):
        self.max_value = value
    
    def update(self, idx:list, vals:np.ndarray):

        max_value = max(self.prior)
        self.max_value = max_value
        idx = np.array(idx)
        self.prior[idx] = vals
    
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
        assert len(self.priority.prior) < (self.maxlen + 1)

        prob = self.priority.prior / (np.sum(self.priority.prior)+1e-3)

        a = [i for i in range(len(prob))]
        try:
            idx = np.random.choice(a, batch_size, p=prob)
        except ValueError:
            # print("Probability is WRONG?")
            d = 1 - sum(prob)
            prob[-1] += d
            idx = np.random.choice(a, batch_size, p=prob)
        bin_data =[self.memory[id] for id in idx]
        s_prob = prob[idx]
        
        return list(bin_data), s_prob, idx
    
    def remove_to_fit(self):
        len_memory = len(self.memory)
        if len_memory <= self.maxlen:
            return 
        else:
            delta = len(self.memory) - self.maxlen
            del self.memory[:delta]
            del self.priority[:delta]

    @property
    def max_weight(self):
        min_value = min(self.priority.prior)
        max_weight = 1 / (len(self.priority) * min_value / np.sum(self.priority.prior)) ** self.beta
        return max_weight