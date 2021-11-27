import numpy as np
from numpy.lib.arraysetops import isin
from baseline.utils import CompressedDeque
from copy import deepcopy


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
        if bias == 0:
            max_value = max(vals)
            if max_value > self.max_value:
                self.max_value = max_value
            self.prior[idx] = vals
        else:
            max_value = max(vals)
            if max_value > self.max_value:
                self.max_value = max_value
            idx = np.array(idx)
            idx_check = idx > bias
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
        self.memory = np.empty(0)
        self.priority = Tree()
        self.maxlen = maxlen
        self.max_value = max_value

        self.bias = 0
        self.switch = False
        # self.alpha = alpha

    def push(self, d): 
        n = len(d)
        self.memory = np.append(self.memory, deepcopy(d))
        self.priority.push(n)
        len_memory = len(self.memory)
        if len_memory > self.maxlen:
            self.switch = True
            delta = len_memory - self.maxlen
            self.bias += delta
            x = [i for i in range(delta)]
            np.delete(self.memory, x)
            np.delete(self.priority.prior, x)
            
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
        self.priority.update(idx, vals, bias=self.bias)
        self.bias = 0
        
    def sample(self, batch_size):
        prob = self.priority.prior / np.sum(self.priority.prior)
        a = [i for i in range(len(prob))]
        # print(len(a), len(prob))
        idx = np.random.choice(a, batch_size, p=prob)
        bin_data = self.memory[idx]
        s_prob = prob[idx]
        return list(bin_data), s_prob, idx
    @property
    def max_weight(self):
        return self.priority.max_value