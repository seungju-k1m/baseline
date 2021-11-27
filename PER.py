import numpy as np
from numpy.lib.arraysetops import isin
from baseline.utils import CompressedDeque


class Tree:

    def __init__(self, value=1.0, maxlen=10000):
        self.max_value = value
        self.prior = np.ones(0)
        self.idx = []
        self.maxlen = maxlen
        self.alpha = 1.0
    
    def push(self):
        self.prior = np.append(
            self.prior, self.max_value ** self.alpha)
        if len(self.prior) > self.maxlen:
            np.delete(self.prior , 0)
    
    def update_max_value(self, value):
        self.max_value = value
    
    def update(self, idx:list, vals:np.ndarray):
        max_value = max(vals)
        if max_value > self.max_value:
            self.max_value = max_value
        self.prior[idx] = vals
    
    def __len__(self):
        return len(self.prior)
        

class PER:
    def __init__(
        self,
        maxlen=1000,
        max_value=1.0,
        alpha=1.0):
        self.length=0
        # self.memory = CompressedDeque(maxlen=maxlen)
        self.memory = np.empty(0)
        self.priority = Tree()
        self.maxlen = maxlen
        self.max_value = max_value
        self.alpha = alpha

    def push(self, d): 
        self.memory = np.append(self.memory, d)
        self.priority.push()
        if len(self.memory) > self.maxlen:
            np.delete(self.memory, 0)
    
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
        prob = self.priority / np.sum(self.priority)
        idx = np.random.choice(len(self.priority), batch_size, p=prob)
        bin_data = self.memory[idx]
        s_prob = prob[idx]
        return list(bin_data), s_prob, idx
    @property
    def max_weight(self):
        return self.priority.max_value