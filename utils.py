import json
import torch
import random

import numpy as np
import _pickle as cPickle
import torchvision.transforms.functional as TF

from collections import deque
from baseline.sumtree import SumTree
from baseline.baseNetwork import (
    MLP,
    CNET,
    LSTMNET,
    CNN1D,
    Cat,
    Unsequeeze,
    View,
    CNNTP2D,
    GovAvgPooling,
    RESCONV2D,
    RESCONV1D,
    AvgPooling,
)


"""
utils의 경우, 다양한 상황에서 사용되는 기타 method이다.
"""


def calGlobalNorm(agent):
    """
    agent의 graident의 norm/sum을 구한다.
    """
    totalNorm = 0
    for p in agent.parameters():
        norm = p.grad.data.norm(2)
        totalNorm += norm
    return totalNorm


def clipByGN(agent, maxNorm):
    totalNorm = calGlobalNorm(agent)
    for p in agent.parameters():
        factor = maxNorm / np.maximum(totalNorm, maxNorm)
        p.grad *= factor


def getOptim(optimData, agent, floatV=False):

    """
    configuration에서 정의된 optimizer setting을 지원한다.

    args:
        optimData:
            name:[str] optimizer의 이름
            lr:[float] learning rate
            decay:[float] decaying(L2 Regularzation)
            eps:[float], 
            clipping:deprecated
        agent:[tuple, torch.nn], 해당 optimizer가 담당하는 weight들 Agentv1.buildOptim을 통해서 호출
        floatV:[bool], weight이 torch.nn이 아니라 tensor인 경우
            
    """

    keyList = list(optimData.keys())

    if "name" in keyList:
        name = optimData["name"]
        lr = optimData["lr"]
        decay = 0 if "decay" not in keyList else optimData["decay"]
        eps = 1e-5 if "eps" not in keyList else optimData["eps"]

        if floatV:
            inputD = [agent]
        elif type(agent) == tuple:
            inputD = []
            for a in agent:
                inputD += list(a.parameters())

        else:
            inputD = agent.parameters()
        if name == "adam":
            beta1 = 0.9 if "beta1" not in keyList else optimData["beta1"]
            beta2 = 0.99 if "beta2" not in keyList else optimData["beta2"]
            optim = torch.optim.Adam(
                inputD, lr=lr, weight_decay=decay, eps=eps, betas=(beta1, beta2)
            )
        if name == "sgd":
            momentum = 0 if "momentum" not in keyList else optimData["momentum"]

            optim = torch.optim.SGD(
                inputD, lr=lr, weight_decay=decay, momentum=momentum
            )
        if name == "rmsprop":
            optim = torch.optim.RMSprop(inputD, lr=lr, weight_decay=decay, eps=eps)

    return optim


def getActivation(actName, **kwargs):
    if actName == "relu":
        act = torch.nn.ReLU()
    if actName == "leakyRelu":
        nSlope = 0.2 if "slope" not in kwargs.keys() else kwargs["slope"]
        act = torch.nn.LeakyReLU(negative_slope=nSlope)
    if actName == "sigmoid":
        act = torch.nn.Sigmoid()
    if actName == "tanh":
        act = torch.nn.Tanh()
    if actName == "linear":
        act = None

    return act


def constructNet(netData):
    """
    configuration에 따라 해당 network를 반환
    
    args:
        netData:dict
    """
    netCat = netData["netCat"]
    if netCat == "Input":
        return None
    Net = [
        MLP,
        CNET,
        LSTMNET,
        CNN1D,
        Cat,
        Unsequeeze,
        View,
        CNNTP2D,
        GovAvgPooling,
        RESCONV2D,
        RESCONV1D,
        AvgPooling,
    ]
    netName = [
        "MLP",
        "CNN2D",
        "LSTMNET",
        "CNN1D",
        "Cat",
        "Unsequeeze",
        "View",
        "CNNTP2D",
        "GovAvgPooling",
        "RESCONV2D",
        "RESCONV1D",
        "AvgPooling",
    ]
    ind = netName.index(netCat)

    baseNet = Net[ind]
    network = baseNet(netData)

    return network


def setValue_dict(Dict, Keys, Values):
    Dict: dict
    Key: list
    Values: list

    for key, value in zip(Keys, Values):
        if key not in Dict.keys():
            Dict[key] = value
    return Dict


def dumps(data):
    return cPickle.dumps(data)


def loads(packed):
    return cPickle.loads(packed)


class jsonParser:
    """
    configuration은 *.json 형태이기 때문에
    이를 dictionary형태로 변환시켜주는 class
    """

    def __init__(self, fileName):
        with open(fileName) as jsonFile:
            self.jsonFile = json.load(jsonFile)
            self.jsonFile: dict
        # keys = [
        #     'time_scale', 'RecordScore', 'no_graphics',
        #     'LSTMNum', 'gamma', 'lambda', 'rScaling',
        #     'entropyCoeff', 'epsilon', 'div', 'epoch',
        #     'updateOldP', 'initLogStd', 'finLogStd',
        #     'annealingStep', 'K1', 'K2', 'updateStep']

        # values = [
        #     1, 1e6, False,
        #     -1, 0.99, 0.95, 1,
        #     0, 0.2, 1, 1,
        #     4, -1.1, -1.5,
        #     1e6, 160, 10, 160
        # ]
        # self.jsonFile = setValue_dict(self.jsonFile, keys, values)

    def loadParser(self):
        return self.jsonFile

    def loadAgentParser(self):
        agentData = self.jsonFile.get("agent")
        agentData["sSize"] = self.jsonFile["sSize"]
        agentData["aSize"] = self.jsonFile["aSize"]
        agentData["device"] = self.jsonFile["device"]
        agentData["gamma"] = self.jsonFile["gamma"]
        return agentData

    def loadOptParser(self):
        return self.jsonFile.get("optim")


class CompressedDeque(deque):
    def __init__(self, *args, **kargs):
        super(CompressedDeque, self).__init__(*args, **kargs)

    def __iter__(self):
        return (loads(v) for v in super(CompressedDeque, self).__iter__())

    def append(self, data):
        super(CompressedDeque, self).append(dumps(data))

    def extend(self, datum):
        for d in datum:
            self.append(d)

    def __getitem__(self, idx):
        return loads(super(CompressedDeque, self).__getitem__(idx))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = CompressedDeque(capacity)

    def push(self, data):
        self.memory.append(data)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()

    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)


class PrioritizedMemory(object):
    def __init__(self, capacity, use_compress=False):
        self.capacity = capacity
        self.transitions = CompressedDeque()
        self.priorities = SumTree()

    def push(self, transitions, priorities):
        self.transitions.extend(transitions)
        self.priorities.extend(priorities)

    def sample(self, batch_size):
        idxs, prios = self.priorities.prioritized_sample(batch_size)
        return [self.transitions[i] for i in idxs], prios, idxs

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def remove_to_fit(self):
        if len(self.priorities) - self.capacity <= 0:
            return
        for _ in range(len(self.priorities) - self.capacity):
            self.transitions.popleft()
            self.priorities.popleft()

    def __len__(self):
        return len(self.transitions)

    def total_prios(self):
        return self.priorities.root.value
