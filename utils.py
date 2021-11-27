import gc
import sys
import json
import torch
import random
import logging

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
    LSTM,
    Select,
    Permute,
    GRU,
    Attention,
    Stack,
    Subtrack,
    Add,
    Mean
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


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='a')
    stream = logging.StreamHandler()
    # handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(stream)

    return logger


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
                inputD, lr=lr, weight_decay=decay, eps=eps, betas=(
                    beta1, beta2)
            )
        if name == "sgd":
            momentum = 0 if "momentum" not in keyList else optimData["momentum"]

            optim = torch.optim.SGD(
                inputD, lr=lr, weight_decay=decay, momentum=momentum
            )
        if name == "rmsprop":
            momentum = 0 if "momentum" not in keyList else optimData["momentum"]
            optim = torch.optim.RMSprop(
                inputD, lr=lr, weight_decay=decay, eps=eps, momentum=momentum
            )

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
        LSTM,
        Select,
        Permute,
        GRU,
        Attention,
        Stack,
        Subtrack,
        Add,
        Mean
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
        "LSTM",
        "Select",
        "Permute",
        "GRU",
        "Attention",
        "Stack",
        "Substract",
        'Add',
        "Mean"
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

    def extend(self, d, priorMode=False):
        # for d in datum:
        if priorMode:
            for _d in d:
                self.append(_d)
        else:
            self.append(d)

    def __getitem__(self, idx):
        return loads(super(CompressedDeque, self).__getitem__(idx))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = CompressedDeque(maxlen=capacity)

    def push(self, data):
        self.memory.append(data)

    def sample(self, batch_size):
        datas = random.sample(self.memory, batch_size)
        # data = []
        # for d in datas:
        #     data.append(loads(d))
        return datas

    def clear(self):
        self.memory.clear()

    def __getitem__(self, idx):
        return self.memory[idx]

    def __len__(self):
        return len(self.memory)


class PrioritizedMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.transitions = CompressedDeque()
        self.priorities = SumTree()

    def push(self, transitions, priorities):
        self.transitions.extend(transitions, priorMode=True)
        self.priorities.extend(priorities)
        # gc.collect()

    def sample(self, batch_size):
        idxs, prios = self.priorities.prioritized_sample(batch_size)
        # gc.collect()
        return [self.transitions[i] for i in idxs], prios, idxs

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
        # gc.collect()

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


class INFO:
    def __init__(self):
        self.info = """
    Configuration for this experiment
    """

    def __add__(self, string):
        self.info += string
        return self

    def __str__(self):
        return self.info


def writeDict(info, data, key, n=0):
    tab = ""
    for _ in range(n):
        tab += "\t"
    if type(data) == dict:
        for k in data.keys():
            dK = data[k]
            if type(dK) == dict:
                info += """
        {}{}:
            """.format(
                    tab, k
                )
                writeDict(info, dK, k, n=n + 1)
            else:
                info += """
        {}{}:{}
        """.format(
                    tab, k, dK
                )
    else:
        info += """
        {}:{}
        """.format(
            key, data
        )


def writeTrainInfo(datas):
    info = INFO()
    key = datas.keys()
    for k in key:
        data = datas[k]
        if type(data) == dict:
            info += """
        {}:
        """.format(
                k
            )

            writeDict(info, data, k, n=1)
        else:
            writeDict(info, data, k)

    return info
