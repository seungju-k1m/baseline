import torch
import torch.nn as nn

from copy import deepcopy
from baseline.utils import constructNet


class Node:
    """
        Node can send output to other nodes, also, collect inputs from others.

        To do this, node must specify the previous nodes.

        the id of node is a kind of string name such as "module00"

        the list of previous nodes is stored when node is initialized

        Also, node have priority used for controling the forwarding flow.
    """

    def __init__(self, data: dict):
        super(Node, self).__init__()

        self.previousNodes: list
        self.previousNodes = []
        self.priority = data["prior"]
        self.storedInput = []
        self.storedOutput = []
        self.data = data

    def setPrevNodes(self, prevNodes):
        prevNodes: list
        self.previousNodes = prevNodes

    def buildModel(self) -> None:
        self.model = constructNet(self.data)

    def clear_savedOutput(self) -> None:

        self.storedInput.clear()
        self.storedOutput.clear()

    def addInput(self, _input) -> None:
        if type(_input) == tuple:
            # this is for hidden states of LSTM
            self.storedInput.append(deepcopy(_input))
        else:
            self.storedInput.append(_input.clone())

    def getStoreOutput(self):
        return self.storedOutput

    def step(self) -> None:
        if len(self.previousNodes) != 0:
            for prevNode in self.previousNodes:
                for prevInput in prevNode.storedOutput:
                    self.storedInput.append(prevInput)
        storedInput = tuple(self.storedInput)
        output = self.model.forward(storedInput)
        self.storedOutput.append(output)


class baseAgent(nn.Module):
    """
        baseAgent can be thought as the collective of nodes.

        baseAgent performs tasks such as

            1. parse the network infomation.

            2. build the network which consists of nodes.

            3. To build optimizer, return the list of weights

            4. update weights from other identical base Agents.

            5. return the norm of gradient

            6. clip the gradients

            7. LSTM control methods

            8. Forwarding
    """

    def __init__(self, mData: dict):
        super(baseAgent, self).__init__()
        # data
        self.mData = mData

        # name
        self.moduleNames = list(self.mData.keys())

        self.LSTMMODULENAME = []
        # sorting the module layer
        self.moduleNames.sort()

        (
            self.priorityModel,
            self.outputModelName,
            self.inputModelName,
        ) = self.buildModel()
        self.loadParameters()
        self.priority = list(self.priorityModel.keys())
        self.priority.sort()
        

    def buildModel(self) -> tuple:
        priorityModel = {}
        """
            key : priority
            element : dict,
                    key name
                    element node
        """

        outputModelName = []
        """
            element: list
                    prior, module name
        """

        inputModelName = {}
        """
            key : num of input
            element : list
                    element: [priority, name]
        """

        name2prior = {}
        """
            key : name of layer
            element: prior
        """

        for name in self.moduleNames:
            data = self.mData[name]
            if data["netCat"] == "LSTMNET" or data['netCat'] == "GRU":
                self.LSTMMODULENAME.append(name)
            data: dict
            name2prior[name] = data["prior"]
            if data["prior"] in priorityModel.keys():
                priorityModel[data["prior"]][name] = Node(data)
                priorityModel[data["prior"]][name].buildModel()
            else:
                priorityModel[data["prior"]] = {name: Node(data)}
                priorityModel[data["prior"]][name].buildModel()

            if "output" in data.keys():
                if data["output"]:
                    outputModelName.append([data["prior"], name])

            if "input" in data.keys():
                for i in data["input"]:
                    if i in inputModelName.keys():
                        inputModelName[i].append([data["prior"], name])
                    else:
                        inputModelName[i] = [[data["prior"], name]]

        for prior in priorityModel.keys():
            node_dict = priorityModel[prior]
            for index in node_dict.keys():
                node = node_dict[index]
                if "prevNodeNames" in node.data.keys():
                    prevNodeNames = node.data["prevNodeNames"]
                    prevNodeNames: list
                    prevNodes = []
                    for name in prevNodeNames:
                        data = self.mData[name]
                        prevNodes.append(priorityModel[data["prior"]][name])
                    node.setPrevNodes(prevNodes)
        self.name2prior = name2prior
        return priorityModel, outputModelName, inputModelName

    def loadParameters(self) -> None:
        for prior, priorDict in self.priorityModel.items():
            for name, module in priorDict.items():
                setattr(self, name, module.model)

    def buildOptim(self) -> tuple:
        listLayer = []
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                listLayer.append(layerDict[name].model)

        return tuple(listLayer)

    def updateParameter(self, Agent, tau) -> None:
        """
        tau = 0, no change
        """
        Agent: AgentV2
        tau: float

        with torch.no_grad():
            for prior in self.priority:
                layerDict = self.priorityModel[prior]
                for name in layerDict.keys():
                    parameters = layerDict[name].model.parameters()
                    tParameters = Agent.priorityModel[prior][name].model.parameters()
                    for p, tp in zip(parameters, tParameters):
                        p.copy_((1 - tau) * p + tau * tp)

    def calculateNorm(self) -> float:
        totalNorm = 0
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                parameters = layerDict[name].model.parameters()
                for p in parameters:
                    norm = p.grad.data.norm(2)
                    totalNorm += norm

        return totalNorm

    def evalMode(self):
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                node = layerDict[name]
                node.model.eval()

    def trainMode(self):
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                node = layerDict[name]
                node.model.train()

    def clippingNorm(self, maxNorm):
        inputD = []
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                inputD += list(layerDict[name].model.parameters())

        torch.nn.utils.clip_grad_norm_(inputD, maxNorm)
    
    def getParameters(self):
        inputD = []
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                inputD += list(layerDict[name].model.parameters())
        return inputD

    def getCellState(self, name=None):
        if name is None:
            name = self.LSTMMODULENAME[0]

        prior = self.name2prior[name]
        return self.priorityModel[prior][name].model.getCellState()

    def setCellState(self, cellstate, name=None):
        if name is None:
            name = self.LSTMMODULENAME[0]
        prior = self.name2prior[name]
        self.priorityModel[prior][name].model.setCellState(cellstate)

    def zeroCellState(self, num=1, name=None):
        if name is None:
            name = self.LSTMMODULENAME[0]
        prior = self.name2prior[name]
        self.priorityModel[prior][name].model.zeroCellState(num)

    def detachCellState(self, name=None):
        if name is None:
            name = self.LSTMMODULENAME[0]
        prior = self.name2prior[name]
        self.priorityModel[prior][name].model.detachCellState()

    def to(self, device) -> None:
        for prior in self.priority:
            layerDict = self.priorityModel[prior]
            for name in layerDict.keys():
                node = layerDict[name]
                node.model.to(device)

    def clear_savedOutput(self):
        for i in self.priority:
            nodeDict = self.priorityModel[i]
            for name in nodeDict.keys():
                node = nodeDict[name]
                node.clear_savedOutput()

    def forward(self, inputs) -> tuple:
        inputs: tuple

        for i, _input in enumerate(inputs):
            priorityName_InputModel = self.inputModelName[i]
            priorityName_InputModel: list
            for inputinfo in priorityName_InputModel:
                self.priorityModel[inputinfo[0]][inputinfo[1]].addInput(_input)

        for prior in range(self.priority[-1] + 1):
            for nodeName in self.priorityModel[prior].keys():
                node: str
                self.priorityModel[prior][nodeName].step()

        output = []
        for outinfo in self.outputModelName:
            out = self.priorityModel[outinfo[0]][outinfo[1]].getStoreOutput()
            for o in out:
                output.append(o)

        output = tuple(output)
        self.clear_savedOutput()
        return output
