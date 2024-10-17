import numpy as np

import config
from environmentgenerator import EnvGen
from bestschedule import BestSchedule

class BSPEnvironment():
    def __init__(self, env_name, logname, thread):
        self.thread = thread
        self.env_name = env_name
        self.params = config.GetNetworkParams(env_name)
        self.casespath = self.params.cases
        envgen = EnvGen()
        self.adjacent = envgen.GetNetwork(env_name, self.casespath)
        self.NumNodes = len(self.adjacent)
        self.slotNumber = 1
        self.chances = self.params.chances
        self.remainingchances = self.params.chances
        NumberOfNodes, NumberOfEdges, self.MaxDegree, MinDegree, self.nodeedges = envgen.GetNetworkStat(env_name, self.casespath)
        #print("self.nodeedges", self.nodeedges)
        self.reward_multiplier = self.params.reward_multiplier
        self.increaseChances = 10
        self.iterations = 0
        self.LowerBound = self.MaxDegree + 1
        self.NodesAllocated = 0
        self.node_selection = self.params.node_selection
        self.CurrentSlotNodes = []
        self.CurrentSlotNodesCount = 0
        self.reward = 0
        self.totalchances = self.chances
        #self._state = np.zeros(self.NumNodes, dtype=np.float32)
        self._state = np.zeros(self.NumNodes, dtype=np.short)
        #print("self.state.shape", self.state.shape)
        #self.state = np.zeros((self.NumNodes, self.NumNodes), dtype=np.short)
        #self.state = np.zeros((1, self.NumNodes), dtype=np.float32)
        self.schedule = BestSchedule(self.LowerBound, self._state, logname, self.NumNodes, env_name, thread)

    def get_state(self):
        return self._state

    def get_numnodes(self):
        return self.NumNodes

    def get_discountfactor(self):
        return self.params.discount_factor

    def reset(self):
        self.slotNumber = 1
        self.iterations += 1
        if self.iterations % self.increaseChances == 0:
            self.chances += 1
        self.remainingchances = self.chances
        #self._state = np.zeros(self.NumNodes, dtype=np.float32)
        self._state = np.zeros(self.NumNodes, dtype=np.short)
        #self.state = np.zeros((self.NumNodes, self.NumNodes), dtype=np.short)
        #self.state = np.zeros((1, self.NumNodes), dtype=np.float32)
        self.NodesAllocated = 0
        self.CurrentSlotNodes = []
        self.CurrentSlotNodesCount = 0
        self.reward = 0
        self.totalchances = self.chances

    def isDone(self):
        if self.remainingchances == 0 :
            if self.CurrentSlotNodesCount == 0:
                #self.slotNumber -= 1
                return True
            self.slotNumber += 1
            self.remainingchances = self.chances
            self.CurrentSlotNodes = []
            self.CurrentSlotNodesCount = 0
            self.reward = 0
            #self.state = np.pad(self.state, ((0, 1), (0, 0)), mode='constant')
            #self.totalchances -= 1

        #if self.slotNumber > self.LowerBound:
            #return True

        if self.NodesAllocated < self.NumNodes:
            return False
        return True

    def BSPstep(self, action):
        # print("action", action)
        node = action
        legal = self.checkActionLegal(node)
        reward = 0
        if legal:
            self._state[node] = self.slotNumber
            #self.state[self.slotNumber - 1, node] = 1
            #self.state[0, node] = 1
            # print("self.slotNumber", self.slotNumber)
            self.NodesAllocated += 1
            #print(node, self.slotNumber, self.thread, self.NodesAllocated, "after update state", self._state)
            #self.reward += 1
            #reward = self.reward
            # if self.slotNumber <= self.LowerBound:
            #print("node", node, "self.nodeedges[node]", self.nodeedges[node])
            reward = 1
            # else:
            # reward = 0
        else:
            self.remainingchances -= 1
            reward = 0
        return self._state, reward

    def checkActionLegal(self, node):
        #print("check action state", self.state)
        if node < 0 or node >= self.NumNodes:
            #print("check action legal returning false 1")
            return False
        if self._state[node] > 0:
        #print(self.state.shape, )
        #print(self.slotNumber)
        #print(node)
        #if self.state[self.slotNumber - 1, node] > 0:
            #print("check action legal returning false 2")
            return False
        for node1 in self.CurrentSlotNodes:
            if self.adjacent[node][node1] > 0:
                #print("check action legal returning false 3")
                return False
        self.CurrentSlotNodes.append(node)
        self.CurrentSlotNodesCount += 1
        #print("check action legal returning true")
        return True

    def UpdateBestSchedule(self):
        self.schedule.UpdateBestSchedule(self.NodesAllocated, self._state, self.slotNumber)