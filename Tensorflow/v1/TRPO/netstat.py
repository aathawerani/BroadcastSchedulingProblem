import numpy as np
from networkgenerator import NetworkGenerator

class NetworkStats():
    def __init__(self, path, logger):
        self.netstatlist = []
        self.path = path
        self.logger = logger

    def CalculateStats(self):
        for i in range(1, 21):
            for j in range(1, 4):
                case = "case" + str(i) + "r" + str(j)
                print("reading case", case)
                self.AddToList(case)
        counter = 0
        for i in range(20):
            averageedges = 0
            averagemax = 0
            averagemin = 0
            nodes = 0
            case = ''
            for j in range(3):
                ns = self.netstatlist[i*3+j]
                averageedges += ns.NumberOfEdges
                averagemax += ns.MaxDegree
                averagemin += ns.MinDegree
                nodes = ns.NumberOfNodes
                case = ns.case
                self.logger.Info("case", ns.case, "NumberOfNodes", ns.NumberOfNodes, "NumberOfEdges", ns.NumberOfEdges,
                      "MaxDegree", ns.MaxDegree, "MinDegree", ns.MinDegree)
            averageedges /=  3
            averagemax /= 3
            averagemin /= 3
            self.logger.Info("case", case, "NumberOfNodes", nodes, "AverageEdges", averageedges, "AverageMaxDegree", averagemax,
                  "AverageMin", averagemin)

    def AddToList(self, case):
        ng = NetworkGenerator(self.path)
        adjacent = ng.ReadNetwork2(case)
        netstat = NetStat()
        netstat.Update(case, adjacent)
        self.netstatlist.append(netstat)

    def CalculateStats2(self, case, adjacent):
        ns = NetStat()
        ns.Update(case, adjacent)
        self.logger.Info("case", ns.case, "NumberOfNodes", ns.NumberOfNodes, "NumberOfEdges", ns.NumberOfEdges,
              "MaxDegree", ns.MaxDegree, "MinDegree", ns.MinDegree)
        return ns.MaxDegree

class NetStat():
    def __init__(self):
        self.case = ""
        self.NumberOfNodes = 0
        self.NumberOfEdges = 0
        self.MaxDegree = 0
        self.MinDegree = 0
        #self.SumEdges = 0

    def Update(self, c, adj):
        self.case = c
        self.NumberOfNodes = len(adj)
        nodeedges = np.zeros((self.NumberOfNodes))
        for a in range(self.NumberOfNodes):
            for b in range(self.NumberOfNodes):
                if adj[a][b] == 1:
                    self.NumberOfEdges += 1
                    nodeedges[a] += 1
                    nodeedges[b] += 1
        #print(nodeedges)
        self.MaxDegree = np.max(nodeedges) / 2
        self.MinDegree = np.min(nodeedges) / 2
        #self.SumEdges = np.sum(nodeedges)
