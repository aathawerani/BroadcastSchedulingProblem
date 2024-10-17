import uuid
import config
from utils import Logger

class BestSchedule():
    def __init__(self, lowerbound, state, logname, numnodes, env_name):
        self.LowerBound = lowerbound
        self.NumNodes = numnodes

        self.BestSchedule = state
        self.state = state
        self.FoundBest = False
        self.BestNodesAllocated = 0
        self.NodesAllocated = 0
        self.BestMinSlotAssigned = 0
        self.MinSlotAssigned = 0
        self.BestMaxSlotAssigned = 0
        self.MaxSlotAssigned = 0

        self.uuid = uuid.uuid4()
        self.logname = logname
        self.logger = Logger(logname=logname)

        self.params = config.GetNetworkParams(env_name)
        self.MaxIterations = self.params.max_iterations
        #print("self.MaxIterations", self.MaxIterations)
        self.Iterations = 0

        self.logger.write("Lower Bound", lowerbound)

    def UpdateBestSchedule(self, nodesallocated, state, slotnumber):
        self.NodesAllocated = nodesallocated
        self.state = state
        self.MaxSlotAssigned = slotnumber
        self.Iterations += 1

        if self.FoundBest:
            print("Already found best")
            return
        coverage = self.CheckCoverage2()
        if coverage == 1:
            self.UpdateBestSchedule2()
            self.CheckBestSchedule()
            return
        elif coverage == 2:
            if self.CheckCompression():
                self.UpdateBestSchedule2()
                self.CheckBestSchedule()
            else:
                return

        if self.Iterations > self.MaxIterations:
            #print("self.MaxIterations", self.MaxIterations)
            #print("self.Iterations", self.Iterations)
            foundmaxlog = Logger(self.logname + "_" + "max")
            foundmaxlog.write(self.uuid, "Max iterations reached", self.BestSchedule)

    def CheckCoverage2(self):
        if self.NodesAllocated > self.BestNodesAllocated:
            return 1
        elif self.NodesAllocated > 0 and self.NodesAllocated == self.BestNodesAllocated:
            return 2
        else:
            return 3

    def CheckCompression(self):
        if self.MaxSlotAssigned < self.BestMaxSlotAssigned:
            return True
        else:
            return False

    def CheckBestSchedule(self):
        self.logger.write(self.uuid, "Schedule", self.BestSchedule, "BestNodesAllocated", self.BestNodesAllocated,
                          "BestMinSlotAssigned", self.BestMinSlotAssigned, "BestMaxSlotAssigned", self.BestMaxSlotAssigned)
        if self.BestMinSlotAssigned > 0 and self.BestMaxSlotAssigned <= self.LowerBound:
            self.logger.printwrite("FOUND BEST", self.BestSchedule)
            self.FoundBest = True
            foundbestlog = Logger(self.logname + "_" + "best")
            foundbestlog.write(self.uuid, "FOUND BEST", self.BestSchedule)

    def UpdateBestSchedule2(self):
        self.Iterations = 0
        self.BestSchedule = self.state
        self.BestNodesAllocated = self.NodesAllocated
        self.BestMaxSlotAssigned = self.MaxSlotAssigned
        if self.NodesAllocated < self.NumNodes:
            self.BestMinSlotAssigned = 0
        else:
            self.BestMinSlotAssigned = 1
