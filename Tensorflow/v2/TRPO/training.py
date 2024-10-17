import multiprocessing
import threading

from actorcritcmodel import ActorCriticModel
from ppo import PPOWorker


class Training():
    def __init__(self, params, env_name, logger):
        self.params = params
        self.env_name = env_name
        self.logger = logger
        self.lock = threading.Lock()
        self.solution = Solution()
        self.ThreadNum = 0
        actorcritic = ActorCriticModel(params, logger)
        self.model = actorcritic.createModel()
        self.threads = params.threads

    def run(self):
        print("multiprocessing.cpu_count()", multiprocessing.cpu_count())
        workers = [PPOWorker(self.params, self.env_name, self.logger, self.model, self.lock, self.solution)
                   for i in range(self.threads)]

        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()

        [w.join() for w in workers]

class Solution():
    def __init__(self):
        self.ThreadNum = 0
        self.MaxNonZero = 0
        self.MaxSlot = 0
        self.FoundBest = False

    def incrementThreadCount(self):
        self.ThreadNum = self.ThreadNum + 1

    def getThreadCount(self):
        return self.ThreadNum

    def update(self, maxnonzero, maxslot):
        self.MaxNonZero = maxnonzero
        self.MaxSlot = maxslot

    def get(self):
        return self.MaxNonZero, self.MaxSlot

