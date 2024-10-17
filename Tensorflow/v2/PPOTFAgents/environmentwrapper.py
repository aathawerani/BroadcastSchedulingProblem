import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from environment import BSPEnvironment

class BSPEnvironmentWrapper(py_environment.PyEnvironment):
    def __init__(self, env_name, logname):

        self.bsp = BSPEnvironment(env_name, logname)

        self.NumNodes = self.bsp.get_numnodes()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=1, maximum=self.NumNodes + 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.NumNodes,), dtype=np.float32, minimum=0, maximum=self.NumNodes, name='observation')
        self._state = self.bsp.get_state()
        self._episode_ended = False

    def get_state(self):
        return self.bsp.get_state()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.bsp.reset()
        self._state = self.bsp.get_state()
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        self._state, reward = self.bsp.BSPstep(action)

        if self.bsp.isDone() == True:
            self._episode_ended = True

        if self._episode_ended:
            self.bsp.UpdateBestSchedule()
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward = reward, discount = self.bsp.get_discountfactor())
