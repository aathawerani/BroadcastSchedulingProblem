import gym
from gym import spaces
import numpy as np


from environment import BSPEnvironment

class BSPEnvironmentWrapper(gym.Env):
    def __init__(self, env_name, logname, thread):
        self.thread = thread
        self.bsp = BSPEnvironment(env_name, logname, thread)

        self.NumNodes = self.bsp.get_numnodes()

        self.observation_space = spaces.Discrete(self.NumNodes)  # Each user observes its own slot
        #self.action_space = spaces.MultiDiscrete([self.NumNodes])  # Each user selects a slot to transmit
        self.action_space = spaces.Discrete(self.NumNodes)

        self.state = self.bsp.get_state()
        self._episode_ended = False

    def get_env_name(self):
        return self.bsp.env_name
    
    def get_state(self):
        return self.bsp.get_state()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reset(self):
        self.bsp.reset()
        self.state = self.bsp.get_state()
        self._episode_ended = False
        #return ts.restart(self._state)
        return self.state, "info"

    def step(self, action):
        if self._episode_ended:
            return self.reset()

        self.state, reward = self.bsp.BSPstep(action)

        if self.bsp.isDone() == True:
            self._episode_ended = True

        if self._episode_ended:
            self.bsp.UpdateBestSchedule()
            #return ts.termination(self._state, reward)
            return self.state, reward, True, {}
        else:
            #return ts.transition(self._state, reward = reward, discount = self.bsp.get_discountfactor())
            return self.state, reward, False, {}
        
        
