import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, Reshape
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K

import threading
from threading import Thread, Lock

import config
from utils import Logger
from environmentwrapper import BSPEnvironmentWrapper

def OurModel(input_shape, action_space, lr):
    X_input = Input(input_shape)
    #X = Flatten(input_shape=input_shape)(X_input)
    #X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)
    #X = LSTM(64, activation="elu", kernel_initializer='he_uniform')(X_input)
    X = LSTM(64, activation="elu", kernel_initializer='he_uniform', return_sequences=True)(X_input)
    X = LSTM(64, activation="elu", kernel_initializer='he_uniform')(X)
    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, activation='linear', kernel_initializer='he_uniform')(X)

    def ppo_loss(y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+action_space], y_true[:, 1+action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 5e-3

        prob = y_pred * actions
        old_prob = actions * prediction_picks
        r = prob/(old_prob + 1e-10)
        p1 = r * advantages
        p2 = K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        loss =  -K.mean(K.minimum(p1, p2) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))

        return loss
        
    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss=ppo_loss, optimizer=RMSprop(lr=lr))

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    return Actor, Critic

class PPOAgent:
    def __init__(self, env_name):
        self.env_name = env_name 

        params = config.GetNetworkParams(self.env_name)
        self._num_nodes = params.num_nodes
        self.action_size = params.action_size
        self.EPISODES, self.episode = params.episodes, 0 # specific for pong
        self.lock = Lock() # lock all to update parameters without other thread interruption
        self.lr = params.learning_rate
        self.EPOCHS = params.num_epochs

        self.scores, self.episodes, self.average = [], [], []
        input_shape = (self._num_nodes, self._num_nodes)  # Initial input shape

        #self.Actor, self.Critic = OurModel(input_shape=self.state_size, action_space = self.action_size, lr=self.lr)
        self.Actor, self.Critic = OurModel(input_shape=input_shape, action_space = params.action_size, lr=self.lr)

    def act(self, state):
        #print("state.shape", state.shape)
        #state = state.reshape(1, 1, -1)
        state = state.reshape(1, self._num_nodes, -1)
        #print("state.shape", state.shape)
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        #print("action", action)
        return action, prediction

    def discount_rewards(self, reward):
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    def replay(self, states, actions, rewards, predictions):
        states = np.vstack(states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        discounted_r = np.vstack(self.discount_rewards(rewards))
        #print("states.shape", states.shape)
        values = self.Critic.predict(states)
        advantages = discounted_r - values

        y_true = np.hstack([advantages, predictions, actions])
        
        self.Actor.fit(states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))
        self.Critic.fit(states, discounted_r, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))
 
    def reset(self, env):
        return env.reset()

    def step(self, action, env, image_memory):
        next_state, reward, done, info = env.step(action)
        #next_state = self.GetImage(next_state, image_memory)
        return next_state, reward, done, info
    
    def train(self, n_threads):
        threads = [threading.Thread(
                target=self.train_threading,
                daemon=True,
                args=(self,
                    #envs[i],
                    i)) for i in range(n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()

        for t in threads:
            time.sleep(10)
            t.join()
            
    def train_threading(self, agent, thread):
        params = config.GetNetworkParams(self.env_name)
        self.logpath = params.logpath
        date1 = datetime.utcnow().strftime("%y%m%d")  # create unique directories
        time1 = datetime.utcnow().strftime("%H%M%S")  # create unique directories
        logname = self.env_name + "_" + date1 + time1
        bestname = logname + "_" + "best"
        maxname = logname + "_" + "max"
        logger = Logger(logname)
        foundbestpath = os.path.join(self.logpath, bestname + ".log")
        foundmaxpath = os.path.join(self.logpath, maxname + ".log")

        logger.printwrite(thread, "Start", self.env_name)
        logger.write(params)

        env = BSPEnvironmentWrapper(self.env_name, logname, thread) 

        while self.episode < self.EPISODES:
            score, done = 0, False
            state = self.reset(env)
            states, actions, rewards, predictions = [], [], [], []
            while not done:
                action, prediction = agent.act(state)
                next_state, reward, done, _ = self.step(action, env, state)
                #print("state.shape", state.shape)
                state_reshaped = state.reshape(1, self._num_nodes, -1)
                #print("state_reshaped.shape", state_reshaped.shape)

                #print("Shape of state_reshaped before concatenation:", state_reshaped.shape)
                states.append(state_reshaped)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append(reward)
                predictions.append(prediction)
                
                score += reward
                state = next_state

            self.lock.acquire()
            self.replay(states, actions, rewards, predictions)
            self.lock.release()

            with self.lock:
                logger.write("episode: {}/{}, thread: {}, score: {}".format(self.episode, self.EPISODES, thread, score))
                if(self.episode < self.EPISODES):
                    self.episode += 1

            if os.path.exists(foundbestpath):
                print("Found best on the main thread breaking loop")
                break

            if os.path.exists(foundmaxpath):
                print("No improvement max iterations passed, main thread breaking loop")
                break
