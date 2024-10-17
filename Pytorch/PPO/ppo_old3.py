import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import collections
import statistics
import tqdm
import threading
from threading import Thread, Lock
from keras import layers
from typing import Any, List, Sequence, Tuple

import config
from utils import Logger
from environmentwrapper import BSPEnvironmentWrapper


class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self,
      num_actions: int,
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.common = layers.Dense(num_hidden_units, activation="relu")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.common(inputs)
    return self.actor(x), self.critic(x)

# Wrap Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

@tf.numpy_function(Tout=[tf.float32, tf.int32, tf.int32])
def env_step(action: np.ndarray, thread) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""

  state, reward, done, info = envs[thread].step(action)
  return (state.astype(np.float32),
          np.array(reward, np.int32),
          np.array(done, np.int32))

def run_episode(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int,
    thread) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = env_step(action, thread)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards

def get_expected_return(
    rewards: tf.Tensor,
    gamma: float,
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) /
               (tf.math.reduce_std(returns) + eps))

  return returns


def compute_loss(
    action_probs: tf.Tensor,
    values: tf.Tensor,
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined Actor-Critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss

@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int,
    thread) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode, thread)

    # Calculate the expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

    # Calculate the loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  with lock:
    # Compute the gradients from the loss
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the model's parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward

def train_threading(thread, th):
    print("thread", thread)
    for i in t:
        initial_state, info = envs[thread].reset()
        initial_state = tf.constant(initial_state, dtype=tf.float32)
        episode_reward = int(train_step(
            initial_state, model, optimizer, gamma, max_steps_per_episode, thread))

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        with lock:
            t.set_postfix(
                episode_reward=episode_reward, running_reward=running_reward)

        # Show the average episode reward every 10 episodes
        if i % 10 == 0:
            pass # print(f'Episode {i}: average reward: {avg_reward}')

        #if running_reward > reward_threshold and i >= min_episodes_criterion:
            #break

        if os.path.exists(foundbestpath):
            print("Found best on the main thread breaking loop")
            break

        if os.path.exists(foundmaxpath):
            print("No improvement max iterations passed, main thread breaking loop")
            break
   


env_name = "network_5"
thread = 0
params = config.GetNetworkParams(env_name)
logpath = params.logpath
date1 = datetime.utcnow().strftime("%y%m%d")  # create unique directories
time1 = datetime.utcnow().strftime("%H%M%S")  # create unique directories
logname = env_name + "_" + date1 + time1
bestname = logname + "_" + "best"
maxname = logname + "_" + "max"
logger = Logger(logname)
foundbestpath = os.path.join(logpath, bestname + ".log")
foundmaxpath = os.path.join(logpath, maxname + ".log")
logger.printwrite(thread, "Start", env_name)
logger.write(params)

lock = Lock() # lock all to update parameters without other thread interruption
# Create the environment
#env = gym.make("CartPole-v1")
#env = BSPEnvironmentWrapper(env_name, logname,thread)

num_envs = 2
envs = [BSPEnvironmentWrapper(env_name, logname,thread) for i in range(num_envs)]

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()
  
num_actions = envs[0].action_space.n  # 2
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)
huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

min_episodes_criterion = 100
max_episodes = 1000
max_steps_per_episode = 500000000

# `CartPole-v1` is considered solved if average reward is >= 475 over 500
# consecutive trials
reward_threshold = 475
running_reward = 0

# The discount factor for future rewards
gamma = 0.99

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

t = tqdm.trange(max_episodes)

threads = [threading.Thread(
                target=train_threading,
                args=(i, i)) for i in range(num_envs)]

for th in threads:
    time.sleep(2)
    th.start()

for th in threads:
    time.sleep(10)
    th.join()

print("Complete")
