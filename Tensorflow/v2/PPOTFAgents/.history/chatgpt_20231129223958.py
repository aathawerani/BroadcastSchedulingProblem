



import numpy as np
import tensorflow as tf
import tf_agents as tfa
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# Define your TDMAEnvironment using tf-agents
class TDMAEnvironment(py_environment.PyEnvironment):
    def __init__(self, num_nodes):
        # Initialize environment variables, observation space, action space, etc.
        self._num_nodes = num_nodes
        self._num_slots = 10  # Example: 10 slots
        self._current_slot = 0
        self._tdma_schedule = np.zeros(num_nodes, dtype=np.int32)  # TDMA schedule representation

        # Define observation and action specs
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(num_nodes + 1,), dtype=np.int32, minimum=0, maximum=num_nodes, name='observation')
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=num_nodes - 1, name='action')

        # Define initial state and action
        self._state = np.zeros(num_nodes + 1, dtype=np.int32)
        self._state[-1] = self._current_slot

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        # Reset environment state to initial state and return initial observation
        self._current_slot = 0
        self._tdma_schedule = np.zeros(self._num_nodes, dtype=np.int32)
        self._state = np.zeros(self._num_nodes + 1, dtype=np.int32)
        self._state[-1] = self._current_slot
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        # Take action in the environment and return a new observation, reward, and whether the episode is done
        self._tdma_schedule[action] = 1  # Update TDMA schedule for the current slot
        self._current_slot += 1
        self._state = np.concatenate((self._tdma_schedule, [self._current_slot]), axis=None)
        if self._current_slot < self._num_slots:
            return ts.transition(np.array(self._state, dtype=np.int32), reward=0.0, discount=1.0)
        else:
            return ts.termination(np.array(self._state, dtype=np.int32), reward=0.0)

# Convert the PyEnvironment to TFEnvironment
num_nodes = 10  # Define the number of nodes in your TDMA system
tdma_py_env = TDMAEnvironment(num_nodes)
tdma_tf_env = tf_py_environment.TFPyEnvironment(tdma_py_env)

# Define Actor and Value networks for the PPO agent
actor_net = tfa.networks.actor_distribution_network.ActorDistributionNetwork(
    tdma_tf_env.observation_spec(),
    tdma_tf_env.action_spec(),
    fc_layer_params=(200, 100))  # Customize the architecture as needed

value_net = tfa.networks.value_network.ValueNetwork(
    tdma_tf_env.observation_spec(),
    fc_layer_params=(200, 100))  # Customize the architecture as needed

# Instantiate PPO agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)
ppo_agent = tfa.agents.PPOAgent(
    tdma_tf_env.time_step_spec(),
    tdma_tf_env.action_spec(),
    optimizer,
    actor_net=actor_net,
    value_net=value_net,
    num_epochs=10,  # Number of epochs for training
    train_step_counter=train_step_counter,
    discount_factor=0.99,  # Discount factor for future rewards
    entropy_regularization=0.0,
    importance_ratio_clipping=0.2,
    use_gae=True,
    num_parallel_environments=1)  # Set parallel environments if applicable

# Initialize agent
ppo_agent.initialize()

# Main training loop
num_iterations = 10000  # Set the number of training iterations
for _ in range(num_iterations):
    time_step = tdma_tf_env.reset()
    policy_state = ppo_agent.collect_policy.get_initial_state(tdma_tf_env.batch_size)
    episode_reward = 0
    while not time_step.is_last():
        action_step = ppo_agent.collect_policy.action(time_step, policy_state)
        time_step = tdma_tf_env.step(action_step.action)
        episode_reward += time_step.reward
        policy_state = action_step.state


import numpy as np
import tensorflow as tf
import tf_agents as tfa
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_rnn_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.utils import common

# Define your TDMAEnvironment using tf-agents
class TDMAEnvironment(py_environment.PyEnvironment):
    # ... Define your environment as previously mentioned ...

# Convert the PyEnvironment to TFEnvironment
num_nodes = 10  # Define the number of nodes in your TDMA system
tdma_py_env = TDMAEnvironment(num_nodes)
tdma_tf_env = tf_py_environment.TFPyEnvironment(tdma_py_env)

# Define LSTM-based Actor and Critic networks
actor_lstm_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
    tdma_tf_env.observation_spec(),
    tdma_tf_env.action_spec(),
    input_fc_layer_params=(200,),
    lstm_size=(40,))
    
value_lstm_net = value_rnn_network.ValueRnnNetwork(
    tdma_tf_env.observation_spec(),
    input_fc_layer_params=(200,),
    lstm_size=(40,))

# Instantiate PPO agent with LSTM networks
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)
ppo_agent = tfa.agents.PPOAgent(
    tdma_tf_env.time_step_spec(),
    tdma_tf_env.action_spec(),
    optimizer,
    actor_net=actor_lstm_net,
    value_net=value_lstm_net,
    num_epochs=10,  # Number of epochs for training
    train_step_counter=train_step_counter,
    discount_factor=0.99,  # Discount factor for future rewards
    entropy_regularization=0.0,
    importance_ratio_clipping=0.2,
    use_gae=True,
    num_parallel_environments=1)  # Set parallel environments if applicable

# Initialize agent
ppo_agent.initialize()

# Main training loop
num_iterations = 10000  # Set the number of training iterations
for _ in range(num_iterations):
    time_step = tdma_tf_env.reset()
    policy_state = ppo_agent.collect_policy.get_initial_state(tdma_tf_env.batch_size)
    episode_reward = 0
    while not time_step.is_last():
        action_step = ppo_agent.collect_policy.action(time_step, policy_state)
        time_step = tdma_tf_env.step(action_step.action)
        episode_reward += time_step.reward
        policy_state = action_step.state


# ... (previous code)

# Main training loop
num_iterations = 10000  # Set the number of training iterations
for _ in range(num_iterations):
    time_step = tdma_tf_env.reset()
    policy_state = ppo_agent.collect_policy.get_initial_state(tdma_tf_env.batch_size)
    episode_reward = 0

    while not time_step.is_last():
        retry_attempts = 0
        while retry_attempts < max_retry_attempts:
            action_step = ppo_agent.collect_policy.action(time_step, policy_state)
            action = action_step.action.numpy()[0]  # Convert action to numpy array for retry logic
            time_step = tdma_tf_env.step(action)
            episode_reward += time_step.reward
            policy_state = action_step.state

            if predicted_schedule_is_conflict_free(time_step):  # Check if the predicted schedule is conflict-free
                break  # Break the retry loop if a conflict-free schedule is predicted

            retry_attempts += 1  # Increment retry attempts

        # If the actor fails to predict a conflict-free schedule after retries, update policy or take other action
        if retry_attempts == max_retry_attempts:
            # Take action when the actor is unable to predict a conflict-free schedule after retries
            # Example: Update policy, handle the situation, etc.

        if time_step.is_last():
            break  # Break the main training loop if the episode is complete
