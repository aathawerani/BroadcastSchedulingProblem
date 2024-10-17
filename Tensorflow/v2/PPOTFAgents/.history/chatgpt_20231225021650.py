



import os
import time
from datetime import datetime
import config
from utils import Logger
from environmentwrapper import BSPEnvironmentWrapper

import numpy as np
import tensorflow as tf
import tf_agents as tfa
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_rnn_network
import random
from tf_agents.replay_buffers import tf_uniform_replay_buffer

def perturb_action(action, noise_scale=0.1):
    noise = np.random.normal(0, noise_scale, action.shape)
    perturbed_action = action + noise
    return perturbed_action

def train_eval(env_name):
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

    logger.printwrite("Start", env_name)

    logger.write(params)

    actor_fc_layers=params.actor_fc_layers
    value_fc_layers=params.value_fc_layers
    actor_lstm_layers = params.actor_lstm_size
    critic_lstm_layers = params.critic_lstm_size
    num_environment_steps = params.num_environment_steps
    num_parallel_environments = params.num_parallel_environments
    num_epochs = params.num_epochs
    learning_rate = params.learning_rate

    #tdma_py_env = BSPEnvironmentWrapper(env_name, logname)
    #tdma_tf_env = tf_py_environment.TFPyEnvironment(tdma_py_env)

    parallel_envs = [lambda: BSPEnvironmentWrapper(env_name, logname) for _ in range(num_parallel_environments)]
    parallel_py_env = parallel_py_environment.ParallelPyEnvironment(parallel_envs)
    tf_env = tf_py_environment.TFPyEnvironment(parallel_py_env)

    actor_lstm_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        input_fc_layer_params=(actor_fc_layers,),
        lstm_size=(actor_lstm_layers,))
        
    value_lstm_net = value_rnn_network.ValueRnnNetwork(
        tf_env.observation_spec(),
        input_fc_layer_params=(value_fc_layers,),
        lstm_size=(critic_lstm_layers,))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_step_counter = tf.Variable(0)
    ppo_agent = tfa.agents.PPOAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        optimizer,
        actor_net=actor_lstm_net,
        value_net=value_lstm_net,
        num_epochs=num_epochs,  # Number of epochs for training
        train_step_counter=train_step_counter,
        discount_factor=params.discount_factor,  # Discount factor for future rewards
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        use_gae=True)  # Set parallel environments if applicable

    ppo_agent.initialize()
    initial_epsilon = 1.0
    min_epsilon = 0.1
    decay_rate = 0.99
    batch_size = 32  # Batch size for training

    #replay_buffer = []  # Initialize replay buffer to store experiences

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=ppo_agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=10000)


    for _ in range(num_environment_steps):
        time_step = tf_env.reset()
        policy_state = ppo_agent.collect_policy.get_initial_state(tf_env.batch_size)
        #policy_state = ppo_agent.collect_policy.get_initial_state(batch_size)
        experience = []

        print("time_step.is_last()", time_step.is_last())
        print("time_step", time_step)
        print("policy_state", policy_state)
        while not time_step.is_last().numpy().all():
            action_step = ppo_agent.collect_policy.action(time_step, policy_state)
            #action_step = ppo_agent.collect_policy.action(time_step)
            action = action_step.action.numpy()[0]  # Convert action to numpy array for retry logic
            policy_state = action_step.state ##can be moved out of while loop to update once

            if random.uniform(0, 1) < initial_epsilon:
                action = perturb_action(action)

            next_time_step = tf_env.step(action)
            reward = next_time_step.reward

            if reward > 0:
                experience.append((time_step.observation, action_step.action, 
                        next_time_step.reward, next_time_step.observation, 
                        next_time_step.is_last()))

            time_step = next_time_step

        initial_epsilon = max(min_epsilon, initial_epsilon * decay_rate)

        for exp in experience:
            replay_buffer.add_batch(exp)

        sampled_batch = replay_buffer.get_next(sample_batch_size=batch_size)

        for _ in range(num_epochs):
            #batched_experience = replay_buffer.sample(batch_size) #AAHT replay buffer is empty

            with tf.GradientTape() as tape:
                #loss = ppo_agent.compute_loss(batched_experience)
                loss = ppo_agent.compute_loss(sampled_batch)
            gradients = tape.gradient(loss, ppo_agent.trainable_variables)
            
            optimizer.apply_gradients(zip(gradients, ppo_agent.trainable_variables))















import tensorflow as tf
import numpy as np
import random
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_rnn_network, value_rnn_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import network
from tf_agents.replay_buffers import tf_uniform_replay_buffer

# Define other necessary functions and classes used in the code (e.g., perturb_action)

# Define hyperparameters and configurations
num_epochs = 1000  # Number of training epochs
num_environment_steps = 10000  # Number of environment steps
num_parallel_environments = 8  # Number of parallel environments
batch_size = 32  # Batch size for training

# Define your environment setup using ParallelPyEnvironment
parallel_envs = [lambda: BSPEnvironmentWrapper(env_name, logname) for _ in range(num_parallel_environments)]
parallel_py_env = parallel_py_environment.ParallelPyEnvironment(parallel_envs)
tf_env = tf_py_environment.TFPyEnvironment(parallel_py_env)

# Define actor and value networks for the PPO agent
actor_lstm_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    input_fc_layer_params=(actor_fc_layers,),
    lstm_size=(actor_lstm_layers,))

value_lstm_net = value_rnn_network.ValueRnnNetwork(
    tf_env.observation_spec(),
    input_fc_layer_params=(value_fc_layers,),
    lstm_size=(critic_lstm_layers,))

# Define optimizer and other PPO agent parameters
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)

# Initialize PPO agent
ppo_agent = ppo_agent.PPOAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    optimizer,
    actor_net=actor_lstm_net,
    value_net=value_lstm_net,
    num_epochs=num_epochs,
    train_step_counter=train_step_counter,
    discount_factor=params.discount_factor,
    entropy_regularization=0.0,
    importance_ratio_clipping=0.2,
    use_gae=True)

ppo_agent.initialize()

# Initialize replay buffer to store experiences
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=ppo_agent.collect_data_spec,
    batch_size=num_parallel_environments * num_environment_steps,
    max_length=replay_buffer_capacity)

# Training loop
for _ in range(num_environment_steps):
    time_step = tf_env.reset()
    policy_state = ppo_agent.collect_policy.get_initial_state(tf_env.batch_size)
    experience = []

    while not time_step.is_last():
        retry_attempts = 0
        while retry_attempts < params.chances:
            action_step = ppo_agent.collect_policy.action(time_step, policy_state)
            action = action_step.action.numpy()[0]  # Convert action to numpy array for retry logic
            policy_state = action_step.state  # Can be moved out of the while loop to update once

            if random.uniform(0, 1) < initial_epsilon:
                action = perturb_action(action)

            if environment.checkActionLegal(action):  # Check if the predicted schedule is conflict-free
                break  # Break the retry loop if a conflict-free schedule is predicted

            retry_attempts += 1  # Increment retry attempts

        next_time_step = tf_env.step(action)
        reward = next_time_step.reward

        if reward > 0:
            experience.append((time_step.observation, action_step.action,
                               next_time_step.reward, next_time_step.observation,
                               next_time_step.is_last()))

    time_step = next_time_step

    initial_epsilon = max(min_epsilon, initial_epsilon * decay_rate)

    # Append experiences to replay buffer
    for exp in experience:
        replay_buffer.add_batch(exp)

    # Sample a batch from replay buffer
    sampled_batch = replay_buffer.get_next(sample_batch_size=batch_size)

    # Training step for PPO agent
    for _ in range(num_epochs):
        with tf.GradientTape() as tape:
            loss = ppo_agent.compute_loss(sampled_batch)
        gradients = tape.gradient(loss, ppo_agent.trainable_variables)

        optimizer.apply_gradients(zip(gradients, ppo_agent.trainable_variables))
















=======================================================================================

class BestSchedule:
    def __init__(self, lowerbound, state, logname, numnodes, env_name):
        self.LowerBound = lowerbound
        self.NumNodes = numnodes
        self.BestSchedule = state
        self.state = state
        self.FoundBest = False
        self.BestNodesAllocated = 0
        self.NodesAllocated = 0
        self.BestMinSlotAssigned = 0
        self.BestMaxSlotAssigned = 0
        self.uuid = uuid.uuid4()
        self.logname = logname
        self.logger = Logger(logname=logname)
        self.params = config.GetNetworkParams(env_name)
        self.MaxIterations = self.params.max_iterations
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
        if coverage < 3 and (coverage == 1 or self.CheckCompression()):
            self.UpdateBestSchedule2()
            self.CheckBestSchedule()

        if self.Iterations > self.MaxIterations:
            foundmaxlog = Logger(self.logname + "_" + "max")
            foundmaxlog.write(self.uuid, "Max iterations reached", self.BestSchedule)

    def CheckCoverage2(self):
        if self.NodesAllocated >= self.BestNodesAllocated:
            return 1 if self.NodesAllocated > self.BestNodesAllocated else 2
        return 3

    def CheckCompression(self):
        return self.MaxSlotAssigned < self.BestMaxSlotAssigned

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
        self.BestMinSlotAssigned = 0 if self.NodesAllocated < self.NumNodes else 1



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






# Assuming you have a PPO agent, optimizer, and collect policy already defined

while not time_step.is_last():
    # ... (previous loop code remains unchanged)

    # If the actor fails to predict a conflict-free schedule after retries
    if retry_attempts == max_retry_attempts:
        # PPO update logic using collected experience
        experience = # Collected experience from the episode
        for _ in range(num_iterations):  # Perform multiple update iterations
            with tf.GradientTape() as tape:
                # Compute loss using the collected experience
                loss = ppo_agent.compute_loss(experience)

            # Compute gradients
            gradients = tape.gradient(loss, ppo_agent.trainable_variables)
            
            # Apply gradients using optimizer
            optimizer.apply_gradients(zip(gradients, ppo_agent.trainable_variables))

    if time_step.is_last():
        break  # Break the main training loop if the episode is complete




from tf_agents.trajectories import trajectory

# Assuming you have a TDMA environment tdma_tf_env and a PPO agent ppo_agent already defined

while not time_step.is_last():
    retry_attempts = 0
    episode_reward = 0
    experience_buffer = []  # List to store experiences within the episode

    while retry_attempts < max_retry_attempts:
        action_step = ppo_agent.collect_policy.action(time_step)
        action = action_step.action.numpy()[0]  # Convert action to numpy array for retry logic
        next_time_step = tdma_tf_env.step(action)
        episode_reward += next_time_step.reward

        # Create a trajectory containing the transition
        traj = trajectory.from_transition(
            time_step, action_step, next_time_step)

        experience_buffer.append(traj)  # Store the transition in the experience buffer

        if predicted_schedule_is_conflict_free(next_time_step):
            break  # Break the retry loop if a conflict-free schedule is predicted

        retry_attempts += 1  # Increment retry attempts

        time_step = next_time_step

    # Convert experience buffer to a format suitable for PPO update
    experience = trajectory.to_transition(experience_buffer)

    # PPO update logic using the collected experience (as shown in the previous PPO update code snippet)

    if time_step.is_last():
        break  # Break the main training loop if the episode is complete






import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.networks import network, sequential
from tf_agents.agents.ppo import ppo_agent
from tf_agents.utils import common

# Assuming you have a TDMA environment tdma_env defined earlier

# Convert the Python environment to a TensorFlow environment
tf_tdma_env = tf_py_environment.TFPyEnvironment(tdma_env)

# Define the neural network architecture for actor and critic
input_tensor_spec = tf_tdma_env.observation_spec()
output_tensor_spec = tf_tdma_env.action_spec()

# Example neural network architecture (modify as needed)
dense_layers = (128, 64)  # Example dense layers for the network
actor_net = sequential.Sequential([networks.fc_layer.Dense(
    num_units, activation_fn=tf.nn.relu) for num_units in dense_layers] +
    [networks.fc_layer.Dense(output_tensor_spec.maximum + 1)])

critic_net = sequential.Sequential([networks.fc_layer.Dense(
    num_units, activation_fn=tf.nn.relu) for num_units in dense_layers] +
    [networks.fc_layer.Dense(1)])

# Define the PPO agent
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
train_step_counter = tf.Variable(0)
actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

ppo_tf_agent = ppo_agent.PPOAgent(
    time_step_spec=tf_tdma_env.time_step_spec(),
    action_spec=tf_tdma_env.action_spec(),
    actor_net=actor_net,
    critic_net=critic_net,
    optimizer=optimizer,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    train_step_counter=train_step_counter,
    use_gae=True,  # Example parameter, can be modified
    use_td_lambda_return=True,  # Example parameter, can be modified
    normalize_rewards=True)  # Example parameter, can be modified

# Initialize the agent
ppo_tf_agent.initialize()

# Initialize other components as needed (e.g., replay buffer, reward calculations)




class BestSchedule:
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
        if coverage == 1 or (coverage == 2 and self.CheckCompression()):
            self.UpdateBestSchedule2()
            self.CheckBestSchedule()

        if self.Iterations > self.MaxIterations:
            foundmaxlog = Logger(self.logname + "_" + "max")
            foundmaxlog.write(self.uuid, "Max iterations reached", self.BestSchedule)

    def CheckCoverage2(self):
        if self.NodesAllocated >= self.BestNodesAllocated:
            return 1 if self.NodesAllocated > self.BestNodesAllocated else 2
        return 3

    def CheckCompression(self):
        return self.MaxSlotAssigned < self.BestMaxSlotAssigned

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
        self.BestMinSlotAssigned = 0 if self.NodesAllocated < self.NumNodes else 1







def checkActionLegal(self, node):
    if node < 0 or node >= self.NumNodes or self.state[node] > 0:
        return False

    for node1 in self.CurrentSlotNodes:
        if self.adjacent[node][node1] > 0:
            return False

    self.CurrentSlotNodes.append(node)
    self.CurrentSlotNodesCount += 1
    return True





class BSPEnvironment:
    def __init__(self, env_name, logname):
        self.env_name = env_name
        self.params = config.GetNetworkParams(env_name)
        self.adjacent = EnvGen().GetNetwork(env_name, self.params.cases)
        self.NumNodes = len(self.adjacent)
        self.slotNumber = 1
        self.chances = self.params.chances
        self.remainingchances = self.params.chances
        self.reward_multiplier = self.params.reward_multiplier
        self.LowerBound = max(len(neighbors) for neighbors in self.adjacent.values()) + 1
        self.NodesAllocated = 0
        self.CurrentSlotNodes = []
        self.CurrentSlotNodesCount = 0
        self.reward = 0
        self.totalchances = self.chances
        self.state = np.zeros(self.NumNodes, dtype=np.float32)
        self.schedule = BestSchedule(self.LowerBound, self.state, logname, self.NumNodes, env_name)

    def reset(self):
        self.slotNumber = 1
        self.remainingchances = self.chances
        self.state.fill(0)
        self.NodesAllocated = 0
        self.CurrentSlotNodes.clear()
        self.CurrentSlotNodesCount = 0
        self.reward = 0
        self.totalchances = self.chances

    def isDone(self):
        if self.remainingchances == 0:
            if self.CurrentSlotNodesCount == 0:
                self.slotNumber -= 1
                return True
            self.slotNumber += 1
            self.remainingchances = self.chances
            self.CurrentSlotNodes.clear()
            self.CurrentSlotNodesCount = 0
            self.reward = 0
        return self.NodesAllocated >= self.NumNodes

    def BSPstep(self, action):
        node = action - 1
        legal = self.checkActionLegal(node)
        reward = -1 if not legal else self.adjacent[node]
        if legal:
            self.state[node] = self.slotNumber
            self.NodesAllocated += 1
        else:
            self.remainingchances -= 1
        return self.state, reward

    def checkActionLegal(self, node):
        return node in range(self.NumNodes) and self.state[node] == 0 and all(self.adjacent[node][n] == 0 for n in self.CurrentSlotNodes)







# Assume tf_agents and TensorFlow setup and initialization

# Define exploration parameters
initial_epsilon = 1.0
min_epsilon = 0.1
decay_rate = 0.99

# Training loop
for episode in range(num_episodes):
    time_step = tf_tdma_env.reset()
    episode_return = 0.0

    while not time_step.is_last():
        # Get action from the PPO agent's policy
        action_step = ppo_agent.collect_policy.action(time_step)
        action = action_step.action.numpy()

        # Exploration: Add randomness based on epsilon-greedy strategy
        if random.uniform(0, 1) < initial_epsilon:
            # Explore: Perturb the selected action
            action = perturb_action(action)

        # Perform action in the environment
        next_time_step = tf_tdma_env.step(action)

        # Collect experience and add to replay buffer (if using)
        # ...

        # Train the agent on collected experiences (if using)
        # ...

        # Decay epsilon for exploration-exploitation balance
        initial_epsilon = max(min_epsilon, initial_epsilon * decay_rate)

        time_step = next_time_step
        episode_return += time_step.reward

    # Logging and evaluation at the end of each episode
    # ...






# Assuming setup, environment creation, and agent initialization

# Training loop
for episode in range(num_episodes):
    time_step = tf_tdma_env.reset()
    episode_return = 0.0
    experience = []
    
    while not time_step.is_last():
        # Collect experience using the PPO agent's policy
        action_step = ppo_agent.collect_policy.action(time_step)
        next_time_step = tf_tdma_env.step(action_step.action)

        # Store experience: (state, action, reward, next_state, done)
        experience.append((time_step.observation, action_step.action, 
                           next_time_step.reward, next_time_step.observation, 
                           next_time_step.is_last()))

        time_step = next_time_step
        episode_return += time_step.reward

    # Train the agent using collected experiences (PPO update)
    time_steps = tf_tdma_env.reset()
    for _ in range(num_epochs):
        # Generate training data from collected experiences
        batched_experience = [experience]  # Convert collected experience to batched data

        with tf.GradientTape() as tape:
            # Compute loss based on the collected experiences
            loss = compute_loss(ppo_agent, batched_experience)

        # Calculate gradients of the loss with respect to the network's trainable variables
        gradients = tape.gradient(loss, ppo_agent.trainable_variables)

        # Apply gradients to update the network weights
        optimizer.apply_gradients(zip(gradients, ppo_agent.trainable_variables))

    # Logging and evaluation at the end of each episode
    # ...






# ... (Previous code)

# Experience Replay Buffer Initialization
replay_buffer = []  # Initialize replay buffer to store experiences

for _ in range(num_environment_steps):
    # ... (Other code)
    
    # Collect experiences from the environment and store in the replay buffer
    # ... (Code to collect experiences and append to replay_buffer)

    # Training loop
    for _ in range(num_epochs):
        # Sample batch from replay_buffer for training
        batched_experience = sample_batch_from_replay_buffer(replay_buffer)

        # Update the PPO agent using the collected experiences
        ppo_agent.train(batched_experience)

    # ... (Other code)






import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_rnn_network, value_rnn_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.utils import common

# Initialize your TDMA environment and wrap it with TF-Agents environment
# Replace BSPEnvironmentWrapper with your TDMA environment wrapper
tdma_py_env = BSPEnvironmentWrapper(env_name, logname)
tdma_tf_env = tf_py_environment.TFPyEnvironment(tdma_py_env)

# Define LSTM-based Actor and Critic networks
actor_lstm_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
    tdma_tf_env.observation_spec(),
    tdma_tf_env.action_spec(),
    input_fc_layer_params=(actor_fc_layers,),
    lstm_size=(actor_lstm_layers,))
    
value_lstm_net = value_rnn_network.ValueRnnNetwork(
    tdma_tf_env.observation_spec(),
    input_fc_layer_params=(value_fc_layers,),
    lstm_size=(critic_lstm_layers,))

# Set hyperparameters
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
num_epochs = params.num_epochs
discount_factor = params.discount_factor

# Instantiate PPO agent with LSTM networks
ppo_agent = ppo_agent.PPOAgent(
    tdma_tf_env.time_step_spec(),
    tdma_tf_env.action_spec(),
    optimizer,
    actor_net=actor_lstm_net,
    value_net=value_lstm_net,
    num_epochs=num_epochs,
    train_step_counter=train_step_counter,
    discount_factor=discount_factor,
    entropy_regularization=0.0,
    importance_ratio_clipping=0.2,
    use_gae=True)

# Initialize agent
ppo_agent.initialize()

# Main training loop
for _ in range(num_environment_steps):
    time_step = tdma_tf_env.reset()
    policy_state = ppo_agent.collect_policy.get_initial_state(tdma_tf_env.batch_size)
    experience = []
    
    while not time_step.is_last():
        action_step = ppo_agent.collect_policy.action(time_step, policy_state)
        action = action_step.action.numpy()[0]
        next_time_step = tdma_tf_env.step(action)
        experience.append((time_step.observation, action_step.action, 
                           next_time_step.reward, next_time_step.observation, 
                           next_time_step.is_last()))
        time_step = next_time_step
    
    for _ in range(num_epochs):
        batched_experience = common.function(sample_batch_from_replay_buffer)(
            experience, batch_size=batch_size)
        
        with tf.GradientTape() as tape:
            loss = ppo_agent.compute_loss(batched_experience)
        gradients = tape.gradient(loss, ppo_agent.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ppo_agent.trainable_variables))




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
