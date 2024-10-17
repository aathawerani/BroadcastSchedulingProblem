



# ================================================
# Modified from the work of Arthur Juliani:
#       Simple Reinforcement Learning with Tensorflow Part 8: Asynchronus Advantage Actor-Critic (A3C)
#       https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb
#
#       Implementation of Asynchronous Methods for Deep Reinforcement Learning
#       Algorithm details can be found here:
#           https://arxiv.org/pdf/1602.01783.pdf
#
# Modified to work with OpenAI Gym environments (currently working with cartpole)
# Author: Liam Pettigrew
# =================================================

import os
import threading
import multiprocessing
import numpy as np
import tensorflow as tf

from worker import Worker
from ac_network import AC_Network

# ===========================
#   Gym Utility Parameters
# ===========================
# Gym environment
ENV_NAME = 'CartPole-v0' # Discrete (4, 2)
STATE_DIM = 4
ACTION_DIM = 2
# Directory for storing gym results
MONITOR_DIR = './results/' + ENV_NAME

# ==========================
#   Training Parameters
# ==========================
RANDOM_SEED = 1234
# Load previously trained model
LOAD_MODEL = False
# Test and visualise a trained model
TEST_MODEL = False
# Directory for storing session model
MODEL_DIR = './model/'
# Learning rate
LEARNING_RATE = 0.0001
# Discount rate for advantage estimation and reward discounting
GAMMA = 0.99

def main(_):
    global master_network
    global global_episodes

    tf.reset_default_graph()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with tf.device("/cpu:0"):
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)

        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        master_network = AC_Network(STATE_DIM, ACTION_DIM, 'global', None)  # Generate global network
        num_workers = multiprocessing.cpu_count()  # Set workers to number of available CPU threads

        # For testing and visualisation we only need one worker
        if TEST_MODEL:
            num_workers = 1

        workers = []
        # Create worker classes
        for i in range(num_workers):
            workers.append(Worker(i, STATE_DIM, ACTION_DIM, trainer, MODEL_DIR, global_episodes,
                                  ENV_NAME, RANDOM_SEED, TEST_MODEL))
        saver = tf.train.Saver(max_to_keep=5)

        # Gym monitor
        if not TEST_MODEL:
            env = workers[0].get_env()
            env.monitor.start(MONITOR_DIR, video_callable=False, force=True)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        if LOAD_MODEL or TEST_MODEL:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        if TEST_MODEL:
            env = workers[0].get_env()
            env.monitor.start(MONITOR_DIR, force=True)
            workers[0].work(GAMMA, sess, coord, saver)
        else:
            # This is where the asynchronous magic happens.
            # Start the "work" process for each worker in a separate thread.
            worker_threads = []
            for worker in workers:
                worker_work = lambda: worker.work(GAMMA, sess, coord, saver)
                t = threading.Thread(target=(worker_work))
                t.start()
                worker_threads.append(t)
            coord.join(worker_threads)

if __name__ == '__main__':
    tf.app.run()





import tensorflow as tf
import scipy.signal
import numpy as np
import gym
from ac_network import AC_Network

# Size of mini batches to run training on
MINI_BATCH = 30
REWARD_FACTOR = 0.001

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Weighted random selection returns n_picks random indexes.
# the chance to pick the index i is give by the weight weights[i].
def weighted_pick(weights,n_picks):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return np.searchsorted(t,np.random.rand(n_picks)*s)

# Discounting function used to calculate discounted returns.
def discounting(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Normalization of inputs and outputs
def norm(x, upper, lower=0.):
    return (x-lower)/max((upper-lower), 1e-12)

class Worker():
    def __init__(self, name, s_size, a_size, trainer, model_path, global_episodes, env_name, seed, test):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))
        self.is_test = test
        self.a_size = a_size

        # Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)

        self.env = gym.make(env_name)
        self.env.seed(seed)

    def get_env(self):
        return self.env

    def train(self, rollout, sess, gamma, r):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        rewards_list = np.asarray(rewards.tolist()+[r])*REWARD_FACTOR
        discounted_rewards = discounting(rewards_list, gamma)[:-1]

        # Advantage estimation
        # JS, P Moritz, S Levine, M Jordan, P Abbeel,
        # "High-dimensional continuous control using generalized advantage estimation."
        # arXiv preprint arXiv:1506.02438 (2015).
        values_list = np.asarray(values.tolist()+[r])*REWARD_FACTOR
        advantages = rewards + gamma * values_list[1:] - values_list[:-1]
        discounted_advantages = discounting(advantages, gamma)


        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # sess.run(self.local_AC.reset_state_op)
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(states),
                     self.local_AC.actions: np.vstack(actions),
                     self.local_AC.advantages: discounted_advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_mini_buffer = []
                episode_values = []
                episode_states = []
                episode_reward = 0
                episode_step_count = 0

                # Restart environment
                terminal = False
                s = self.env.reset()

                rnn_state = self.local_AC.state_init

                # Run an episode
                while not terminal:
                    episode_states.append(s)
                    if self.is_test:
                        self.env.render()

                    # Get preferred action distribution
                    a_dist, v, rnn_state = sess.run([self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                                         feed_dict={self.local_AC.inputs: [s],
                                                    self.local_AC.state_in[0]: rnn_state[0],
                                                    self.local_AC.state_in[1]: rnn_state[1]})

                    a0 = weighted_pick(a_dist[0], 1) # Use stochastic distribution sampling
                    if self.is_test:
                        a0 = np.argmax(a_dist[0]) # Use maximum when testing
                    a = np.zeros(self.a_size)
                    a[a0] = 1

                    s2, r, terminal, info = self.env.step(np.argmax(a))

                    episode_reward += r

                    episode_buffer.append([s, a, r, s2, terminal, v[0, 0]])
                    episode_mini_buffer.append([s, a, r, s2, terminal, v[0, 0]])

                    episode_values.append(v[0, 0])

                    # Train on mini batches from episode
                    if len(episode_mini_buffer) == MINI_BATCH and not self.is_test:
                        v1 = sess.run([self.local_AC.value],
                                      feed_dict={self.local_AC.inputs: [s],
                                                    self.local_AC.state_in[0]: rnn_state[0],
                                                    self.local_AC.state_in[1]: rnn_state[1]})
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_mini_buffer, sess, gamma, v1[0][0])
                        episode_mini_buffer = []

                    # Set previous state for next step
                    s = s2
                    total_steps += 1
                    episode_step_count += 1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if episode_count % 10 == 0 and not episode_count % 100 == 0 and not self.is_test:
                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                if self.name == 'worker_0':
                    if episode_count % 100 == 0 and not self.is_test:
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')

                    print("| Reward: " + str(episode_reward), " | Episode", episode_count)
                    sess.run(self.increment) # Next global episode

                episode_count += 1



import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

# Clipping ratio for gradients
CLIP_NORM = 40.0
# Cell units
CELL_UNITS = 128

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.variable_scope(scope):
            # Input
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(CELL_UNITS, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = [c_in, h_in]
            rnn_in = tf.expand_dims(self.inputs, [0])
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in,
                initial_state=state_in,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, CELL_UNITS])

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None, a_size], dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions, [1])

                # Value loss function
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))

                # Softmax policy loss function
                self.policy_loss = -tf.reduce_sum(tf.log(tf.maximum(self.responsible_outputs, 1e-12)) * self.advantages)

                # Softmax entropy function
                self.entropy = - tf.reduce_sum(self.policy * tf.log(tf.maximum(self.policy, 1e-12)))

                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


