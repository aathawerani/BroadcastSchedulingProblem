import os
import time
from datetime import datetime
import numpy as np

import tensorflow as tf

from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.networks import actor_distribution_network, value_network, actor_distribution_rnn_network, \
    value_rnn_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

import config
from utils import Logger
import drlmodel


def train_eval(env_name):
    # env_name = "network_1"
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

    # actor_fc_layers=params.actor_fc_layers
    # value_fc_layers=params.value_fc_layers
    actor_lstm_layers = params.actor_lstm_size
    critic_lstm_layers = params.critic_lstm_size
    num_environment_steps = params.num_environment_steps
    collect_episodes_per_iteration = params.collect_episodes_per_iteration
    num_parallel_environments = params.num_parallel_environments
    replay_buffer_capacity = params.replay_buffer_capacity
    num_epochs = params.num_epochs
    learning_rate = params.learning_rate

    log_interval = 50
    debug_summaries = False
    summarize_grads_and_vars = False

    global_step = tf.compat.v1.train.get_or_create_global_step()

    tf_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            [lambda: config.load(env_name, logname)] * num_parallel_environments))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    # actor_net = actor_distribution_network.ActorDistributionNetwork(
    # tf_env.observation_spec(),
    # tf_env.action_spec(),
    # fc_layer_params=actor_fc_layers,
    # activation_fn=tf.keras.activations.tanh)

    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        input_fc_layer_params=None,
        output_fc_layer_params=None,
        lstm_size=actor_lstm_layers)

    # actor_net = drlmodel.get_actor_net(
    # tf_env.observation_spec(),
    # tf_env.action_spec(),
    # actor_lstm_layers)

    value_net = value_rnn_network.ValueRnnNetwork(
        tf_env.observation_spec(),
        input_fc_layer_params=None,
        output_fc_layer_params=None,
        lstm_size=critic_lstm_layers)

    # value_net = value_network.ValueNetwork(
    # tf_env.observation_spec(),
    # fc_layer_params=value_fc_layers)

    # value_net = drlmodel.get_value_net(
    # tf_env.observation_spec(),
    # critic_lstm_layers)

    tf_agent = ppo_clip_agent.PPOClipAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        optimizer,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=num_epochs,

        discount_factor=params.discount_factor,
        use_td_lambda_return=True,
        lambda_value=params.lambda_value,
        compute_value_and_advantage_in_train=True,

        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step)
    tf_agent.initialize()

    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]

    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(
            batch_size=num_parallel_environments),
        tf_metrics.AverageEpisodeLengthMetric(
            batch_size=num_parallel_environments),
    ]

    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_buffer_capacity)

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=collect_episodes_per_iteration)

    def train_step():
        trajectories = replay_buffer.gather_all()
        return tf_agent.train(experience=trajectories)

    collect_driver.run = common.function(collect_driver.run, autograph=False)
    tf_agent.train = common.function(tf_agent.train, autograph=False)
    train_step = common.function(train_step)

    collect_time = 0
    train_time = 0
    timed_at_step = global_step.numpy()

    while environment_steps_metric.result() < num_environment_steps:
        global_step_val = global_step.numpy()
        start_time = time.time()
        collect_driver.run()
        collect_time += time.time() - start_time

        start_time = time.time()
        total_loss, _ = train_step()
        replay_buffer.clear()
        train_time += time.time() - start_time

        # for train_metric in train_metrics:
        # train_metric.tf_summaries(
        # train_step=global_step, step_metrics=step_metrics)

        if global_step_val % log_interval == 0:
            steps_per_sec = ((global_step_val - timed_at_step) / (collect_time + train_time))
            logger.write('step = {}, loss = {:f} steps/sec = {:f} collect_time = {:f} train_time = {:f}'
                         .format(global_step_val, total_loss, steps_per_sec, collect_time, train_time))
            # with tf.compat.v2.summary.record_if(True):
            # tf.compat.v2.summary.scalar(
            # name='global_steps_per_sec', data=steps_per_sec, step=global_step)

            timed_at_step = global_step_val
            collect_time = 0
            train_time = 0

        if os.path.exists(foundbestpath):
            print("Found best on the main thread breaking loop")
            break

        if os.path.exists(foundmaxpath):
            print("No improvement max iterations passed, main thread breaking loop")
            break

    logger.printwrite("End", env_name)
