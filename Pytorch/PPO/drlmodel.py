import tensorflow as tf
import numpy as np

from tf_agents.networks import actor_distribution_network, value_network, actor_distribution_rnn_network, value_rnn_network
from tf_agents.keras_layers import dynamic_unroll_layer

KERAS_LSTM_FUSED = 2

def get_actor_net(observation_spec, action_spec, actor_lstm_layers):
    #actor_net = actor_distribution_network.ActorDistributionNetwork(
        #tf_env.observation_spec(),
        #tf_env.action_spec(),
        #fc_layer_params=actor_fc_layers,
        #activation_fn=tf.keras.activations.tanh)

    rnn_construction_kwargs = {"lstm_size":actor_lstm_layers, "dtype":np.float32,
                               "dropout": 0.2, "recurrent_dropout": 0.2}

    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        input_fc_layer_params=None,
        output_fc_layer_params=None,
        #lstm_size=actor_lstm_layers
        rnn_construction_fn=rnn_construction_fn,
        rnn_construction_kwargs=rnn_construction_kwargs)

    return actor_net


def get_value_net(observation_spec, actor_lstm_layers):

    rnn_construction_kwargs = {"lstm_size": actor_lstm_layers, "dtype": np.float32,
                               "dropout": 0.2, "recurrent_dropout": 0.2}

    value_net = value_rnn_network.ValueRnnNetwork(
        observation_spec,
        input_fc_layer_params=None,
        output_fc_layer_params=None,
        #lstm_size=critic_lstm_layers
        rnn_construction_fn=rnn_construction_fn,
        rnn_construction_kwargs=rnn_construction_kwargs)

    #value_net = value_network.ValueNetwork(
        #tf_env.observation_spec(),
        #fc_layer_params=value_fc_layers)

    return value_net


def rnn_construction_fn(**rnn_construction_kwargs):

    lstm_size  = rnn_construction_kwargs.get("lstm_size")
    dtype = rnn_construction_kwargs.get("dtype")
    dropout = rnn_construction_kwargs.get("dropout")
    recurrent_dropout = rnn_construction_kwargs.get("recurrent_dropout")

    if len(lstm_size) == 1:
        cell = tf.keras.layers.LSTMCell(
        lstm_size[0],
        dtype=dtype,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=KERAS_LSTM_FUSED)
    else:
        cell = tf.keras.layers.StackedRNNCells(
        [tf.keras.layers.LSTMCell(size, dtype=dtype,
                                  dropout=dropout,
                                  recurrent_dropout=recurrent_dropout,
                                    implementation=KERAS_LSTM_FUSED)
            for size in lstm_size])
    lstm_network = dynamic_unroll_layer.DynamicUnroll(cell)

    return lstm_network