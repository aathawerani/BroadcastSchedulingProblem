import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers


class ActorCriticModel():
  def __init__(self, params, logger):
    self.logger = logger
    self.obs_dim = params.obs_dim
    self.act_dim = params.act_dim
    self.numhiddenlayers = params.critic_hidden_layers
    self.hiddenlayerunits = params.critic_hidden_layer_units

  def createModel(self):

    inputs = layers.Input(shape=(1, self.obs_dim))
    common = inputs

    for i in range(self.numhiddenlayers):
      if i < self.numhiddenlayers - 1:
        common = layers.LSTM(units=self.hiddenlayerunits[i], return_sequences=True, activation='relu',
                                    recurrent_activation='relu', unroll=True,
                                    kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.hiddenlayerunits[i])))(common)
      else:
        common = layers.LSTM(units=self.act_dim, return_sequences=False, activation='relu',
                                    recurrent_activation='relu', unroll=True,
                                    kernel_initializer=tf.random_normal_initializer(
                                      stddev=np.sqrt(1 / self.act_dim)))(common)

    action = layers.Dense(self.act_dim, activation="softmax")(common)
    critic = layers.Dense(1)(common)

    model = keras.Model(inputs=inputs, outputs=[action, critic])
    return model

