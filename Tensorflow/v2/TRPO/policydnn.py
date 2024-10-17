import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.models import Model

class PolicyDNN(Model):
    def __init__(self, params, obs_dim, act_dim, logger, **kwargs):
        super(PolicyDNN, self).__init__(**kwargs)
        self.logger = logger
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.numhiddenlayers = params.actor_hidden_layers
        self.hiddenlayerunits = params.actor_hidden_layer_units

        self.dropout = Dropout(0.5)

        #print("here 1")
        self.LSTMlayers = []
        #print("here 2")
        for i in range(self.numhiddenlayers):
            #print("self.LSTMlayers", self.LSTMlayers)
            if i < self.numhiddenlayers - 1:
                self.LSTMlayers.append(LSTM(units=self.hiddenlayerunits[i], return_sequences=True, activation = 'relu',
                    recurrent_activation='relu', unroll=True,
                    kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.hiddenlayerunits[i]))))
            else :
                self.LSTMlayers.append(LSTM(units=self.act_dim, return_sequences=False, activation = 'relu',
                    recurrent_activation='relu', unroll=True,
                    kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.act_dim))))

    def __call__(self, input, training=True):
        output = input
        #print("self.LSTMlayers", self.LSTMlayers)
        for i in range(self.numhiddenlayers):
            output = self.LSTMlayers[i](output)
            #if training:
                #output = self.dropout(output)
        return output

