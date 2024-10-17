import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

class ValueDNN(Model):
    def __init__(self, params, obs_dim, logger, **kwargs):
        super(ValueDNN, self).__init__(**kwargs)
        self.logger = logger
        self.obs_dim = obs_dim
        self.numhiddenlayers = params.critic_hidden_layers
        self.hiddenlayerunits = params.critic_hidden_layer_units

        self.dropout = Dropout(0.5)
        self.Denselayers = []
        for i in range(self.numhiddenlayers):
            self.Denselayers.append(Dense(units=self.hiddenlayerunits[i], activation='relu',
                    kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.obs_dim))))

    def __call__(self, input, training=True):
        output = input
        for i in range(self.numhiddenlayers):
            output = self.Denselayers[i](output)
            #if training:
                #output = self.dropout(output)
        return tf.squeeze(output)

    def BuildNetwork(self, obs_ph, input):

        self.lr = 1e-2 / np.sqrt(self.hiddenlayerunits[0])  # 1e-3 empirically determined

        return self.lr, self.value, self.model, self.model1
