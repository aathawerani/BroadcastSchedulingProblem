import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from valuednn import ValueDNN

class Critic(object):
    def __init__(self, params, obs_dim, logger):
        self.device = params.device
        self.logger = logger
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = obs_dim
        self.epochs = params.critic_epochs
        self.value = ValueDNN(params, self.obs_dim, logger)
        self.hiddenlayerunits = params.actor_hidden_layer_units
        self.lr = 1e-2 / np.sqrt(self.hiddenlayerunits[0])  # 1e-3 empirically determined
        self.optimizer = tf.keras.optimizers.RMSprop(self.lr)

    def update_model(self, x_train, y_train, num_batches, batch_size):
        #print("here 1")

        batch_losses = []
        #print("here 2")
        for e in range(self.epochs):
            #print("here 3")
            #x_train, y_train = shuffle(x_train, y_train)
            #print("here 4")
            #for j in range(num_batches):
                #print("here 5")
                #start = j * batch_size
                #print("here 6")
                #end = (j + 1) * batch_size
                #print("here 7")

                #with tf.GradientTape() as tape:
                    #print("here 8")
                    #current_loss = tf.reduce_mean(tf.square(tf.cast(self.value(x_train[start:end, :]), dtype=tf.float64) - y_train[start:end]))
                    #print("here 9")
                    #grads = tape.gradient(current_loss, self.value.trainable_variables)
                #print("here 10")

                #self.optimizer.apply_gradients(zip(grads, self.value.trainable_variables))
                #print("here 11")

                #batch_losses.append(current_loss)


            with tf.GradientTape() as tape:
                #print("here 8")
                current_loss = tf.reduce_mean(
                    tf.square(tf.cast(self.value(x_train), dtype=tf.float64) - y_train))
                #print("here 9")
                grads = tape.gradient(current_loss, self.value.trainable_variables)
            #print("here 10")

            self.optimizer.apply_gradients(zip(grads, self.value.trainable_variables))
            #print("here 11")

            batch_losses.append(current_loss)

            #print("here 12")


        return current_loss

    def fit(self, x, y, logger):
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        y_hat = self.predict(x)  # check explained variance prior to update
        old_exp_var = 1 - np.var(y - y_hat)/np.var(y)
        x_train, y_train = x, y

        self.update_model(x_train, y_train, num_batches, batch_size)

        #batch_losses = []
        #for e in range(self.epochs):
            #x_train, y_train = shuffle(x_train, y_train)
            #for j in range(num_batches):
                #start = j * batch_size
                #end = (j + 1) * batch_size
                #current_loss = self.update_model(x_train[start:end, :], y_train[start:end])
                #batch_losses.append(current_loss)

        y_hat = self.predict(x)
        loss = np.mean(np.square(y_hat - y))         # explained variance after update
        exp_var = 1 - np.var(y - y_hat) / np.var(y)  # diagnose over-fitting of val func

    def predict(self, x):
        y_hat =  self.value.predict(x)
        return np.squeeze(y_hat)

