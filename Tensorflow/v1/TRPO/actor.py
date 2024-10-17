import numpy as np
import tensorflow as tf
from policydnn import PolicyDNN

class Actor(object):
    def __init__(self, params, obs_dim, act_dim, logger):
        self.device = params.device
        self.logger = logger
        self.beta = params.actor_beta  # dynamically adjusted D_KL loss multiplier
        self.eta = params.actor_eta  # multiplier for D_KL-kl_targ hinge-squared loss
        self.kl_targ = params.actor_kl_targ
        self.epochs = params.actor_epochs
        self.lr = None
        self.lr_multiplier = params.actor_lr_multiplier  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clipping_range = params.actor_clipping_range
        self.hiddenlayerunits = params.actor_hidden_layer_units
        self.policy_logvar = params.actor_policy_logvar
        self.mu = PolicyDNN(params, obs_dim, act_dim, logger)
        self.lr = 9e-4 / np.sqrt(self.hiddenlayerunits[0])  # 9e-4 empirically determined
        logvar_speed = (10 * self.hiddenlayerunits[0]) // 48
        log_vars = tf.compat.v1.get_variable('logvars', (logvar_speed, self.act_dim), tf.float32,
                                   tf.constant_initializer(0.0))
        self.log_vars = tf.reduce_sum(log_vars, axis=0) + self.policy_logvar
        self.optimizer = tf.keras.optimizers.RMSprop(self.lr)

    def update_model(self, observes, actions, advantages):

        old_log_vars = self.log_vars
        old_means = self.mu(observes)
        #kl = 0.0

        batch_losses = []
        for e in range(self.epochs):
            #print("here 25")
            #kl = self.kl(observes, old_log_vars, old_means)
            #print("here 26")
            #print("kl", kl)
            #print("test kl", tf.math.greater(kl,  self.kl_targ * 4))
            #if tf.math.greater(kl,  self.kl_targ * 4):  # early stopping if D_KL diverges badly
                #print("here 27")
                #break
            #print("here 1")

            with tf.GradientTape() as tape:
                # print("here 7")
                logp = -0.5 * tf.reduce_sum(self.log_vars)
                # print("here 8")
                logp += -0.5 * tf.reduce_sum(tf.square(actions - self.mu(observes)) /
                                             tf.exp(self.log_vars), axis=1)
                # print("here 10")
                logp_old = -0.5 * tf.reduce_sum(old_log_vars)
                # print("here 11")
                logp_old += -0.5 * tf.reduce_sum(tf.square(actions - old_means) /
                                                 tf.exp(old_log_vars), axis=1)
                # print("here 5")

                log_det_cov_old = tf.reduce_sum(old_log_vars)
                log_det_cov_new = tf.reduce_sum(self.log_vars)
                tr_old_new = tf.reduce_sum(tf.exp(old_log_vars - self.log_vars))

                kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                          tf.reduce_sum(tf.square(self.mu(observes) - old_means) /
                                                        tf.exp(self.log_vars), axis=1) -
                                          self.act_dim)

                # print("here 6.5")
                if self.clipping_range is not None:
                    # print("here 13")
                    pg_ratio = tf.exp(logp - logp_old)
                    # print("here 14")
                    clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.clipping_range[0],
                                                        1 + self.clipping_range[1])
                    # print("here 15")
                    surrogate_loss = tf.minimum(advantages * pg_ratio,
                                                advantages * clipped_pg_ratio)
                    # print("here 16")
                    loss = -tf.reduce_mean(surrogate_loss)
                else:
                    # print("here 17")
                    loss1 = -tf.reduce_mean(advantages * tf.cast(tf.exp(logp - logp_old), dtype=tf.float64))
                    # print("here 18")
                    loss2 = tf.reduce_mean(self.beta * kl)
                    # print("here 19")
                    loss3 = self.eta * tf.square(tf.maximum(0.0, kl - 2.0 * self.kl_targ))
                    # print("here 20")
                    loss = loss1 + tf.cast(loss2 + loss3, dtype=tf.float64)

                # print("here 3")
                grads = tape.gradient(loss, self.mu.trainable_variables)
            # print("here 21")

            #print("here 2")
            batch_losses.append(loss)
            #print("here 22")
            self.optimizer.apply_gradients(zip(grads, self.mu.trainable_variables))
            #print("here 23")

        #print("here 29")
        #if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            #print("here 30")
            #self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            #print("here 31")
            #if self.beta > 30 and self.lr_multiplier > 0.1:
                #print("here 32")
                #self.lr_multiplier /= 1.5
                #print("here 33")
        #elif kl < self.kl_targ / 2:
            #print("here 34")
            #self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            #print("here 35")
            #if self.beta < (1 / 30) and self.lr_multiplier < 10:
                #print("here 36")
                #self.lr_multiplier *= 1.5
                #print("here 37")

        #print("here 28")
        #return kl

    def sample(self, obs):
        obs = obs.reshape(-1, obs.shape[0], obs.shape[1])
        return self.mu.predict(obs) + tf.exp(self.log_vars / 2.0) * tf.compat.v1.random_normal(shape=(self.act_dim,))

    def update(self, observes, actions, advantages, logger):
        observes = observes.reshape(-1, 1, self.obs_dim)

        #print("here 0")
        self.update_model(observes, actions, advantages)
        #print("here 24")
