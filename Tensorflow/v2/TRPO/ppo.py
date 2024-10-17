import threading
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tensorflow.keras as keras

from environment import Environment

class PPOWorker(threading.Thread):
    def __init__(self, params, env_name, logger, model, lock, solution):
        super(PPOWorker, self).__init__()

        self.env = Environment(env_name, params, logger)
        self.model = model
        self.lock = lock
        self.solution = solution
        self.logger = logger
        self.lock.acquire()
        self.solution.incrementThreadCount()
        self.thread = self.solution.getThreadCount()
        print("self.thread", self.thread)
        self.lock.release()

        self.maxnonzero = 0
        self.completeschedules = 0
        self.LowerBound = self.env.GetLowerBound()
        self.AllSchedule = []

        self.optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
        self.huber_loss = keras.losses.Huber()
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.running_reward = 0
        self.episode_count = 0
        self.episodes = params.episodes
        self.batch_size = params.batch_size

        #self.seed = 42
        #self.gamma = 0.99  # Discount factor for past rewards
        self.gamma = params.gamma
        #self.max_steps_per_episode = 10000
        self.max_steps_per_episode = params.max_steps_per_episode
        #env = gym.make("CartPole-v0")  # Create the environment
        #env.seed(seed)
        self.eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
        self.act_dim = params.act_dim
        self.node_select = params.node_select

    def run(self):
        startTime = datetime.now()
        self.logger.Critical("Thread", self.thread, "start time", startTime)
        self.BestSchedule = self.env.getSchedule()
        while True:  # Run until solved
            state = self.env.reset()

            episode_reward = 0
            with tf.GradientTape() as tape:
                for timestep in range(1, self.max_steps_per_episode):
                    # env.render(); Adding this line would show the attempts
                    # of the agent in a pop up window.
                    #print("state.shape 1", state.shape)
                    state = state.reshape(1, 1, state.shape[0])
                    #print("state.shape 2", state.shape)
                    state = tf.convert_to_tensor(state)
                    #state = tf.expand_dims(state, 0)

                    # Predict action probabilities and estimated future rewards
                    # from environment state

                    #prob = self.actor(np.array([state]))
                    #prob = prob.numpy()
                    #dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
                    #action = dist.sample()
                    #return int(action.numpy()[0])

                    action_probs, critic_value = self.model(state)
                    #print("action_probs", action_probs, "critic_value", critic_value,
                          #"critic_value[0, 0]", critic_value[0, 0])
                    self.critic_value_history.append(critic_value[0, 0])
                    # Sample action from action probability distribution
                    #action = np.random.choice(self.act_dim, p=np.squeeze(action_probs))
                    action_size = np.random.choice(self.act_dim)
                    #print("action_size", action_size)
                    prob = action_probs.numpy()
                    dist = tfp.distributions.Categorical(probs=prob, dtype=tf.int32)
                    action2 = dist.sample(sample_shape=(action_size))
                    #print("action2", action2)
                    action3 = action2.numpy()
                    #print("action2", action2)
                    #print("action3.shape", action3.shape)
                    action3 = action3.reshape(action3.shape[1], action3.shape[0])
                    action3 = np.unique(np.squeeze(action3))
                    #print("action3", action3)
                    #print("action3.shape", action3.shape)

                    #nonzero = np.nonzero(action_probs)
                    #print("action_size", action_size, "nonzero", nonzero, "len(nonzero)", len(nonzero))
                    #if action_size >= len(nonzero) :
                        #action_size = len(nonzero) - 1
                        #action = np.random.choice(self.act_dim, replace=False,
                                                            #size=action_size, p=np.squeeze(action_probs))
                    #print("action_size", action_size, "action", action)
                    #action = action_probs
                    #action = np.squeeze(action, axis=0)
                    #print("action", action)
                    #action2, count = self.getAction(action)
                    #probs = self.getProbs(action, action_probs)
                    #self.action_probs_history.append(tf.math.log(action_probs[0, action2]))
                    self.action_probs_history.append(tf.math.log(action_probs))
                    action = action3
                    # Apply the sampled action in our environment
                    state, reward, done, _ = self.env.step(action)
                    #print("next state", state, "reward", reward, "done", done)
                    self.rewards_history.append(reward)
                    episode_reward += reward

                    if done:
                        break

                schedule = self.env.getSchedule()
                print("Thread", self.thread, "schedule", schedule)
                self.AllSchedule.append(schedule)

                # Update running reward to check condition for solving
                self.running_reward = 0.05 * episode_reward + (1 - 0.05) * self.running_reward

                # Calculate expected value from rewards
                # - At each timestep what was the total reward received after that timestep
                # - Rewards in the past are discounted by multiplying them with gamma
                # - These are the labels for our critic
                returns = []
                discounted_sum = 0
                for r in self.rewards_history[::-1]:
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
                returns = returns.tolist()
                #print("returns", returns)
                # Calculating loss values to update our network
                history = zip(self.action_probs_history, self.critic_value_history, returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    #print("ret", ret, "value", value, "diff", diff)
                    #print("-log_prob * diff", -log_prob * diff)
                    actor_losses.append(-log_prob * diff)  # actor loss

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    #print("self.huber_loss(value, ret)", self.huber_loss(value, ret))
                    critic_losses.append(
                        #self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                        self.huber_loss(value, ret)
                    )

                schedule, foundbest, completeschedules = self.GetBestSchedule()
                maxslot = max(schedule)
                self.lock.acquire()
                # Backpropagation
                #print("actor_losses", actor_losses)
                #print("critic_losses", critic_losses)
                loss_value = sum(actor_losses) + sum(critic_losses)
                grads = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                if self.completeschedules == 0 or maxslot > self.LowerBound:
                    if self.solution.FoundBest == False:
                        self.episodes += self.batch_size
                else:
                    self.solution.FoundBest = True
                Time = datetime.now().strftime("%H-%M-%S")
                self.logger.Critical("Thread", self.thread, "Time", Time, "Episode", self.episode_count, "nodes",
                                     self.maxnonzero, "max", maxslot, "complete", completeschedules, "total",
                                     self.completeschedules, "totalepisodes", self.episodes)
                self.lock.release()

                # Clear the loss and reward history
                self.action_probs_history.clear()
                self.critic_value_history.clear()
                self.rewards_history.clear()

            # Log details
            self.episode_count += 1
            #if self.episode_count % 10 == 0:
                #template = "running reward: {:.2f} at episode {}"
                #print(template.format(self.running_reward, self.episode_count))

            #if running_reward > 195:  # Condition to consider the task solved
                #print("Solved at episode {}!".format(self.episode_count))
                #break
            if self.solution.FoundBest == True:
                break

        maxslot = max(self.BestSchedule)
        nonzero = np.count_nonzero(self.BestSchedule)
        self.logger.Critical("Thread", self.thread, 'Best', self.BestSchedule, "nonzero", nonzero, "max", maxslot,
                             "complete", self.completeschedules)
        self.logger.Critical("Thread", self.thread, "Time taken:", datetime.now() - startTime)

    def getProbs(self, action, actionprobs):
        probs = []
        for node in action:
            probs.append(actionprobs[0,node])
        return probs

    def GetBestSchedule(self):
        length = len(self.AllSchedule)
        #self.logger.Debug("length", length)
        best = self.BestSchedule
        #self.logger.Debug("best", best)
        maxbest = max(best)
        minbest = min(best)
        #self.logger.Debug("best", best, "maxbest", maxbest)
        foundbest = False
        complete = 0
        for i in range(length):
            nextschedule = self.AllSchedule[i]
            maxslot = max(nextschedule)
            minslot = min (nextschedule)
            nonzero = np.count_nonzero(nextschedule)
            #self.logger.Debug("best", best, "maxbest", maxbest, "nextschedule", nextschedule, "maxslot", maxslot, "minslot", minslot)
            if nonzero > self.maxnonzero:
                self.maxnonzero = nonzero
                best = nextschedule
                maxbest = maxslot
                self.maxslot = maxslot
            if minslot > 0 :
                self.completeschedules += 1
                complete += 1
                if minbest == 0 :
                    best = nextschedule
                    maxbest = maxslot
                    foundbest = True
                if maxslot < maxbest:
                    best = nextschedule
                    maxbest = maxslot
                    self.maxslot = maxslot
                    if maxslot <= self.LowerBound:
                        foundbest = True
                        self.logger.Critical("best", best)
        self.BestSchedule = best
        self.AllSchedule = []
        return best, foundbest, complete
