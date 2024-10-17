import tensorflow as tf
from utils import GracefulKiller
from critic import Critic
from actor import Actor
from train import Train
import threading
#from environment import Solution

class TRPOAgent():
    def __init__(self, params, env_name, episodes, logger):

        self.killer = GracefulKiller()

        #tf.reset_default_graph()
        #seed = params.seed * 1958
        #tf.set_random_seed(seed)
        #np.random.seed(seed)

        self.episodes = episodes
        self.batch_size = params.batch_size
        self.lamda = params.lamda
        self.gamma = params.gamma
        self.env_name = env_name
        self.params = params
        self.threads = params.threads
        self.obs_dim = params.obs_dim
        self.act_dim = params.act_dim
        self.obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())
        self.lock = threading.Lock()
        #self.solution = Solution()
        self.logger = logger

        self.config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.config.gpu_options.allow_growth = True
        self.config.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

        self.policy = Actor(params, self.obs_dim, self.act_dim, self.logger)
        self.val_func = Critic(params, self.obs_dim, self.logger)

    def run(self):

        #self.train = [Train(self.params, self.policy, self.val_func, self.episodes, self.env_name, self.obs_dim,
                            #self.lock, self.solution)
                      #for i in range(self.threads)]

        #for agent in self.train:
            #agent.start()

        #for agent in self.train:
            #agent.join()

        self.close_sess()

    def close_sess(self):
        self.logger.close()
