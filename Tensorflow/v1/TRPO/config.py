import collections
from datetime import datetime

from training import Training
from trpoagent import TRPOAgent
from netstat import NetworkStats
from environment import Environment
from utils import Logger

class Config():
    def __init__(self):
        self.date = datetime.utcnow().strftime("%y%m%d")  # create unique directories
        self.time = datetime.utcnow().strftime("%H%M%S")  # create unique directories

        Variables = collections.namedtuple('params', ['device', 'node_select', 'batch_size', 'lamda', 'gamma',
                                                      'step_size', 'seed', 'chances', 'threads', 'obs_dim', 'act_dim',
                                                      'actor_beta', 'actor_eta', 'actor_epochs', 'actor_lr_multiplier',
                                                      'actor_kl_targ', 'actor_policy_logvar', 'actor_clipping_range',
                                                      'actor_hidden_layers', 'actor_hidden_layer_units',
                                                      'critic_epochs', 'critic_hidden_layers',
                                                      'critic_hidden_layer_units', 'path',
                                                      'max_steps_per_episode', 'episodes'])
        self.params = Variables("/cpu:0", 0.5, 20, 0.995, 0.98,
                           1e-3, 777, 20, 1, 100, 100,
                           1.0, 50, 20, 1.0,
                           0.003, -1.0, None,
                           3, [100, 100, 100],
                           10, 4, [32, 16, 32, 1],
            "E:\\aaht14\\OneDrive\\OneDrive - Institute of Business Administration\\TD\\CodeBSP_20200214\\cases\\",
                                10000, 20)

    def NetworkStats(self, case):
        logger = Logger(logname=case, date=self.date, time=self.time, loglevel=2)
        env = Environment(case, self.params, logger)

    def run(self, env_name,  episodes):
        logger = Logger(logname=env_name, date=self.date, time=self.time, loglevel=2)
        logger.Info(self.params)
        #trpo = TRPOAgent(self.params, env_name, episodes, logger)
        #trpo.run()
        training = Training(self.params, env_name, logger)
        training.run()

    def network1(self):
        self.params = self.params._replace(obs_dim=9, act_dim=9, actor_hidden_layer_units=[9, 9, 9])
        self.run("network_1", 100)
        #self.run("network_1", 100)
        #self.run("network_1", 100)

    def network2(self):
        self.params = self.params._replace(obs_dim=16, act_dim=16, actor_hidden_layer_units=[16, 16, 16])
        self.run("network_2", 100)
        #self.run("network_2", 100)
        #self.run("network_2", 100)

    def network3(self):
        self.params = self.params._replace(obs_dim=25, act_dim=25, actor_hidden_layer_units=[25, 25, 25])
        self.run("network_3", 100)
        #self.run("network_3", 30000)
        #self.run("network_3", 30000)

    def network4(self):
        self.params = self.params._replace(obs_dim=36, act_dim=36, actor_hidden_layer_units=[36, 36, 36])
        self.run("network_4", 100)
        #self.run("network_4", 30000)
        #self.run("network_4", 30000)

    def network5(self):
        self.params = self.params._replace(obs_dim=49, act_dim=49, actor_hidden_layer_units=[49, 49, 49])
        self.run("network_5", 100)
        #self.run("network_5", 30000)
        #self.run("network_5", 30000)

    def BM1(self):
        self.params = self.params._replace(obs_dim=15, act_dim=15, actor_hidden_layer_units=[15, 15, 15])
        self.run("BM1", 100)
        #self.run("BM1", 100)
        #self.run("BM1", 100)

    def BM2(self):
        self.params = self.params._replace(obs_dim=30, act_dim=30, actor_hidden_layer_units=[30, 30, 30])
        self.run("BM2", 100)
        #self.run("BM2", 100)
        #self.run("BM2", 100)

    def BM3(self):
        self.params = self.params._replace(obs_dim=40, act_dim=40, actor_hidden_layer_units=[40, 40, 40])
        self.run("BM3", 100)
        #self.run("BM3", 100)
        #self.run("BM3", 100)

    def Case1(self):
        self.params = self.params._replace(obs_dim=100, act_dim=100, actor_hidden_layer_units=[100, 100, 100])
        self.run("case1r1", 100)
        self.run("case1r2", 100)
        self.run("case1r3", 100)

    def Case6(self):
        self.params = self.params._replace(obs_dim=100, act_dim=100, actor_hidden_layer_units=[100, 100, 100])
        self.run("case6r1", 100)
        self.run("case6r2", 100)
        self.run("case6r3", 100)

    def Case11(self):
        self.params = self.params._replace(obs_dim=100, act_dim=100, actor_hidden_layer_units=[100, 100, 100])
        self.run("case11r1", 100)
        self.run("case11r2", 100)
        self.run("case11r3", 100)

    def Case16(self):
        self.params = self.params._replace(obs_dim=100, act_dim=100, actor_hidden_layer_units=[100, 100, 100])
        self.run("case16r1", 100)
        self.run("case16r2", 100)
        self.run("case16r3", 100)

    def Case2(self):
        self.params = self.params._replace(obs_dim=300, act_dim=300, actor_hidden_layer_units=[300, 300, 300])
        self.run("case2r1", 100)
        self.run("case2r2", 100)
        self.run("case2r3", 100)

    def Case7(self):
        self.params = self.params._replace(obs_dim=300, act_dim=300, actor_hidden_layer_units=[300, 300, 300])
        self.run("case7r1", 100)
        self.run("case7r2", 100)
        self.run("case7r3", 100)

    def Case12(self):
        self.params = self.params._replace(obs_dim=300, act_dim=300, actor_hidden_layer_units=[300, 300, 300])
        self.run("case12r1", 100)
        self.run("case12r2", 100)
        self.run("case12r3", 100)

    def Case17(self):
        self.params = self.params._replace(obs_dim=300, act_dim=300, actor_hidden_layer_units=[300, 300, 300])
        self.run("case17r1", 100)
        self.run("case17r2", 100)
        self.run("case17r3", 100)

    def Case3(self):
        self.params = self.params._replace(obs_dim=500, act_dim=500, actor_hidden_layer_units=[500, 500, 500])
        self.run("case3r1", 100)
        self.run("case3r2", 100)
        self.run("case3r3", 100)

    def Case8(self):
        self.params = self.params._replace(obs_dim=500, act_dim=500, actor_hidden_layer_units=[500, 500, 500])
        self.run("case8r1", 100)
        self.run("case8r2", 100)
        self.run("case8r3", 100)

    def Case13(self):
        self.params = self.params._replace(obs_dim=500, act_dim=500, actor_hidden_layer_units=[500, 500, 500])
        self.run("case13r1", 100)
        self.run("case13r2", 100)
        self.run("case13r3", 100)

    def Case18(self):
        self.params = self.params._replace(obs_dim=500, act_dim=500, actor_hidden_layer_units=[500, 500, 500])
        self.run("case18r1", 100)
        self.run("case18r2", 100)
        self.run("case18r3", 100)

    def Case4(self):
        self.params = self.params._replace(obs_dim=750, act_dim=750, actor_hidden_layer_units=[750, 750, 750])
        self.run("case4r1", 100)
        self.run("case4r2", 100)
        self.run("case4r3", 100)

    def Case9(self):
        self.params = self.params._replace(obs_dim=750, act_dim=750, actor_hidden_layer_units=[750, 750, 750])
        self.run("case9r1", 100)
        self.run("case9r2", 100)
        self.run("case9r3", 100)

    def Case14(self):
        self.params = self.params._replace(obs_dim=750, act_dim=750, actor_hidden_layer_units=[750, 750, 750])
        self.run("case14r1", 100)
        self.run("case14r2", 100)
        self.run("case14r2", 100)

    def Case19(self):
        self.params = self.params._replace(obs_dim=750, act_dim=750, actor_hidden_layer_units=[750, 750, 750])
        self.run("case19r1", 100)
        self.run("case19r2", 100)
        self.run("case19r3", 100)

    def Case5(self):
        self.params = self.params._replace(obs_dim=1000, act_dim=1000, actor_hidden_layer_units=[1000, 1000, 1000])
        self.run("case5r1", 100)
        self.run("case5r2", 100)
        self.run("case5r3", 100)

    def Case10(self):
        self.params = self.params._replace(obs_dim=1000, act_dim=1000, actor_hidden_layer_units=[1000, 1000, 1000])
        self.run("case10r1", 100)
        self.run("case10r2", 100)
        self.run("case10r3", 100)

    def Case15(self):
        self.params = self.params._replace(obs_dim=1000, act_dim=1000, actor_hidden_layer_units=[1000, 1000, 1000])
        self.run("case15r1", 100)
        self.run("case15r2", 100)
        self.run("case15r3", 100)

    def Case20(self):
        self.params = self.params._replace(obs_dim=1000, act_dim=1000, actor_hidden_layer_units=[1000, 1000, 1000])
        self.run("case20r1", 100)
        self.run("case20r2", 100)
        self.run("case20r3", 100)



