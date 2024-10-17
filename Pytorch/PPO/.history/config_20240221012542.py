import os
import collections

#from environmentwrapper import BSPEnvironmentWrapper
#from tf_agents.environments import py_environment

def GetParams():
    #print(tf.__version__)

    logpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs')

    #print("self.rootdir", root_dir)
    cases = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cases')
    #print("self.cases", cases)

    Variables = collections.namedtuple('params',
        [
            'chances', 'episodes', 'learning_rate', 'num_epochs', 'threads',
            'action_size', 'num_nodes', 'num_slots',
            'discount_factor', 'seed', 'num_environment_steps', 'collect_episodes_per_iteration',
            'num_parallel_environments', 'replay_buffer_capacity', 
            'num_eval_episodes', 'cases', 'logpath', 'lambda_value', 'initial_adaptive_kl_beta',
            'adaptive_kl_target', 'actor_lstm_size', 'critic_lstm_size', 'actor_fc_layers', 'value_fc_layers',
            'reward_multiplier', 'node_selection', 'max_iterations'
            ])

    params = Variables(
        10, 10000, 0.0001, 10, 2,
        1000, 1000, 1000,
        0.95, 999, 9000000000, 30,
        30, 1001,
        30, cases, logpath, 0.95, 1.0,
        0.003, (9,9), (9,9), (1, 1), (1, 1),
        1, 0.5, 10000
        )

    return params

#def load(env_name, logpath) -> py_environment.PyEnvironment:
    #return BSPEnvironmentWrapper(env_name, logpath)

def GetNetworkParams(env_name):
    params = GetParams()
    if env_name == "network_1": return network1(params)
    elif env_name == "network_2": return network2(params)
    elif env_name == "network_3": return network3(params)
    elif env_name == "network_4": return network4(params)
    elif env_name == "network_5": return network5(params)
    elif env_name == "BM1": return BM1(params)
    elif env_name == "BM2": return BM2(params)
    elif env_name == "BM3": return BM3(params)
    elif env_name == "case1r1": return Case1r1(params)
    elif env_name == "case1r2": return Case1r2(params)
    elif env_name == "case1r3": return Case1r3(params)
    elif env_name == "case6r1": return Case6r1(params)
    elif env_name == "case6r2": return Case6r2(params)
    elif env_name == "case6r3": return Case6r3(params)
    elif env_name == "case11r1": return Case11r1(params)
    elif env_name == "case11r2": return Case11r2(params)
    elif env_name == "case11r3": return Case11r3(params)
    elif env_name == "case16r1": return Case16r1(params)
    elif env_name == "case16r2": return Case16r2(params)
    elif env_name == "case16r3": return Case16r3(params)
    elif env_name == "case2r1": return Case2r1(params)
    elif env_name == "case2r2": return Case2r2(params)
    elif env_name == "case2r3": return Case2r3(params)
    elif env_name == "case7r1": return Case7r1(params)
    elif env_name == "case7r2": return Case7r2(params)
    elif env_name == "case7r3": return Case7r3(params)
    elif env_name == "case12r1": return Case12r1(params)
    elif env_name == "case12r2": return Case12r2(params)
    elif env_name == "case12r3": return Case12r3(params)
    elif env_name == "case17r1": return Case17r1(params)
    elif env_name == "case17r2": return Case17r2(params)
    elif env_name == "case17r3": return Case17r3(params)
    elif env_name == "case3r1": return Case3r1(params)
    elif env_name == "case3r2": return Case3r2(params)
    elif env_name == "case3r3": return Case3r3(params)
    elif env_name == "case8r1": return Case8r1(params)
    elif env_name == "case8r2": return Case8r2(params)
    elif env_name == "case8r3": return Case8r3(params)
    elif env_name == "case13r1": return Case13r1(params)
    elif env_name == "case13r2": return Case13r2(params)
    elif env_name == "case13r3": return Case13r3(params)
    elif env_name == "case18r1": return Case18r1(params)
    elif env_name == "case18r2": return Case18r2(params)
    elif env_name == "case18r3": return Case18r3(params)
    elif env_name == "case4r1": return Case4r1(params)
    elif env_name == "case4r2": return Case4r2(params)
    elif env_name == "case4r3": return Case4r3(params)
    elif env_name == "case9r1": return Case9r1(params)
    elif env_name == "case9r2": return Case9r2(params)
    elif env_name == "case9r3": return Case9r3(params)
    elif env_name == "case14r1": return Case14r1(params)
    elif env_name == "case14r2": return Case14r2(params)
    elif env_name == "case14r3": return Case14r3(params)
    elif env_name == "case19r1": return Case19r1(params)
    elif env_name == "case19r2": return Case19r2(params)
    elif env_name == "case19r3": return Case19r3(params)
    elif env_name == "case5r1": return Case5r1(params)
    elif env_name == "case5r2": return Case5r2(params)
    elif env_name == "case5r3": return Case5r3(params)
    elif env_name == "case10r1": return Case10r1(params)
    elif env_name == "case10r2": return Case10r2(params)
    elif env_name == "case10r3": return Case10r3(params)
    elif env_name == "case15r1": return Case15r1(params)
    elif env_name == "case15r2": return Case15r2(params)
    elif env_name == "case15r3": return Case15r3(params)
    elif env_name == "case20r1": return Case20r1(params)
    elif env_name == "case20r2": return Case20r2(params)
    elif env_name == "case20r3": return Case20r3(params)

def network1(params):
    params = params._replace(
                            actor_lstm_size=(9,), critic_lstm_size=(9,),
                            actor_fc_layers=(9,), value_fc_layers=(9,))
    return params

def network2(params):
    params = params._replace(
                            actor_lstm_size=(16,), critic_lstm_size=(16,),
                            actor_fc_layers=(16,), value_fc_layers=(16,))
    return params

def network3(params):
    params = params._replace(
                            actor_lstm_size=(25,), critic_lstm_size=(25,),
                            actor_fc_layers=(25,), value_fc_layers=(25,))
    return params

def network4(params):
    params = params._replace(
                            actor_lstm_size=(36,), critic_lstm_size=(36,),
                            actor_fc_layers=(36,), value_fc_layers=(36,))
    return params

def network5(params):
    params = params._replace(
                            actor_lstm_size=(49,), critic_lstm_size=(49,),
                            actor_fc_layers=(49,), value_fc_layers=(49,))
    return params

def BM1(params):
    params = params._replace(
                            actor_lstm_size=(15,), critic_lstm_size=(15,),
                            actor_fc_layers=(15,), value_fc_layers=(15,))
    return params

def BM2(params):
    params = params._replace(
                            actor_lstm_size=(30,), critic_lstm_size=(30,),
                            actor_fc_layers=(30,), value_fc_layers=(30,)) 
    return params

def BM3(params):
    params = params._replace(
                            actor_lstm_size=(40,), critic_lstm_size=(40,),
                            actor_fc_layers=(40,), value_fc_layers=(40,))
    return params

def Case1r1(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case1r2(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case1r3(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case2r1(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))
    return params

def Case2r2(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))
    return params

def Case2r3(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))
    return params

def Case3r1(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case3r2(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case3r3(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case4r1(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case4r2(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case4r3(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case5r1(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params

def Case5r2(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params

def Case5r3(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params

def Case6r1(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case6r2(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case6r3(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case7r1(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))

    return params

def Case7r2(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))

    return params

def Case7r3(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))

    return params

def Case8r1(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case8r2(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case8r3(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case9r1(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case9r2(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case9r3(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case10r1(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params

def Case10r2(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params

def Case10r3(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params

def Case11r1(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case11r2(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case11r3(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case12r1(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))
    return params

def Case12r2(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))
    return params

def Case12r3(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))
    return params

def Case13r1(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case13r2(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case13r3(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(78,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case14r1(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case14r2(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case14r3(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case15r1(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params

def Case15r2(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params

def Case15r3(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params

def Case16r1(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case16r2(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case16r3(params):
    params = params._replace(
                            actor_lstm_size=(100,), critic_lstm_size=(100,),
                            actor_fc_layers=(100,), value_fc_layers=(100,))
    return params

def Case17r1(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))
    return params

def Case17r2(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))
    return params

def Case17r3(params):
    params = params._replace(
                            actor_lstm_size=(300,), critic_lstm_size=(300,),
                            actor_fc_layers=(300,), value_fc_layers=(300,))
    return params

def Case18r1(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case18r2(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case18r3(params):
    params = params._replace(
                            actor_lstm_size=(500,), critic_lstm_size=(500,),
                            actor_fc_layers=(500,), value_fc_layers=(500,))
    return params

def Case19r1(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case19r2(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case19r3(params):
    params = params._replace(
                            actor_lstm_size=(750,), critic_lstm_size=(750,),
                            actor_fc_layers=(750,), value_fc_layers=(750,))
    return params

def Case20r1(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params
    
def Case20r2(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params

def Case20r3(params):
    params = params._replace(
                            actor_lstm_size=(1000,), critic_lstm_size=(1000,),
                            actor_fc_layers=(1000,), value_fc_layers=(1000,))
    return params
