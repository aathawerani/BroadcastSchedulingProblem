import sys

from ppo import PPOAgent
import config

def main(*args):
    env_name = ""

    for arg in sys.argv[1:]:
        #print(arg)
        env_name = arg

    params = config.GetNetworkParams(env_name)

    agent = PPOAgent(env_name)
    #agent.run() # use as PPO
    agent.train(n_threads=params.threads) # use as APPO

if __name__ == '__main__':
    main()
