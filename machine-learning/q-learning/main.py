import gym
from experiments import *

def main():
    env = gym.make("Taxi-v3")
    # train(env)
    # test_prob(env)
    # test_learn_rate(env)
    test_dyskonto(env)
    # test_variation(env)
    env.close()

if __name__ == "__main__":
    main()