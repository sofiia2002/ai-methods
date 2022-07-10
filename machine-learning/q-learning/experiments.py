from q_learning import *
import pandas as pd
import matplotlib.pyplot as plt

def test_dyskonto(env):
    observation_space = 500
    action_space = 6
    iterations = 350
    learning_rate = 0.1
    random_action_prob = 0.8

    costs = []
    dyskonto = [0.2, 0.4, 0.6, 0.8]
    for i in range(0, len(dyskonto)):
        _, cost = q_learning_valid(observation_space, action_space, iterations, env, learning_rate, dyskonto[i], random_action_prob)
        costs.append(cost)
    plot(costs, dyskonto)

def test_learn_rate(env):
    observation_space = 500
    action_space = 6
    iterations = 350
    dyskonto = 0.6
    random_action_prob = 0.8

    rates = [0.1, 0.3, 0.5, 0.7]
    costs = []
    for i in range(0, len(rates)):
        _, cost = q_learning_valid(observation_space, action_space, iterations, env, rates[i], dyskonto, random_action_prob)
        costs.append(cost)
    plot(costs, rates)

def test_prob(env):
    observation_space = 500
    action_space = 6
    iterations = 350
    dyskonto = 0.6
    learning_rate = 0.1

    prob = [0.2, 0.4, 0.6, 0.8]
    costs = []
    for i in range(0, len(prob)):
        _, cost = q_learning_valid(observation_space, action_space, iterations, env, learning_rate, dyskonto, prob[i])
        costs.append(cost)
    plot(costs, prob)

def test_variation(env):
    epochs = 50
    observation_space = 500
    action_space = 6
    iterations = 60
    dyskonto = 0.8
    learning_rate = 0.7
    random_action_prob = 0.8

    costs = []
    for _ in range(0, epochs):
        _, cost = q_learning_valid(observation_space, action_space, iterations, env, learning_rate, dyskonto, random_action_prob)
        costs.append(cost)
    plot_variation(np.array(costs))

def plot_variation(y):
    aggregated_values = pd.DataFrame(y).agg([np.mean, np.std])
    x = np.arange(0, len(aggregated_values.iloc[0, :]))
    plt.errorbar(x, aggregated_values.iloc[0, :], yerr=aggregated_values.iloc[1, :], fmt='-o')
    plt.ylabel("Cost")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.show()

def train(env):
    observation_space = 500
    action_space = 6
    iterations = 200
    dyskonto = 0.8
    learning_rate = 0.7
    random_action_prob = 0.8

    reward_table = q_learning_train(observation_space, action_space, iterations, env, learning_rate, dyskonto, random_action_prob)
    test = q_learning_test(env, reward_table)
    print("Reward table:")
    print("Cost after training:")
    print(test)

def plot(data, labels):
    fig, ax = plt.subplots()
    for i in range(0,len(data)):
        ax.plot(data[i], label=labels[i])  
    plt.ylabel("Cost")
    plt.xlabel("Number of iterations")
    plt.legend()
    plt.show()