import numpy as np

def q_learning_train(observation_space, action_space, iter, env, learning_rate, dyskonto, random_action_prob):
    reward_table = np.zeros((observation_space, action_space))
    for _ in range(0, iter):
        state = env.reset(seed=42)
        done = False
        while not done:
            if np.random.uniform(0, 1) > random_action_prob:
                action = env.action_space.sample()
            else:
                action = np.where(reward_table[state]==reward_table[state].max())[0][0]
            next_state, reward, done, _ = env.step(action)
            curr_reward = reward_table[int(state), int(action)]
            max_reward_next_step = reward_table[next_state].max()
            new_reward = (1 - learning_rate) * curr_reward + learning_rate * (reward + dyskonto*max_reward_next_step)
            reward_table[int(state), int(action)] = new_reward
            state = next_state
    return reward_table

def q_learning_valid(observation_space, action_space, iter, env, learning_rate, dyskonto, random_action_prob):
    reward_table = np.zeros((observation_space, action_space))
    sum_cost = []
    for _ in range(0, iter):
        cost = 0
        state = env.reset(seed=42)
        done = False
        while not done:
            if np.random.uniform(0, 1) > random_action_prob:
                action = env.action_space.sample()
            else:
                action = np.where(reward_table[state]==reward_table[state].max())[0][0]
            next_state, reward, done, _ = env.step(action)
            curr_reward = reward_table[int(state), int(action)]
            max_reward_next_step = reward_table[next_state].max()
            new_reward = (1 - learning_rate) * curr_reward + learning_rate * (reward + dyskonto*max_reward_next_step)
            reward_table[int(state), int(action)] = new_reward
            state = next_state
            cost+=reward 
        sum_cost.append(cost)
    return reward_table, sum_cost

def q_learning_test(env, reward_table):
    state = env.reset(seed=42)
    env.render()
    done = False
    reward_sum = 0
    while not done:
        action = np.where(reward_table[state]==reward_table[state].max())[0][0]
        next_state, reward, done, _ = env.step(action)
        env.render()
        state = next_state
        reward_sum+=reward
    return reward_sum