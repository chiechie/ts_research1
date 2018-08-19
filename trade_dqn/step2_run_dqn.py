# -*- coding: utf-8 -*-
# Standard library
from os.path import join
# Third Party Library
import pandas as pd
from sklearn.cross_validation import train_test_split
try:
    import cPickle as pickle
except:
    import _pickle as pickle
# My Library
from settings import Config_json, get_user_data_dir
from common.path_helper import load_data, loadPklfrom
from trade_dqn.dqn_model import DQN, new_stage_data, show_trader_path
# ---------------------------------------------------------
# Hyper Parameters
EPISODE = 100  # Episode limitation
STEP = 9  # Steps in an episode
TEST = 10  # The number of experiment test every 100 episode
ITERATION = 3
config_json = Config_json()
root_dir = get_user_data_dir()
input_dir = join(root_dir, config_json.get_config("rf_test_data"))

data_dict = {}


def get_intial_data(data_path, data_dict_path, supervised_y_data_path):
    data = load_data(data_path, episode=10)
    # np.array(data).shape == 62773, 10, 20
    global data_dict
    supervised_y_data = loadPklfrom(supervised_y_data_path)
    x_train, x_test, y_train, y_test = train_test_split(data, supervised_y_data, test_size=0.10, random_state=123)
    data_dictionary = {}
    #here one is portfolio value
    data_dictionary["input"] = len(x_train[0][0]) + 1

    # short, buy and hold
    data_dictionary["action"] = 3
    data_dictionary["hidden_layer_1_size"] = 40

    # will be using later
    data_dictionary["hidden_layer_2_size"] = 20
    data_dictionary["x_train"] = x_train
    data_dictionary["x_test"] = x_test
    data_dictionary["y_test"] = y_test
    data_dictionary["y_train"] = y_train
    return data_dictionary



def main():
    # initialize OpenAI Gym env and dqn agent
    # env = gym.make(ENV_NAME)
    data_dictionary = get_intial_data(join(input_dir, "data.pkl"),
                                      join(input_dir, "data_dict.pkl"))
    #初始化agent
    agent = DQN(data_dictionary)

    test_rewards = {}
    # ITERATION为全局变量，实验重复次数
    for iter in range(ITERATION):
        data = data_dictionary["x_train"]
        for episode_idx, episode_data in enumerate(data):
            # initialize task
            # len(episode_data) == 10
            portfolio = 0
            portfolio_value = 0
            # Train
            total_reward = 0
            for step in range(STEP):
                tmp = env_stage_data(agent, step, episode_data, portfolio, portfolio_value, True)
                state, action, next_state, reward, done, portfolio, portfolio_value = tmp
                total_reward += reward
                # agent在新数据（reward， next_state）到达之后,在perceive中实现:
                # 1. 维护q_table：将新的数据存入，旧的输出丢掉
                # 2. 重新训练Q_network
                
                agent.perceive(state, action, reward, next_state, done)
    
                if done:
                    break

            # Test every 100 episodes
            if episode_idx % 100 == 0 and episode_idx > 10:
                total_reward = 0
                for i in range(10):
                    """对没一批数据，进行10次试验，求平均ave_reward"""
                    for step in range(STEP):
                        tmp = env_stage_data(agent, step, episode_data, portfolio, portfolio_value, True)
                        state, action, next_state, reward, done, portfolio, portfolio_value = tmp
                        total_reward += reward
                        if done:
                            break
                ave_reward = total_reward / 10
                print('episode: ', episode, 'Evaluation Average Reward:',ave_reward)

        # on test data
        data = data_dictionary["x_test"]
        iteration_reward = []
        for episode in range(len(data)):
            episode_data = data[episode]
            portfolio = 0
            portfolio_list = []
            portfolio_value = 0
            portfolio_value_list = []
            reward_list = []
            total_reward = 0
            action_list = []
            for step in range(STEP):
                tmp = env_stage_data(agent, step, episode_data, portfolio, portfolio_value, False)
                state, action, next_state, reward, done, portfolio, portfolio_value = tmp
                action_list.append(action)
                portfolio_list.append(portfolio)
                portfolio_value_list.append(portfolio_value)
                reward_list.append(reward)
                total_reward += reward
                if done:
                    episode_reward = show_trader_path(action_list, episode_data, portfolio_list, portfolio_value_list,
                                                      reward_list)
                    iteration_reward.append(episode_reward)
                    break
                    # print 'episode: ',episode,'Testing Average Reward:',total_reward
        avg_reward = sum(iteration_reward)  # / float(len(iteration_reward))
        # print(avg_reward)
        test_rewards[iter] = [iteration_reward, avg_reward]
    for key, value in test_rewards.iteritems():
        print(value[0])
    for key, value in test_rewards.iteritems():
        print(key)
        print(value[1])


def env_stage_data(agent, step, episode_data, portfolio, portfolio_value, train):
    # state :[初始股价， 持有量]
    state = episode_data[step] + [portfolio]
    if train:
        action = agent.egreedy_action(state)  # e-greedy action for train
    else:
        action = agent.action(state)
    # print(step)
    if step < STEP - 2:
        new_state = episode_data[step + 1]
    else:
        new_state = episode_data[step + 1]
    if step == STEP - 1:
        done = True
    else:
        done = False
    next_state, reward, done, portfolio, portfolio_value = new_stage_data(action, portfolio, state, new_state,
                                                                          portfolio_value, done, episode_data[step])
    return state, action, next_state, reward, done, portfolio, portfolio_value


if __name__ == '__main__':
    main()
