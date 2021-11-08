import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn
import torch.nn.functional as Func
import random
from torch.autograd import Variable
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimplePolicy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(SimplePolicy, self).__init__()
        learning_rate = 0.01
        self.num_actions = a_size
        self.linear1 = nn.Linear(s_size, h_size)
        self.linear2 = nn.Linear(h_size, self.num_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        f = Func.relu(self.linear1(x))
        f = Func.softmax(self.linear2(f), dim=1)
        return f


class StateValueNetwork(nn.Module):
    # Takes in state
    def __init__(self, s_size=4, h_size=16):
        super(StateValueNetwork, self).__init__()
        learning_rate = 0.01
        self.linear1 = nn.Linear(s_size, h_size)
        self.linear2 = nn.Linear(h_size, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        #input layer
        f = self.linear1(x)
        #activiation relu
        f = Func.relu(f)
        #get state value
        state_value = self.linear2(f)
        return state_value

def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    returns = []
    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + gamma ** pw * r
            pw = pw + 1
        returns.append(Gt)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (
            returns.std())
    return returns


def reinforce(env, policy_model, seed, learning_rate,
              number_episodes,
              max_episode_length,
              gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    env.seed(seed)
    policy = []
    numsteps = []
    avgNumsteps = []
    allRewards = []
    for episode in range(number_episodes):
        state = env.reset()
        lProbs = []
        scores = []

        for steps in range(max_episode_length):
            # you can uncomment this to display cart
            # env.render()
            state = torch.from_numpy(state).float().unsqueeze(0)
            #this section is from https://gist.github.com/cyoon1729/bc41d466b868ea10e794a7c04321ff3b#file-reinforce_model-py
            probs = policy_model.forward(Variable(state))
            action = np.random.choice(env.action_space.n, p=np.squeeze(probs.detach().numpy()))
            lProb = torch.log(probs.squeeze(0)[action])
            nextState,score,done, _ = env.step(action)
            lProbs.append(lProb)
            scores.append(score)

            if done:
                returns = compute_returns(scores, gamma)
                #this part calculates the policy gradient
                #this section is from https://gist.github.com/cyoon1729/3920da556f992909ace8516e2f321a7c#file-reinforce_update-py
                policyGrad = []
                #training policy
                for logProb, Gt in zip(lProbs, returns):
                    policyGrad.append(-logProb * Gt)
                policy_model.optimizer.zero_grad()
                policyGrad = torch.stack(policyGrad).sum()
                policyGrad.backward()
                policy_model.optimizer.step()
                numsteps.append(steps)
                avgNumsteps.append(np.mean(numsteps[-10:]))
                allRewards.append(np.sum(scores))
                if episode % 1 == 0:
                    print("episode: {}, total reward: {}, average_reward: {}, length: {}".format(episode,np.round(
                                                                                                                  np.sum(
                                                                                                                      scores),
                                                                                                                  decimals=3),
                                                                                                              np.round(
                                                                                                                  np.mean(
                                                                                                                      allRewards[
                                                                                                                      -10:]),
                                                                                                                  decimals=3),
                                                                                                      steps))
                break
            state = nextState
    env.close()
    return policy, allRewards


def compute_returns_naive_baseline(rewards, gamma):
    returns = []
    #calculates the return values
    for t in range(len(rewards)):
        Gt = 0
        for r in rewards[t:]:
            Gt = Gt * gamma + r
        returns.append(Gt)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (
        returns.std())
    return returns


def reinforce_naive_baseline(env, policy_model, state_model, seed, learning_rate,
                             number_episodes,
                             max_episode_length,
                             gamma, verbose=True):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    env.seed(seed)
    policy = []
    numsteps = []
    avgNumsteps = []
    allRewards = []
    for episode in range(number_episodes):
        state = env.reset()
        lProbs = []
        scores = []
        states = []
        for steps in range(max_episode_length):
            #you can uncomment this to display cart
            #env.render()
            state = torch.from_numpy(state).float().unsqueeze(0)#.to(device)
            #this section is from https://gist.github.com/cyoon1729/bc41d466b868ea10e794a7c04321ff3b#file-reinforce_model-py
            probs = policy_model.forward(Variable(state))
            action = np.random.choice(env.action_space.n, p=np.squeeze(probs.detach().numpy()))
            lprob = torch.log(probs.squeeze(0)[action])
            nextState, score, done, _ = env.step(action)
            lProbs.append(lprob)
            scores.append(score)
            states.append(state)
            if done:
                returns = compute_returns_naive_baseline(scores, gamma)
                #this section calculates the state values
                #calculate MSE loss
                stateValues = []
                for i in states:
                    stateValues.append(state_model.forward(Variable(i)))
                stateValues = torch.stack(stateValues).squeeze()
                valLoss = Func.mse_loss(stateValues, returns)
                #backpropagate
                state_model.optimizer.zero_grad()
                valLoss.backward()
                state_model.optimizer.step()
                deltas = []
                for gt, val in zip(returns, stateValues):
                    deltas.append(gt-val)
                deltas = torch.tensor(deltas)#.to(device)
                #this section is where we calculate the policy gradient
                #this section is from https://gist.github.com/cyoon1729/3920da556f992909ace8516e2f321a7c#file-reinforce_update-py
                policyGrad = []
                #training policy
                for logProb, Dt in zip(lProbs, deltas):
                    policyGrad.append(-logProb * Dt)
                policy_model.optimizer.zero_grad()
                policyGrad = torch.stack(policyGrad).sum()
                #backpropagate
                policyGrad.backward()
                policy_model.optimizer.step()
                numsteps.append(steps)
                avgNumsteps.append(np.mean(numsteps[-10:]))
                allRewards.append(np.sum(scores))
                if episode % 1 == 0:
                    print(
                        "episode: {}, total reward: {}, average_reward: {}, length: {}".format(episode, np.round(
                            np.sum(
                                scores),
                            decimals=3),
                                                                                                 np.round(
                                                                                                     np.mean(
                                                                                                         allRewards[
                                                                                                         -10:]),
                                                                                                     decimals=3),
                                                                                                 steps))
                break
            state = nextState
    env.close()
    return policy, allRewards


def run_reinforce():
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 500
    np.random.seed(53)
    seeds = np.random.randint(50, size=5)
    size = seeds.shape[0]
    allScores = np.zeros(150)
    for i in seeds:
        #env.seed(0)
        policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)
        policy, scores = reinforce(env=env, policy_model=policy_model, seed=42, learning_rate=1e-2,
                                   number_episodes=150,
                                   max_episode_length=1000,
                                   gamma=0.99,
                                   verbose=True)
        env.reset()
        arr = np.array(scores)
        allScores = allScores + arr
    allScores = allScores / size
    return allScores, arr
    # Plot learning curve
    #gamma = 1.0
    #max_episode_length=1000
    #h_size=50
def investigate_variance_in_reinforce(baselineScore, reinforceScore):
    env = gym.make('CartPole-v1')
    seeds = np.random.randint(1000, size=5)
    baselineScorePosi = np.std(baselineScore, axis=0) + baselineScore
    baselineScoreNega =  - np.std(baselineScore, axis=0) + baselineScore
    reinforceScorePosi = np.std(reinforceScore, axis=0) + reinforceScore
    reinforceScoreNega = - np.std(reinforceScore, axis=0) + reinforceScore
    plt.plot(baselineScore)
    #plt.plot(baselineScorePosi)
    #plt.plot(baselineScoreNega)
    plt.plot(reinforceScore)
    #plt.plot(reinforceScorePosi)
    #plt.plot(reinforceScoreNega)
    plt.legend(['Baseline', 'Reinforce'], loc='upper left')
    plt.show()
    #raise NotImplementedError

    #return mean, std


def run_reinforce_with_naive_baseline():
    env = gym.make('CartPole-v1')
    env._max_episode_steps = 500
    np.random.seed(53)
    seeds = np.random.randint(50, size=5)
    size = seeds.shape[0]
    print(seeds)
    allScores = np.zeros(150)
    for i in seeds:
        print(i)
        #env.seed(0)
        policy_model = SimplePolicy(s_size=env.observation_space.shape[0], h_size=50, a_size=env.action_space.n)#.to(device)
        stateval_model = StateValueNetwork(s_size=env.observation_space.shape[0], h_size=50)#.to(device)
        policy, scores = reinforce_naive_baseline(env=env, policy_model=policy_model, state_model=stateval_model, seed=42, learning_rate=1e-2,
                                   number_episodes=150,
                                   max_episode_length=1000,
                                   gamma=0.99,
                                   verbose=True)
        env.reset()
        arr = np.array(scores)
        allScores = allScores + arr
    allScores = allScores / size
    return allScores, arr
    #raise NotImplementedError


if __name__ == '__main__':
    mean = 0
    std = 0
    baselineScores, baselineScore = run_reinforce_with_naive_baseline()
    reinforceScores, reinforceScore = run_reinforce()
    investigate_variance_in_reinforce(baselineScores, reinforceScores)
    #mean, std = investigate_variance_in_reinforce()


