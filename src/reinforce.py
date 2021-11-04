
import sys

import numpy as np
import matplotlib.pyplot as plt
import gym
import minihack
import torch
import torch.nn as nn
import torch.nn.functional as Func
import random
from torch.autograd import Variable
from collections import deque

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimplePolicy(nn.Module):
    def __init__(self, s_size, h_size, a_size):
        super(SimplePolicy, self).__init__()
        learning_rate = 3e-4
        self.linear1 = nn.Linear(s_size, h_size)
        self.linear2 = nn.Linear(h_size, a_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        #print("before: ",x.shape)
        x = torch.flatten(x)
        x = torch.reshape(x, (1,x.shape[0]))
        #print("after: ",x.shape)
        #normalize tensors
        x = torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12, out=None)
        func = Func.relu(self.linear1(x))
        #print("func: ",func)
        func = Func.softmax(self.linear2(func), dim=1)
        return func


def moving_average(a, n):
    ret = np.cumsum(a, dtype=np.float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def compute_returns(rewards, gamma):
    returns = []
    #calculates returns
    for t in range(len(rewards)):
        Gt = 0
        power = 0
        for r in rewards[t:]:
            Gt = Gt + gamma ** power * r
            power = power + 1
        returns.append(Gt)
    #convert returns to tensor
    returns = torch.tensor(returns)
    #normalize returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
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
    policy = []
    env.seed(seed)
    for episode in range(number_episodes):
        state = env.reset()['glyphs']
        logProbs = []
        scores = []
        done = False
        #for steps in range(max_episode_length)
        while True:
            env.render()
            state = torch.from_numpy(state).float().unsqueeze(0)
            #this section is from https://gist.github.com/cyoon1729/bc41d466b868ea10e794a7c04321ff3b#file-reinforce_model-py
            probs = policy_model.forward(Variable(state))
            action = np.random.choice(env.action_space.n, p=np.squeeze(probs.detach().numpy()))
            logProb = torch.log(probs.squeeze(0)[action])
            #score here is just reward
            nextState,score,done, _ = env.step(action)
            logProbs.append(logProb)
            scores.append(score)
            policy.append(state)
            #print(done)
            if done:
                returns = compute_returns(scores, gamma)
                #this section is from https://gist.github.com/cyoon1729/3920da556f992909ace8516e2f321a7c#file-reinforce_update-py
                #this part does the learning
                policyGrad = []
                for logProb, Gt in zip(logProbs, returns):
                    policyGrad.append(-logProb * Gt)
                policy_model.optimizer.zero_grad()
                policyGrad = torch.stack(policyGrad).sum()
                #back propagation
                policyGrad.backward()
                policy_model.optimizer.step()
                break
            state = nextState['glyphs']
    return policy, scores

def run_reinforce():
    env = gym.make('MiniHack-Quest-Hard-v0',observation_keys=("glyphs", "chars", "colors", "pixel","screen_descriptions"),)
    #print(env.observation_space['glyphs'])
    #deimension of game space
    size = 21 * 79
    policy_model = SimplePolicy(s_size=size, h_size=128, a_size=env.action_space.n)
    policy, scores = reinforce(env=env, policy_model=policy_model, seed=42, learning_rate=1e-2,
                               number_episodes=10,
                               max_episode_length=100000,
                               gamma=0.9,
                               verbose=True)
    # Plot learning curve
    #gamma = 1.0
    #number_episodes=1500
    #max_episode_length=1000
    #h_size=50
    plt.plot(scores,'o')
    plt.show()


if __name__ == '__main__':
    run_reinforce()

