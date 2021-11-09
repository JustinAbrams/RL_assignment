import gym
import numpy as np
from collections import deque
import random

import logging
import sys
import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model

class actor_critic_lstm(nn.Module):
    def __init__(self,
                 action_space,
                 lstm_input_size = 52,
                 lstm_seq = 52,
                 lstm_hidden_size = 128,
                 lstm_num_layers = 1,
                ):
        super(actor_critic_lstm,self).__init__()
        self.action_space = action_space
        self.lstm_input_size = lstm_input_size
        self.lstm_seq = lstm_seq
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.glyph_model = nn.Sequential(
                  nn.Conv2d(1, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(32, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                )

        self.around_agent_model = nn.Sequential(
                  nn.Conv2d(1, 16, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(32, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=3, stride=1,padding = 1),
                  nn.ReLU(),
                )
        
        self.agent_stats_mlp = nn.Sequential(
                  nn.Linear(25,32),
                  nn.ReLU(),
                  nn.Linear(32,32),
                  nn.ReLU(),
                )
        self.lstm = nn.LSTM(self.lstm_input_size,
                            self.lstm_hidden_size,
                            self.lstm_num_layers,
                            batch_first=True) #(Input,Hidden,Num Layers)
        self.mlp_o = nn.Sequential(
                  nn.Linear(111392,2704),
                  nn.ReLU(),
                )
        self.policy = nn.Linear(self.lstm_hidden_size*self.lstm_seq,
                                     self.action_space)
        self.state_value = nn.Linear(self.lstm_hidden_size*self.lstm_seq,1)

    def forward(self, x):
        batch_size = state[0].shape[0]
        x = self.glyph_model(state[0])
        x = torch.reshape(x,(x.size(0),-1))
        
        y = self.around_agent_model(state[1])
        y = torch.reshape(y,(y.size(0),-1))
        
        z = self.agent_stats_mlp(state[2])
        z = torch.reshape(z,(z.size(0),-1))
        
        o = torch.cat((x, y, z), 1)

        o_t = self.mlp_o(o)
 
        #LSTM
        h = o_t.view(batch_size,self.lstm_seq,
                     self.lstm_input_size)
        if dones == None:
            h,(hidden_state,cell_state) = self.lstm(h,(hidden_state,cell_state))
        else:
            output_list = []
            for input_state,nd in zip(h.unbind(),dones.unbind()):
                # Reset core state to zero whenever an episode ends
                # Make done broadcastable with (num_layers,batch,hidden_size)
                nd = nd.view(1,-1,1)
                out,(hidden_state,cell_state) =\
                self.lstm(input_state.unsqueeze(0),(nd*hidden_state,nd*cell_state))
                # h (batch_size,seq_len,hidden_size)
                output_list.append(out)
            h = torch.cat(output_list)# -> (batch_size,seq_len,hidden_size)
               
        #(batch_size,sequence_length,hidden_size) -> (batch,sequence*hidden)
        h=h.view(h.shape[0],-1)

        policy_logits = self.policy(h)
        state_value = self.state_value(h)
        prob = F.softmax(policy_logits,dim=-1)
        dist = Categorical(prob) 
        return dist, value


def compute_returns(rewards, gamma):
     R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def reinforce_learned_baseline(env, policy_model, seed, learning_rate,
                               number_episodes,
                               gamma, verbose=False):
    # set random seeds (for reproducibility)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    env.seed(seed)
    random.seed(seed)

    raise NotImplementedError
    return policy_value_net, scores


def main():
    env = gym.make('LunarLander-v2')
    print('Action space: ', env.action_space)
    print('Observation space: ', env.observation_space)

    # hyper-parameters
    gamma = 0.99
    learning_rate = 0.02
    # seed = 214
    seed = 401
    number_episodes = 1250
    policy_model = PolicyValueNetwork()

    net, scores = reinforce_learned_baseline(env, policy_model, seed, learning_rate,
                                             number_episodes,
                                             gamma, verbose=True)

    state = env.reset()
    for t in range(2000):
        state = torch.from_numpy(state).float().to(device)
        dist, value = net(state)
        action = dist.sample().item()
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            state = env.reset()
    env.close()


if __name__ == '__main__':
    main()
