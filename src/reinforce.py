
import sys
import math
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
from nle import nethack
from minihack import RewardManager
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimplePolicy(nn.Module):
    def __init__(self, s_size, h_size, a_size, learning_rate=0.001):
        super(SimplePolicy, self).__init__()
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


class StateValueNetwork(nn.Module):
    # Takes in state
    def __init__(self, s_size=4, h_size=16, learning_rate=0.001):
        super(StateValueNetwork, self).__init__()
        self.linear1 = nn.Linear(s_size, h_size)
        self.linear2 = nn.Linear(h_size, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        #input layer
        x = torch.flatten(x)
        x = torch.reshape(x, (1,x.shape[0]))
        f = self.linear1(x)
        #activiation relu
        f = Func.relu(f)
        #get state value
        state_value = self.linear2(f)
        return state_value

def compute_returns_naive_baseline(rewards, gamma):
    returns = []
    #calculates the return values
    for t in range(len(rewards)):
        Gt = 0
        for r in rewards[t:]:
            Gt = Gt * gamma + r
        returns.append(Gt)
    returns = torch.tensor(returns).to(device)
    returns = (returns - returns.mean()) / (
        returns.std())
    return returns

def learning(states ,scores, state_model, policy_model, lProbs, env, gamma):
    returns = compute_returns_naive_baseline(scores, gamma)
    env.render()
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
    deltas = torch.tensor(deltas).to(device)
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

def reinforce_naive_baseline(env, policy_model, state_model, seed,
                             number_episodes,
                             max_episode_length,
                             gamma, verbose=True):
    global hyper_params
    # set random seeds (for reproducibility)
    torch.manual_seed(hyper_params['seed'])
    torch.cuda.manual_seed_all(hyper_params['seed'])
    np.random.seed(hyper_params['seed'])
    random.seed(hyper_params['seed'])
    env.seed(hyper_params['seed'])
    policy = []
    numsteps = []
    avgNumsteps = []
    allRewards = []
    for episode in range(number_episodes):
        state = env.reset()['glyphs']
        lProbs = []
        scores = []
        states = []
        for steps in range(max_episode_length):
            #This displays the last 1 episodes
            #if episode > number_episodes - 2:
            env.render()
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            #this section is from https://gist.github.com/cyoon1729/bc41d466b868ea10e794a7c04321ff3b#file-reinforce_model-py
            probs = policy_model.forward(Variable(state))
            action = np.random.choice(env.action_space.n, p=np.squeeze(probs.detach().numpy()))
            lprob = torch.log(probs.squeeze(0)[action])
            nextState, score, done, _ = env.step(action)
            lProbs.append(lprob)
            scores.append(score)
            states.append(state)
            if steps%100==0:
                p = 0
                #learning(states, scores, state_model, policy_model, lProbs, env, gamma)
            if done:
                learning(states, scores, state_model, policy_model, lProbs, env, gamma)
                allRewards.append(np.sum(scores))
                break
            state = nextState['glyphs']
    env.close()
    return policy, allRewards

def makeEnv():
    global hyper_params
    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
        nethack.Command.OPEN,
        nethack.Command.SEARCH,
        nethack.Command.LOOK, 
        nethack.Command.JUMP, 
        nethack.Command.PICKUP,
        nethack.Command.WIELD, 
        nethack.Command.SWAP,
        nethack.Command.EAT,
        nethack.Command.ZAP,
        nethack.Command.LOOT,
        nethack.Command.PUTON,
        nethack.Command.APPLY,
        nethack.Command.CAST,
        nethack.Command.DIP,
        nethack.Command.READ,
        nethack.Command.INVOKE,
        nethack.Command.RUSH,
        nethack.Command.WEAR,
        nethack.Command.ENHANCE

        # Might need more? All actions and descriptions found here
        # https://minihack.readthedocs.io/en/latest/getting-started/action_spaces.html
    )
    reward_gen = RewardManager()
    reward_gen.add_kill_event("minotaur", reward=100)
    reward_gen.add_kill_event("goblin", reward=10)
    reward_gen.add_kill_event("jackal", reward=10)
    reward_gen.add_kill_event("giant rat", reward=10)
    #reward_gen.add_wield_event("wand", reward=2)
    strings = list()
    strings.append("The door opens.")
    reward_gen.add_message_event(strings, reward=2)
    # Create env with modified actions
    # Probably can limit the observations as well
    env = gym.make(
        hyper_params["env-name"],
        observation_keys=("glyphs", "chars", "colors", "pixel","screen_descriptions"),
        actions=NAVIGATE_ACTIONS,
        reward_lose=-2,
        reward_win=1000,
        penalty_step = -5,
        penalty_time = 2,
        reward_manager=reward_gen,
        max_episode_steps = 10000
    )
    env.seed(hyper_params["seed"])
    return env

def run_reinforce():
    global hyper_params
    env = makeEnv()
    print("number of actions: ",env.action_space)
    #print(env.observation_space['glyphs'])
    #deimension of game space
    size = 21 * 79
    hSize = round(size/2)
    num_epi = 200
    policy_model = SimplePolicy(s_size=size, h_size=hSize, a_size=env.action_space.n,learning_rate=hyper_params['learning-rate']).to(device)
    stateval_model = StateValueNetwork(s_size=size, h_size=hSize,learning_rate=hyper_params['learning-rate']).to(device)
    policy, scores = reinforce_naive_baseline(env=env, policy_model=policy_model, state_model=stateval_model, seed=42,
                               number_episodes=num_epi,
                               max_episode_length=hyper_params['num-steps'],
                               gamma=hyper_params['discount-factor'],
                               verbose=True)
    # Plot learning curve
    plt.plot(scores,'o')
    plt.xlabel('Episodes')
    plt.ylabel('Average reward')
    plt.title('Average reward per episode')
    plt.show()


if __name__ == '__main__':
    hyper_params = {
        "seed": 42,  # which seed to use
        "env-name": "MiniHack-Quest-Hard-v0",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1000),  # total number of steps to run the environment for
        "batch-size": 256,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 2,  # number of iterations between every optimization step
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.1,  # e-greedy end threshold
        "eps-fraction": 0.2,  # fraction of num-steps
        "print-freq": 25, # number of iterations between each print out
        "save-freq": 500, # number of iterations between each model save
    }
    run_reinforce()

