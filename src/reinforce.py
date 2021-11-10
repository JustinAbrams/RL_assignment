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
        self.loss_fn=nn.CrossEntropyLoss()
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
    #env.render()
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
        state = env.reset()['glyphs_crop']
        lProbs = []
        scores = []
        states = []
        for steps in range(max_episode_length):
            #env.render()
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
                numsteps.append(steps)
                avgNumsteps.append(np.mean(numsteps[-10:]))
                allRewards.append(np.sum(scores))
                if episode % 1 == 0:
                    print("Reinforce with baseline -> episode: {}, total reward: {}, average_reward: {}, length: {}".format(episode,np.round(
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
            state = nextState['glyphs_crop']
    env.close()
    return policy, allRewards
import cv2
cv2.ocl.setUseOpenCL(False)


class RenderRGB(gym.Wrapper):
    def __init__(self, env, key_name="pixel"):
        super().__init__(env)
        self.last_pixels = None
        self.viewer = None
        self.key_name = key_name

        render_modes = env.metadata['render.modes']
        render_modes.append("rgb_array")
        env.metadata['render.modes'] = render_modes

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_pixels = obs[self.key_name]
        return obs, reward, done, info

    def render(self, mode="human", **kwargs):
        img = self.last_pixels

        # Hacky but works
        if mode != "human":
            return img
        else:
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def reset(self):
        obs = self.env.reset()
        self.last_pixels = obs[self.key_name]
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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
    reward_gen.add_kill_event("minotaur", reward=10)
    reward_gen.add_kill_event("goblin", reward=1)
    reward_gen.add_kill_event("jackal", reward=1)
    reward_gen.add_kill_event("giant rat", reward=1)
    strings = list()
    strings.append("The door opens.")
    reward_gen.add_message_event(strings, reward=1)
    # Create env with modified actions
    # Probably can limit the observations as well
    pixel_obs = "pixel_crop"
    env = gym.make(
        hyper_params["env-name"],
        observation_keys=("glyphs", "chars", "colors", "pixel","screen_descriptions", "pixel_crop","glyphs_crop"),
        actions=NAVIGATE_ACTIONS,
        reward_lose=-10,
        reward_win=10,
        penalty_step = -0.002,
        penalty_time = 0.002,
        reward_manager=reward_gen,
        max_episode_steps = hyper_params['num-steps']
    )
    env.seed(hyper_params["seed"])
    env = RenderRGB(env, pixel_obs)
    env = gym.wrappers.Monitor(env, "recordings", force=True)
    return env

def run_reinforce():
    global hyper_params
    env = makeEnv()
    print("number of actions: ",env.action_space)
    #print(env.observation_space['glyphs'])
    #deimension of game space
    size = 9 * 9
    hSize = round(size/2)
    num_epi = 10
    policy_model = SimplePolicy(s_size=size, h_size=size, a_size=env.action_space.n,learning_rate=hyper_params['learning-rate']).to(device)
    stateval_model = StateValueNetwork(s_size=size, h_size=size,learning_rate=hyper_params['learning-rate']).to(device)
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
        "num-steps": int(5000),  # total number of steps to run the environment for
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
