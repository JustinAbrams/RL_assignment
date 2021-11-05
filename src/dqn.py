#===================================================================================================
#=================================================Model=============================================
#===================================================================================================

from gym import spaces
import torch.nn as neuraln


class DQN(neuraln.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, input_size, n_actions):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()

        self.fc = neuraln.Sequential(
            neuraln.Linear(in_features=input_size , out_features=1024),
            neuraln.ReLU(),
            neuraln.Linear(in_features=1024 , out_features=512),
            neuraln.ReLU(),
            neuraln.Linear(in_features=512 , out_features=512),
            neuraln.ReLU(),
            neuraln.Linear(in_features=512, out_features=n_actions)
        )

    def forward(self, x):
        return self.fc(x)


import numpy as np


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)
        return self._encode_sample(indices)


#===================================================================================================
#=================================================AGENT=============================================
#===================================================================================================

from gym import spaces
import numpy as np
import torch.nn.functional as F
import torch

device = "cuda"


class DQNAgent:
    def __init__(
        self,
        input_size,
        n_actions,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
        device=torch.device("cuda"),
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory = replay_buffer
        self.use_double_dqn = use_double_dqn
        self.target_network = DQN(input_size, n_actions).to(device)
        self.policy_network = DQN(input_size, n_actions).to(device)
        self.update_target_network()
        self.target_network.eval()
        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters(), lr=lr)        
        self.device = device

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        device = self.device
        s, a, r, ns, dones = self.memory.sample(self.batch_size)
        s = np.array(s)
        ns = np.array(ns)
        s = torch.from_numpy(s).float().to(device)
        r = torch.from_numpy(r).float().to(device)
        a = torch.from_numpy(a).long().to(device)
        ns = torch.from_numpy(ns).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        with torch.no_grad():
            if self.use_double_dqn:
                _, a_max = self.policy_network(ns).max(1)
                max_q = self.target_network(ns).gather(1, a_max.unsqueeze(1)).squeeze()
            else:
                next_q_values = self.target_network(ns)
                max_q, _ = next_q_values.max(1)
            t_qvals = r + (1 - dones) * self.gamma * max_q

        i_qvals = self.policy_network(s)
        i_qvals = i_qvals.gather(1, a.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(i_qvals, t_qvals)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del s
        del ns
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save(self, file_name):
        torch.save(self.policy_network, file_name)

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        device = self.device
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_network(state)
            _, action = q_values.max(1)
            return action.item()

#===================================================================================================
#=================================================Train=============================================
#===================================================================================================

import random
import numpy as np
import gym
import minihack
from nle import nethack
import argparse
import torch

if __name__ == "__main__":

    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e7),  # total number of steps to run the environment for
        "batch-size": 256,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 25,
        "save-freq": 500,
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    # ACTIONS
    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
        nethack.Command.OPEN,
        nethack.Command.KICK,
        nethack.Command.SEARCH,
        nethack.Command.JUMP,
        nethack.Command.LOOK,
        nethack.Command.LOOT,
        nethack.Command.PICKUP,
        nethack.Command.PRAY,
        nethack.Command.WEAR,
        nethack.Command.WIELD,
        nethack.Command.UNTRAP,
    )

    env = gym.make(
        "MiniHack-Quest-Easy-v0",
        actions=NAVIGATE_ACTIONS
    )
    env.seed(hyper_params["seed"])

    # Monitor for gif
    # env = gym.wrappers.Monitor(
    #     env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    input_size = 1659 # 21 * 79

    agent = DQNAgent(
        input_size,
        env.action_space.n,
        replay_buffer,
        use_double_dqn=hyper_params["use-double-dqn"],
        lr=hyper_params['learning-rate'],
        batch_size=hyper_params['batch-size'],
        gamma=hyper_params['discount-factor'],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    e_rewards = [0.0]
    s = env.reset()['glyphs']
    s = s.flatten()
    
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        sample = random.random()
        if(sample > eps_threshold):
            a = agent.act(s)
        else:
            a = env.action_space.sample()

        ns, reward, done, info = env.step(a)
        ns = ns['glyphs'].flatten()
        agent.memory.add(s, a, reward, ns, float(done))
        s = ns

        e_rewards[-1] += reward
        if done:
            s = env.reset()['glyphs']
            s = s.flatten()
            e_rewards.append(0.0)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            agent.optimise_td_loss()

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()

        num_episodes = len(e_rewards)

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(e_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(e_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")

        if (
            done
            and hyper_params["save-freq"] is not None
            and len(e_rewards) % hyper_params["save-freq"] == 0
        ):
            agent.save("model" + str(len(e_rewards)))
