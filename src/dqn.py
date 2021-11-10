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
            neuraln.Linear(in_features=input_size , out_features=256),
            neuraln.ReLU(),
            neuraln.Linear(in_features=256 , out_features=256),
            neuraln.ReLU(),
            neuraln.Linear(in_features=256 , out_features=128),
            neuraln.ReLU(),
            neuraln.Linear(in_features=128, out_features=n_actions)
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
        model_dir=None,
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
        if (model_dir==None):
            self.batch_size = batch_size
            self.gamma = gamma
            self.memory = replay_buffer
            self.use_double_dqn = use_double_dqn
            self.target_network = DQN(input_size, n_actions).to(device)
            self.policy_network = DQN(input_size, n_actions).to(device)
            self.update_target_network()
            self.target_network.eval()
            self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr=lr)        
            self.device = device
        else:
            self.policy_network = torch.load(model_dir)    
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
#==============================================Events===============================================
#===================================================================================================
import minihack
from minihack import RewardManager
# np.set_printoptions(threshold=np.inf)

class ExploreEvent(minihack.reward_manager.Event):
    def __init__(self, reward: float, repeatable: bool, terminal_required: bool, terminal_sufficient: bool):
        super().__init__(reward, repeatable, terminal_required, terminal_sufficient)

    def check(self, env, previous_observation, action, observation) -> float:
        # blank spots are 32
        # agent is 64
        # agent spawn point is 60
        # pathways are 35
        # obs[1] is the char observation 
        # print("+++++++++++++++++++\nobs = \n++++++++++++++++++++++\n", observation[1])
        current = sum(np.count_nonzero(i == 35) for i in observation[1])
        current += sum(np.count_nonzero(i == 60) for i in observation[1])
        prev = sum(np.count_nonzero(i == 35) for i in previous_observation[1])
        prev += sum(np.count_nonzero(i == 60) for i in previous_observation[1])
        if current > prev:
            return self.reward
        else:
            return 0



#===================================================================================================
#==============================================Training=============================================
#===================================================================================================

import random
import numpy as np

import gym
from nle import nethack
import argparse
import time
import torch

def train(env, agent):
    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    e_rewards = [0.0]
    s = env.reset()['glyphs_crop']
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
        ns = ns['glyphs_crop'].flatten()
        agent.memory.add(s, a, reward, ns, float(done))
        s = ns

        e_rewards[-1] += reward

        if done:
            s = env.reset()['glyphs_crop']
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

        

        if done:
            print("Episode:", str(num_episodes))

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

        



def test(env, agent, random=False):
    s = env.reset()['glyphs_crop']
    s = s.flatten()
    episode_count = 0
    reward_for_episode = 0
    prev_reward = 0
    for t in range(hyper_params["num-steps"]):

        time.sleep(0.02)
        env.render()
        if random:
            a = env.action_space.sample()
        else:
            a = agent.act(s)

        print("Episode:", str(episode_count), "\tTime step:", str(t),  "\tAction Taken:", str(a))
        print("Current Episode Reward:", str(reward_for_episode))
        print("Previous Episode Reward:", str(prev_reward))

        ns, reward, done, info = env.step(a)
        reward_for_episode += reward
        ns = ns['glyphs_crop'].flatten()
        s = ns

        if (reward_for_episode < -200):
            done = True

        if done:
            s = env.reset()['glyphs_crop']
            s = s.flatten()
            episode_count += 1
            prev_reward = reward_for_episode
            reward_for_episode = 0


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

def create_env():
    global hyper_params
    # ACTIONS define the actions allowed by the agent
    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
        nethack.Command.OPEN,   # Not sure if needed
        nethack.Command.PICKUP, 
        nethack.Command.WEAR,   
        nethack.Command.WIELD,  
        nethack.Command.QUAFF,
        nethack.Command.INVOKE,
        nethack.Command.ZAP,
        nethack.Command.SWAP,   # Not sure if needed

        # Might need more? All actions and descriptions found here
        # https://minihack.readthedocs.io/en/latest/getting-started/action_spaces.html
    )
    pixel_obs = "pixel_crop"

    reward_manager = RewardManager()
    reward_manager.add_kill_event("minotaur", reward=1, terminal_required=False)
    strings = list()
    strings.append("The door opens.")
    reward_manager.add_message_event(strings, reward=1, terminal_required=True)

    strings = list()
    strings.append("It's solid stone.")
    reward_manager.add_message_event("It's solid stone.", reward=-0.75, terminal_required=False, repeatable=True)

    reward_manager.add_event(ExploreEvent(0.5, True, True, False))
    
    # Create env with modified actions
    # Probably can limit the observations as well
    env = gym.make(
        hyper_params["env-name"],
        observation_keys=("glyphs_crop", "chars", "colors", "pixel", "message", "blstats", pixel_obs),
        actions=NAVIGATE_ACTIONS,
        reward_lose=-1,
        reward_win=1,
        reward_manager=reward_manager
    )

    env.seed(hyper_params["seed"])
    env = RenderRGB(env, pixel_obs)
    env = gym.wrappers.Monitor(env, "recordings", force=True)

    return env


def create_agent(input_size=81, n_actions=12, testing_only=False, model_dir=""):
    global hyper_params
    if testing_only:
        if (model_dir==""): 
            print("No model dir given")
            exit(1)

        agent = DQNAgent(
            None,
            None,
            None,
            use_double_dqn=None,
            lr=None,
            batch_size=None,
            gamma=None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            model_dir=model_dir
        )
        return agent
    else:
        agent = DQNAgent(
            input_size,
            n_actions,
            replay_buffer,
            use_double_dqn=hyper_params["use-double-dqn"],
            lr=hyper_params['learning-rate'],
            batch_size=hyper_params['batch-size'],
            gamma=hyper_params['discount-factor'],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        return agent


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser(description='MiniHack RL project')

    # Add the arguments
    my_parser.add_argument('-test',
                       type=str,
                       help='test a model')

    my_parser.add_argument('-random',
                       action="store_true",
                       help='test a random model')

    # Execute the parse_args() method
    args = my_parser.parse_args()

    hyper_params = {
        "seed": 42,  # which seed to use
        "env-name": "MiniHack-Quest-Hard-v0",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-3,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 128,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 2,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.3,  # fraction of num-steps
        "print-freq": 25, # number of iterations between each print out
        "save-freq": 100, # number of iterations between each model save
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    # Monitor for gif
    # env = gym.wrappers.Monitor(
    #     env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    # Glyphs are 21 by 79
    # Glyphs_crop centered 9x9 around agent
    input_size = 81 # 9*9

    if args.test:
        env = create_env()
        agent = create_agent(testing_only=True, model_dir=args.test)
        test(env, agent)
    elif args.random:
        env = create_env()
        test(env, None, random=True)
    else:
        env = create_env()
        agent = create_agent(input_size, env.action_space.n)
        train(env, agent)

    


    
