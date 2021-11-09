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


def critic_model(num_input_nodes, num_output_nodes, lr = 0.001, size = [256]):
	model = Sequential()
    model.add(Dense(size[0], input_shape = (8,), activation = 'relu'))
	
	for i in range(1,len(size)):
		model.add(Dense(size[i], activation = 'relu'))
	
	model.add(Dense(num_output_nodes, activation = 'linear')) 
	adam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
	model.compile(loss = 'mse', optimizer = adam)
	
	return model

def actor_model(num_input_nodes, num_output_nodes, lr = 0.001, size = [256]):
	model = Sequential()
	model.add(Dense(size[0], input_shape = (num_input_nodes,), activation = 'relu'))
	
	for i in range(1, len(size)):
		model.add(Dense(size[i], activation = 'relu'))
	
	model.add(Dense(num_output_nodes, activation = 'softmax')) 
	adam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999)
	model.compile(loss = 'categorical_crossentropy', optimizer = adam)
	
	return model

def decide_action(actor, state):
	flat_state = np.reshape(state, [1,8])
	action = np.random.choice(4, 1, p = actor.predict(flat_state)[0])[0]	
	return(action)  

#Run episodes
def run_episode(env, actor, r = False):
  memory = []
  state = env.reset()
  episode_reward = 0
  step = 0 
  done = False

  while not done and step <1000:
    step += 1
    if r:
      env.render()
      
    action = decide_action(actor, state)
    observation, reward, done, _ = env.step(action)  
    episode_reward += reward
    state_new = observation 
    memory.append((state, action, reward, state_new, done))
    state = state_new 

  return(memory, episode_reward)

#train model
def train_models(actor, critic, memory, gamma):
	random.shuffle(memory)
	
	for i in range(len(memory)):
		state, action, reward, state_new, done = memory[i]	
		flat_state_new = np.reshape(state_new, [1,8])
		flat_state = np.reshape(state, [1,8])
		target = np.zeros((1, 1))
		advantages = np.zeros((1, 4))

		value = critic.predict(flat_state)
		next_value = critic.predict(flat_state_new)

		# screen = env.render(mode='rgb_array')
		# plt.imshow(screen)
		# ipythondisplay.clear_output(wait=True)
		# ipythondisplay.display(plt.gcf())

		if done:
			advantages[0][action] = reward - value
			target[0][0] = reward
		else:
			advantages[0][action] = reward + gamma * (next_value) - value
			target[0][0] = reward + gamma * next_value
		
		actor.fit(flat_state, advantages, epochs=1, verbose=0)
		critic.fit(flat_state, target, epochs=1, verbose=0)
        
  #play game
def train_models(actor, critic, memory, gamma):
	random.shuffle(memory)
	
	for i in range(len(memory)):
		state, action, reward, state_new, done = memory[i]	
		flat_state_new = np.reshape(state_new, [1,8])
		flat_state = np.reshape(state, [1,8])
		target = np.zeros((1, 1))
		advantages = np.zeros((1, 4))

