import gym
import gym.wrappers
import gym.spaces
import numpy as np
from collections import deque
import random

import time
import pickle
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from tensorflow import keras
import tensorflow as tf 
from keras.layers import Flatten, Dense
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from multiprocessing import Pool, freeze_support



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
def play_game(iters, r = True):
  env = gym.make('LunarLander-v2')
  totalrewardarray = []
  render = True
  save_gif = False
    
  for i in range(iters):
    state = env.reset()
    totalreward = 0
    step = 0
    done = False

    while not done and step <1000:
      step += 1
      
      if r:
        screen = env.render(mode='rgb_array')
        plt.imshow(screen)
       
        ipythondisplay.display(plt.gcf())

        ipythondisplay.clear_output(wait=True)
        
        import PIL
        PIL.Image.fromarray(env.render(mode='rgb_array')).resize((320, 420))
        if render:
          env.render()
          if save_gif:
            img = env.render(mode = 'rgb_array')
            img = Image.fromarray(img)
            img.save('./gif/{}.jpg'.format(t))

    action = decide_action(actor, state)
    observation, reward, done, _ = env.step(action)  
    totalreward += reward
    state_new = observation 
    state = state_new
    totalrewardarray.append(totalreward)

  return totalrewardarray

#Training plot

def run_train_plot(alr, clr, gamma, numepisodes):
   env = gym.make('LunarLander-v2')
   render = True
   save_gif = False
   i = 0
   
   actor = actor_model(num_input_nodes = 8, num_output_nodes = 4, lr = alr, size = [64,64,64])
   critic = critic_model(num_input_nodes = 8, num_output_nodes = 1, lr= clr, size = [64,64,64])
   
   tot_reward = [] 
   best_score = float('-inf')
   episodes = len(tot_reward)
   
   while episodes < numepisodes:
     i+= 1
     memory, episode_reward = run_episode(env, actor, r = False)
     tot_reward.append(episode_reward)
     episodes = len(tot_reward)

    #  screen = env.render(mode='rgb_array')
    #  plt.imshow(screen)
    #  ipythondisplay.clear_output(wait=True)
    #  ipythondisplay.display(plt.gcf())
     
     if episodes >= 100:
       score = np.average(tot_reward[-100:-1])
       if score > best_score:
         best_score = score
         actor.save('actormodel.h5')
         critic.save('criticmodel.h5')
      
       if episodes%50==0:
         print('ALR:', alr, ' CLR:', clr, 'episode ', episodes, 'of',numepisodes, 'Average Reward (last 100 eps)= ', score)
        
      #  import PIL
      #  PIL.Image.fromarray(env.render(mode='rgb_array')).resize((320, 420))
      #  if render:
      #    env.render()
      #    if save_gif:
      #      img = env.render(mode = 'rgb_array')
      #      img = Image.fromarray(img)
      #      img.save('./gif/{}.jpg'.format(t))
       screen = env.render(mode='rgb_array')
       plt.imshow(screen)
       
       ipythondisplay.display(plt.gcf())

     ipythondisplay.clear_output(wait=True)

     train_models(actor, critic, memory, gamma)

     avgarray = []
     countarray = []
      
   for i in range(100,len(tot_reward),10):
     avgarray.append(np.average(tot_reward[i-100:i]))
     countarray.append(i)

   plt.plot(countarray, avgarray, label = 'Best 100 ep av. reward = '+str(best_score))
   plt.title('Rolling Average (previous 100) vs Iterations')
   plt.xlabel('Iterations')
   plt.ylabel('Reward')
   plt.legend(loc='best')
    
   plt.show()
