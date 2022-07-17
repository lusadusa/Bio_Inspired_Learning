#!/usr/bin/env python
# coding: utf-8

# ## Import libraries:

# In[1]:


pip install gym[box2d]


# In[2]:


import os
import torch
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from utils import test_policy_network, seed_everything, plot_stats, plot_action_probs
from parallel_env import ParallelEnv, ParallelWrapper
import statistics as stats


# ## Create and preprocess the environment

# ### Create the environment

# In[3]:


env = gym.make('LunarLander-v2')


# In[4]:


# retrieve number of states and number of actions

dims = env.observation_space.shape[0] 
actions = env.action_space.n 


# ### Parallelize the environment

# In[6]:


num_envs = 8


# In[7]:


def create_env(env_name):
    env = gym.make(env_name)
    seed_everything(env)
    return env


# In[8]:


env_fns = [lambda: create_env('LunarLander-v2') for _ in range(num_envs)] 

# use the function ParallelEnv to create the parallelized environment
parallel_env = ParallelEnv(env_fns) 


# ### Prepare the environment to work with PyTorch

# In[ ]:


class PreprocessEnv(ParallelWrapper):
    
    def __init__(self, env):
        ParallelWrapper.__init__(self, env)
    
    def reset(self):
        state = self.venv.reset()
        return torch.from_numpy(state).float()
    
    def step_async(self, actions):
        actions = actions.squeeze().numpy()
        self.venv.step_async(actions)
     
    def step_wait(self):
        next_state, reward, done, info = self.venv.step_wait()
        next_state = torch.from_numpy(next_state).float()
        reward = torch.tensor(reward).unsqueeze(1).float()
        done = torch.tensor(done).unsqueeze(1)
        return next_state, reward, done, info


# In[ ]:


parallel_env = PreprocessEnv(parallel_env)


# ### Create the policy $\pi(s)$

# According to the case considered, the policy is adjusted. Therefore, only few example of the used policies are left for conciseness purpose

# In[ ]:


policy_C1 = nn.Sequential(
    nn.Linear(dims, 64),
    nn.ReLU(),
    nn.Linear(64, actions),
    nn.Softmax(dim=-1))


# In[ ]:


policy_A3 = nn.Sequential(
    nn.Linear(dims, 256),
    nn.Tanh(),
    nn.Linear(256, actions),
    nn.Softmax(dim=-1))


# In[ ]:


policy_A4 = nn.Sequential(
    nn.Linear(dims, 512),
    nn.Tanh(),
    nn.Linear(512, actions),
    nn.Softmax(dim=-1))


# In[ ]:


policy_D3 = nn.Sequential(
    nn.Linear(dims,64),
    nn.Tanh(),
    nn.Linear(64,32),
    nn.Tanh(),
    nn.Linear(32, actions),
    nn.Softmax(dim=-1))


# # Create the algorithm

# In[ ]:


def reinforce(policy, episodes, alpha=1e-4, gamma=0.99, beta = 0.01):
    
    # create the object that will update the parameters of the NN. The AdamW class is an improved version of the 
    #     stochastic gradient descent update rule. 
    optim = AdamW(policy.parameters(), lr=alpha)
    
    # declare the dictionary inside which the statistics of the execution of the algorithm are stored
    stats = {'PG Loss': [], 'Returns': []}
    
    # repeat the main loop for every episode, wrapping it with the tqdm function to follow on the screen the execution of the loop
    for episode in tqdm(range(1, episodes + 1)):
        
        # set the parallel environment to receive the initial state of every individual environment
        state = parallel_env.reset()
        
        # boolean column vector that tells whether each of the individual environments has finished the episode or not
        done_b = torch.zeros((num_envs, 1), dtype=torch.bool)
        
        # create a list to store the state transitions
        transitions = []
        
        # used to keep track of the returns of each of the individual episodes
        ep_return = torch.zeros((num_envs, 1))

        # inner loop to generate the trajectory of experience. The loop goes on until all the episodes are finished
        while not done_b.all():
            
            # the policy selects an action for every individual environment
            action = policy(state).multinomial(1).detach()
            
            # execute the action in the environment and observe the outcome
            next_state, reward, done, _ = parallel_env.step(action)
            
            # store the transition in the list of transitions (if the environment is finished, its reward shouldn't be updated.
            #    That's why it's multiplied for ~done_b)
            transitions.append([state, action, ~done_b * reward])
            
            # update the values
            ep_return += reward
            done_b |= done
            state = next_state
        
        # initialize the return obtained for each parallel environment as a column vector of zeros
        G = torch.zeros((num_envs, 1))
        
        # iterate over each moment of time in reverse order
        for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
            
            # update the value of the return
            G = reward_t + gamma * G
            
            # obtain the vector of probabilities for the state of each environment at a certain moment in time
            #    and compute its log. "+ 1e-6" is added to avoid issues with the log in case of probabilities equal to 0.
            probs_t = policy(state_t)
            log_probs_t = torch.log(probs_t + 1e-6)
            
            # choose only the probability of the action acutally taken at time t
            action_log_prob_t = log_probs_t.gather(1, action_t)
            
            # compute the entropy of each vector of probabilities at time t
            entropy_t = - torch.sum(probs_t * log_probs_t, dim=-1, keepdim=True)
            
            # compute the estimate of the policy performance. The minus sign comes from the fact that the AdamW optimizer
            #    can only perform gradient descent, but in the REINFORCE algorithm gradient ascent is performed.
            gamma_t = gamma ** t
            pg_loss_t = - gamma_t * action_log_prob_t * G
            
            # subtract the entropy scaled by beta and compute the mean over every environment that
            #   the agent is facing in parallel
            total_loss_t = (pg_loss_t - beta * entropy_t).mean()
            
            # wipe the gradients of the NN
            policy.zero_grad()
            
            # use the backpropagation algorithm to compute the gradient of the loss function with respect to
            #    each of the NN parameters
            total_loss_t.backward()
            
            # update the NN parameters
            optim.step()
        
        # update the dictionary with the statistics
        stats['PG Loss'].append(total_loss_t.item())
        stats['Returns'].append(ep_return.mean().item())
        

    return stats


# # Results

# ## Create the filtering function 

# In[ ]:


def filter_values(values, window=10):
    weight = np.repeat(1.0, window)/window
    smas = np.convolve(values,weight,'valid')
    return filtered_values


# ## Policy A3

# In[ ]:


# apply the reiforce algorithm with policy A3 for 10000 episodes

A3 = reinforce(policy_A3, 10000)


# In[ ]:


# save the returns and the filtered returns

returns_A3 = A3["Returns"]
filtered_A3 = filter_values(returns_A3)


# In[ ]:


# plot the results

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(returns_A3)
ax.plot(filtered_A3)
plt.axhline(y = 195,color = 'g')
ax.set_ylabel('Returns')
ax.set_xlabel('Episodes')
ax.set_title('Policy A3')
p = stats.mean(returns_A3[1000:-1])
plt.axhline(y = p, color = 'r')
plt.legend(['Returns','Filtered Returns','Target Return','Average from 1000th episode'])


# ## Policy C1

# In[ ]:


C1 = reinforce(policy_C1, 10000)


# In[ ]:


returns_C1 = C1["Returns"]
filtered_C1 = filter_values(returns_C1)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))

ax.plot(returns_C1)
ax.plot(filtered_C1)
plt.axhline(y = 195,color = 'g')
ax.set_ylabel('Returns')
ax.set_xlabel('Episodes')
ax.set_title('Policy C1')
p = stats.mean(returns_C1[1000:-1])
plt.axhline(y = p, color = 'r')
plt.legend(['Returns','Filtered Returns','Target Return','Average from 1000th episode'])


# ## Policy A4

# In[ ]:


A4 = reinforce(policy_A4, 10000)


# In[ ]:


returns_A4 = A4["Returns"]
filtered_A4 = filter_values(returns_A4)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))

ax.plot(returns_A4)
ax.plot(filtered_A4)
plt.axhline(y = 195,color = 'g')
ax.set_ylabel('Returns')
ax.set_xlabel('Episodes')
ax.set_title('Policy A4')
p = stats.mean(returns_A4[1000:-1])
plt.axhline(y = p, color = 'r')
plt.legend(['Returns','Filtered Returns','Target Return','Average from 1000th episode'])


# ## Policy D3

# In[ ]:


D3 = reinforce(policy_D3, 5000)


# In[ ]:


returns_D3 = D3["Returns"]
filtered_D3 = filter_values(returns_D3)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))

ax.plot(returns_D3)
ax.plot(filtered_D3)
plt.axhline(y = 195,color = 'g')
ax.set_ylabel('Returns')
ax.set_xlabel('Episodes')
ax.set_title('Policy D3')
p = stats.mean(returns_D3[1000:-1])
plt.axhline(y = p, color = 'r')
plt.legend(['Returns','Filtered Returns','Target Return','Average from 1000th episode'])


# # Test Network

# Select in the next cell the policy that needs to be tested

# In[ ]:


# select the name of the policy that needs to be tested
policy = policy_A4


# In[ ]:


test_policy_network(env, policy, episodes=1)

