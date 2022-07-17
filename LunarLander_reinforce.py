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


dims = env.observation_space.shape[0] #retrieves the number of states
actions = env.action_space.n #retrieves the number of actions

print(f"State dimensions: {dims}. Actions: {actions}") #prints the number of states and actions
print(f"Sample state: {env.reset()}") #prints a random state


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
parallel_env = ParallelEnv(env_fns) #uses the function ParallelEnv to create the parallelized environment


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


def reinforce(policy, episodes, alpha=1e-4, gamma=0.99):
    optim = AdamW(policy.parameters(), lr=alpha)
    stats = {'PG Loss': [], 'Returns': []}
    
    for episode in tqdm(range(1, episodes + 1)):
        state = parallel_env.reset()
        done_b = torch.zeros((num_envs, 1), dtype=torch.bool)
        transitions = []
        ep_return = torch.zeros((num_envs, 1))

        while not done_b.all():
            action = policy(state).multinomial(1).detach()
            next_state, reward, done, _ = parallel_env.step(action)
            transitions.append([state, action, ~done_b * reward])
            ep_return += reward
            done_b |= done
            state = next_state
        
        G = torch.zeros((num_envs, 1))
        for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
            G = reward_t + gamma * G
            probs_t = policy(state_t)
            log_probs_t = torch.log(probs_t + 1e-6)
            action_log_prob_t = log_probs_t.gather(1, action_t)

            entropy_t = - torch.sum(probs_t * log_probs_t, dim=-1, keepdim=True)
            gamma_t = gamma ** t
            pg_loss_t = - gamma_t * action_log_prob_t * G
            total_loss_t = (pg_loss_t - 0.01 * entropy_t).mean()
            
            policy.zero_grad()
            total_loss_t.backward()
            optim.step()

        stats['PG Loss'].append(total_loss_t.item())
        stats['Returns'].append(ep_return.mean().item())
        

    return stats


# # Results

# ## Create the filtering function 

# In[ ]:


def m_a(values, window=10):
    weight = np.repeat(1.0, window)/window
    smas = np.convolve(values,weight,'valid')
    return smas


# ## Policy A3

# In[ ]:


A3 = reinforce(policy_A3, 10000)


# In[ ]:


returns_A3 = A3["Returns"]
smas_A3 = m_a(returns_A3)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))

ax.plot(returns_A3)
ax.plot(smas_A3)
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
smas_C1 = m_a(returns_C1)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))

ax.plot(returns_C1)
ax.plot(smas_C1)
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
smas_A4 = m_a(returns_A4)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))

ax.plot(returns_A4)
ax.plot(smas_A4)
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
smas_D3 = m_a(returns_D3)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,5))

ax.plot(returns_D3)
ax.plot(smas_D3)
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


policy = policy_A4


# In[ ]:


test_policy_network(env, policy, episodes=1)

