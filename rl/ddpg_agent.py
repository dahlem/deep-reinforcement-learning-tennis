# coding: utf-8
from abc import ABCMeta, abstractmethod

import random
import copy

import logging

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from . model import Actor, Critic

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""
    __metaclass__ = ABCMeta
    
    def __init__(self, params):
        """Initialize an Agent object given a dictionary of parameters.
        
        Params
        ======
        * **params** (dict-like) --- a dictionary of parameters
        """
        logger.debug('Parameter: %s', params)

        self.params = params
        self.tau = params['tau']
        
        # Replay memory: to be defined in derived classes
        self.memory = None
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    @abstractmethod
    def act(self, state, action):
        """Returns actions for given state as per current policy.
        
        Params
        ======
        * **state** (array_like) --- current state
        * **action** (array_like) --- the action values
        """
        pass


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
        * **local_model** (PyTorch model) --- weights will be copied from
        * **target_model** (PyTorch model) --- weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    @abstractmethod
    def step(self, states, actions, rewards, next_states, dones):
        """Perform a step in the environment given a state, action, reward,
        next state, and done experience.

        Params
        ======
        * **states** (torch.Variable) --- the current state
        * **actions** (torch.Variable) --- the current action
        * **rewards** (torch.Variable) --- the current reward
        * **next_states** (torch.Variable) --- the next state
        * **dones** (torch.Variable) --- the done indicator
        * **betas** (float) --- a potentially tempered beta value for prioritzed replay sampling

        """
        pass

    @abstractmethod
    def learn_(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
        * **experiences** (Tuple[torch.Variable]) --- tuple of (s, a, r, s', done) tuples 
        """
        pass

class DDPGAgent(Agent):
    """Interacts with and learns from the environment."""
    
    def __init__(self, idx, params):
        """Initialize an Agent object.
        
        Params
        ======
            params (dict-like): dictionary of parameters for the agent
        """
        super().__init__(params)

        self.idx = idx
        self.params = params
        self.update_every = params['update_every']
        self.gamma = params['gamma']
        self.num_agents = params['num_agents']
        
        # Actor Network (w/ Target Network)
        if params['actor_local'] != None:
            self.actor_local = params['actor_local']
        else:
            self.actor_local = Actor(params['actor_params']).to(device)

        if params['actor_target'] != None:
            self.actor_target = params['actor_target']
        else:
            self.actor_target = Actor(params['actor_params']).to(device)

        if params['actor_optimizer'] != None:
            self.actor_optimizer = params['actor_optimizer']
        else:
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=params['actor_params']['lr'])
        
        # Critic Network (w/ Target Network)
        if params['critic_local'] != None:
            self.critic_local = params['critic_local']
        else:
            self.critic_local = Critic(params['critic_params']).to(device)

        if params['critic_target'] != None:
            self.critic_target = params['critic_target']
        else:
            self.critic_target = Critic(params['critic_params']).to(device)

        if params['critic_optimizer'] != None:
            self.critic_optimizer = params['critic_optimizer']
        else:
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                               lr=params['critic_params']['lr'],
                                               weight_decay=params['critic_params']['weight_decay'])

        # Noise process
        self.noise = OUNoise(self.params['noise_params'])

        # Replay memory
        self.memory = params['experience_replay']
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        next_state = torch.from_numpy(next_states[self.idx]).float().unsqueeze(0).to(device)
        state = torch.from_numpy(states[self.idx]).float().unsqueeze(0).to(device)
        
        # Save experience / reward
        self.memory.add(state, actions[self.idx], rewards[self.idx], next_state, dones[self.idx])

        # Learn, if enough samples are available in memory
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.ready():
                for i in range(self.num_agents):
                    experiences = self.memory.sample()
                    self.learn_(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1., 1.)

    def reset(self):
        self.noise.reset()

    def learn_(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)                     

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, params):
        """Initialize parameters and noise process."""

        mu = params['mu']
        theta = params['theta']
        sigma = params['sigma']
        seed = params['seed']
        size = params['action_size']
        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed.next())
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
