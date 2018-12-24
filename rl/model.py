import logging

import torch
import torch.nn as nn

import numpy as np

logger = logging.getLogger(__name__)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, params):
        """Initialize parameters and build model.
        Params
        ======
            params (dict-lie): dictionary of parameters
        """
        super(Actor, self).__init__()

        logger.debug('Parameter: %s', params)

        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.seed = torch.manual_seed(params['seed'].next())
        self.act_fn = params['act_fn']
        self.batchnorm = params['norm']
        
        hidden_layers = params['hidden_layers']
        self.hidden_layers = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], self.action_size)
        self.dropout = nn.Dropout(p = params['dropout'])

        self.norms = []
        if self.batchnorm:
            self.norm = nn.BatchNorm1d(self.state_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.hidden_layers:
            linear.weight.data.uniform_(*hidden_init(linear))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        if self.batchnorm:
            x = self.norm(state)
        for i, linear in enumerate(self.hidden_layers):
            x = linear(x)
            x = self.act_fn[i](x)
            x = self.dropout(x)
        
        x = self.act_fn[-1](self.output(x))
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, params):
        """Initialize parameters and build model.
        Params
        ======
            params (dict-lie): dictionary of parameters
        """
        super(Critic, self).__init__()

        logger.debug('Parameter: %s', params)

        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.seed = torch.manual_seed(params['seed'].next())
        self.act_fn = params['act_fn']
        self.action_layer = params['action_layer']
        self.batchnorm = params['norm']
        
        hidden_layers = params['hidden_layers'].copy()

        if self.batchnorm:
            self.norm = nn.BatchNorm1d(self.state_size)
        
        self.hidden_layers = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        hidden_layers[self.action_layer-1] += self.action_size
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.dropout = nn.Dropout(p = params['dropout'])

        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.hidden_layers:
            linear.weight.data.uniform_(*hidden_init(linear))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = state
        if self.batchnorm:
            x = self.norm(state)
        for i, linear in enumerate(self.hidden_layers):
            if i == self.action_layer:
                x = torch.cat((x, action), dim=1)
            x = linear(x)
            x = self.act_fn[i](x)
            x = self.dropout(x)
        
        x = self.act_fn[-1](self.output(x))
        return x
