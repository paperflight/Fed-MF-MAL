import torch
from torch import nn
from torch.nn import functional as F
import GLOBAL_PRARM as gp
import math
import numpy as np


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, inputs):
        if self.training:
            return F.linear(inputs, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(inputs, self.weight_mu, self.bias_mu)


class Actor_Critic(nn.Module):
    def __init__(self, args, action_space):
        super(Actor_Critic, self).__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.archit = args.architecture

        if 'canonical' in args.architecture and '61obv' in args.architecture and '16ap' in args.architecture:
            self.convs = nn.Sequential(nn.Conv2d(args.history_length * gp.OBSERVATION_DIMS, 16, 8, stride=3, padding=2), nn.LeakyReLU(),
                                       nn.Conv2d(16, 32, 4, stride=2, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.Conv2d(32, 32, 3, stride=1, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU())
            self.conv_output_size = 512  # 41: 2: 1600  # 61: 2: 2368 3: 3200 4: 4288  # 4 uav: 4992
        else:
            raise TypeError('No such structure or incorrect structure')
        # TODO: Calculate the output_size carefully!!!

        self.actor_end = nn.Sequential(NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std),
                                       NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std),
                                       nn.Linear(args.hidden_size, action_space), nn.LeakyReLU())
        self.value_end = nn.Sequential(NoisyLinear(self.conv_output_size + self.action_space, args.hidden_size,
                                                   std_init=args.noisy_std), nn.LeakyReLU(),
                                       NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std),
                                       nn.LeakyReLU(),
                                       nn.Linear(args.hidden_size, self.atoms))

    def forward(self, x, actor_or_critic=True, action=None, log=False):
        x = self.convs(x.float()).view(x.size(0), -1)
        if actor_or_critic:  # actor run if ture
            return self.actor_end(x)
        else:  # critic run if false
            v = self.value_end(torch.cat([x, torch.reshape(action, (action.size(0), -1))], 1))
            if log:  # Use log softmax for numerical stability
                v = F.log_softmax(v, dim=-1)  # Log probabilities with action over second dimension
            else:
                v = F.softmax(v, dim=-1)  # Probabilities with action over second dimension
            return v

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name and name != 'fc':
                module.reset_noise()
