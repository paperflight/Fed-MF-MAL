# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
import GLOBAL_PRARM as gp


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, indim, block, layers):
        super(ResNet, self).__init__()
        self.conv = conv3x3(indim, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = 16
        self.conv1 = self.make_layer(block, 16, layers[0])
        self.conv2 = self.make_layer(block, 32, layers[0], 2)
        self.conv3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(2)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.avg_pool(out)
        return out


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


class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.archit = args.architecture

        if 'canonical' in args.architecture and '61obv' in args.architecture and '16ap' in args.architecture:
            self.convs = nn.Sequential(nn.Conv2d(args.history_length * gp.OBSERVATION_DIMS, 16, 8, stride=3, padding=2), nn.LeakyReLU(),
                                       nn.Conv2d(16, 32, 4, stride=2, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.Conv2d(32, 32, 3, stride=1, padding=0), nn.BatchNorm2d(32), nn.LeakyReLU(),
                                       nn.Dropout2d(0.2))
            self.conv_output_size = 512  # 41: 2: 1600  # 61: 2: 2368 3: 3200 4: 4288  # 4 uav: 4992
        elif 'canonical' in args.architecture and 'pooling' in args.architecture and '20ap' in args.architecture:
            self.convs = nn.Sequential(nn.Conv2d(args.history_length * gp.OBSERVATION_DIMS, 16, 47, stride=23,
                                                 groups=2, padding=23),
                                       nn.BatchNorm2d(16), nn.LeakyReLU())
            self.conv_output_size = 144  # 41: 2: 1600
            self.conv_output_size = 64 * 4 * 4
        else:
            raise TypeError('No such strucure')
        # TODO: Calculate the output_size carefully!!!
        # if args.architecture == 'canonical':
        #     self.convs = nn.Sequential(nn.Conv2d(args.state_dims, 32, 3, stride=1, padding=1), nn.ReLU(),
        #                                nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
        #                                nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
        #     self.conv_output_size = 576
        # elif args.architecture == 'data-efficient':
        #     self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 3, stride=1, padding=0), nn.ReLU(),
        #                                nn.Conv2d(32, 64, 3, stride=1, padding=0), nn.ReLU())
        #     self.conv_output_size = 576
        self.actor_end = nn.Sequential(NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std),
                                       NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std),
                                       nn.Linear(args.hidden_size, action_space), nn.LeakyReLU())
        self.value_end = nn.Sequential(NoisyLinear(self.conv_output_size, args.hidden_size,
                                                   std_init=args.noisy_std), nn.LeakyReLU(),
                                       NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std),
                                       nn.LeakyReLU(),
                                       nn.Linear(args.hidden_size, self.atoms))

    def forward(self, x, actor_or_critic=True, log=False):
        x = self.convs(x.float()).view(x.size(0), -1)
        if actor_or_critic:  # actor run if ture
            p = self.actor_end(x)
            if log:  # Use log softmax for numerical stability
                p = F.log_softmax(p, dim=-1)  # Log probabilities with action over second dimension
            else:
                p = F.softmax(p, dim=-1)  # Probabilities with action over second dimension
            return p
        else:  # critic run if false
            v = self.value_end(x)
            if log:  # Use log softmax for numerical stability
                v = F.log_softmax(v, dim=-1)  # Log probabilities with action over second dimension
            else:
                v = F.softmax(v, dim=-1)  # Probabilities with action over second dimension
            return v

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name and name != 'fc':
                module.reset_noise()
