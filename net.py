import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


""" AlphaZero Architecture """


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels, affine=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = x.float()  # Convert x to float32
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_filters, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_filters, affine=False),
        )
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        out += x
        return self.LeakyReLU(out)


class PolicyBlock(nn.Module):
    def __init__(self, state_size, action_dim, in_channels=256, num_filters=32):
        super(PolicyBlock, self).__init__()

        # Define individual layers
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=1, stride=1)
        # self.bn = nn.BatchNorm2d(num_filters, affine=False)  # Uncomment if you want to use BatchNorm
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(state_size * num_filters * 6, action_dim)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)  # Uncomment if using BatchNorm
        x = self.leaky_relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class ValueBlock(nn.Module):
    def __init__(self, state_size, in_channels=256, num_filters=6, num_hidden=256):
        super(ValueBlock, self).__init__()

        # Define individual layers
        self.conv = nn.Conv2d(
            in_channels, num_filters, kernel_size=1, stride=1, padding=0
        )
        # self.bn = nn.BatchNorm2d(num_filters, affine=False)  # Uncomment if you want to use BatchNorm
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(state_size * num_filters * 6, num_hidden)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)  # Uncomment if using BatchNorm
        x = self.leaky_relu1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.leaky_relu2(x)
        x = self.linear2(x)
        return x


class AZResNet(nn.Module):
    def __init__(
        self,
        state_dims: tuple[int, ...],
        action_dims: int,
        res_filters=128,
        res_layers=10,
        head_filters=32,
        **kwargs
    ):
        super(AZResNet, self).__init__()

        self.conv = ConvBlock(1, res_filters)
        self.res_layers = nn.Sequential(
            *[ResBlock(res_filters) for _ in range(res_layers)]
        )

        self.policy_head_0 = PolicyBlock(
            int(np.prod(state_dims[1:])), action_dims, res_filters, head_filters
        )
        self.value_head_0 = ValueBlock(
            int(np.prod(state_dims[1:])), res_filters, head_filters
        )

        self.to(DEVICE)

    def forward(self, x: torch.tensor):
        # batch_size, w, h = x.shape
        # sides = torch.ones((batch_size, 1, w, h), dtype=torch.float32, device=DEVICE)
        # if isinstance(side, int):
        #     sides = torch.ones((batch_size, 1, w, h), dtype=torch.float32, device=DEVICE)
        # else:
        #     sides = side.view(-1, 1, 1, 1).repeat(1, 1, w, h)

        # x = torch.cat((x, sides), dim=1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.res_layers(x)

        value = self.value_head_0(x)  # Get the value from the value head
        action_probs = F.softmax(
            self.policy_head_0(x), dim=-1
        )  # Get the action probabilities
        dist = Categorical(
            action_probs
        )  # Create a distribution for calculating log probabilities
        action = dist.sample()

        # Calculate the log probability of the sampled action
        log_prob = dist.log_prob(action)

        return action, value, log_prob

    def sample_actions(self, env):
        states = torch.from_numpy(env.obs()).unsqueeze(0).to(DEVICE)
        masks = torch.from_numpy(env.get_masks()).float().to(DEVICE)

        masks = masks.unsqueeze(0)
        with torch.no_grad():
            _, policy = self(states)
            return (masks * policy + (1 - masks) * -(1e9)).argmax(dim=1).unsqueeze(1)


from stable_baselines3.common.policies import ActorCriticPolicy


class CustomAZPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomAZPolicy, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )
        self.az_resnet = AZResNet(
            state_dims=observation_space.shape,
            action_dims=action_space.n,
            **kwargs  # Any additional parameters
        )

    def forward(self, obs, deterministic=False):
        # Process observations through your network
        action, value, log_prob = self.az_resnet(obs)
        return action, value, log_prob


class AZDQNResNet(nn.Module):
    def __init__(
        self,
        state_dims: tuple[int, ...],
        action_dims: int,
        res_filters=128,
        res_layers=10,
        head_filters=32,
        **kwargs
    ):
        super(AZDQNResNet, self).__init__()

        self.conv = ConvBlock(1, res_filters)
        self.res_layers = nn.Sequential(
            *[ResBlock(res_filters) for _ in range(res_layers)]
        )
        self.value_head_0 = ValueBlock(
            int(np.prod(state_dims[1:])), res_filters, head_filters
        )

        self.to(DEVICE)

    def forward(self, x: torch.tensor):
        # batch_size, w, h = x.shape
        # sides = torch.ones((batch_size, 1, w, h), dtype=torch.float32, device=DEVICE)
        # if isinstance(side, int):
        #     sides = torch.ones((batch_size, 1, w, h), dtype=torch.float32, device=DEVICE)
        # else:
        #     sides = side.view(-1, 1, 1, 1).repeat(1, 1, w, h)

        # x = torch.cat((x, sides), dim=1)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.res_layers(x)

        value = self.value_head_0(x)  # Get the value from the value head
        return value


class CustomAZDQNPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomAZDQNPolicy, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )
        self.q_net = AZDQNResNet(
            state_dims=observation_space.shape,
            action_dims=action_space.n,
            **kwargs  # Any additional parameters
        )
        self.q_net_target = AZDQNResNet(
            state_dims=observation_space.shape, action_dims=action_space.n, **kwargs
        )

    def forward(self, obs, deterministic=False):
        # Process observations through your network
        value = self.q_net(obs)
        return value
