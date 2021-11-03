import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=128, action_dim=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_dim)
        self.l_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        y = self.fc4(x)
        y = F.softmax(y, dim=-1)
        return y

    def sample_action(self, state):
        """
        以一定的概率选择action，用于在训练阶段，增加一定的action选择的灵活度
        """
        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        if len(state.size()) == 3:
            state = state.unsqueeze(0)
        y = self(state)
        dist = Categorical(y)
        action = dist.sample()
        log_probability = dist.log_prob(action)
        return action.item(), log_probability.item()

    def best_action(self, state):
        """
        确定性的选择某一个action
        """
        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        y = self(state)
        action = torch.argmax(y)
        return action.item()

    def evaluete_action(self, state, actions):
        y = self(state)
        dist = Categorical(y)
        entropy = dist.entropy()
        log_probabilities = dist.log_prob(actions)
        return log_probabilities, entropy


class ValueNetwork(nn.Module):
    def __init__(self, state_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        self.l_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        y = self.fc4(x)
        return y.squeeze(0)

    def state_value(self, state):
        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        if len(state.size()) == 3:
            state = state.unsqueeze(1)
        y = self(state)
        return y.item()


def train_value_network(value_model: nn.Module, 
                        value_optimizer: torch.optim.Optimizer,
                        data_loader, epoches=4):
    epoches_losses = []
    for i in range(epoches):
        losses = []
        for observations, _, _, _, rewards_to_go in data_loader:
            observations = observations.float().to(device)
            rewards_to_go = rewards_to_go.float().to(device)

            value_optimizer.zero_grad()
            values = value_model(observations)
            loss = F.mse_loss(values, rewards_to_go)
            loss.backward()
            value_optimizer.step()
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        epoches_losses.append(mean_loss)
    return epoches_losses


def train_policy_network(
    policy_model: PolicyNetwork, 
    policy_optimizer: torch.optim.Optimizer, data_loader, epoches=4, clip=0.2
):
    epoches_losses = []
    c1 = 0.01
    for i in range(epoches):
        losses = []
        for observations, actions, advantages, log_probabilities, _ in data_loader:
            observations = observations.float().to(device)
            actions = actions.long().to(device)
            advantages = advantages.float().to(device)
            old_log_probabilities = log_probabilities.float().to(device)

            policy_optimizer.zero_grad()
            new_log_probabilities, entropy = policy_model.evaluete_action(observations, actions)

            probabilities_ratio = torch.exp(new_log_probabilities - old_log_probabilities)
            clipped_probabilities_ratio = torch.clamp(probabilities_ratio, 1 - clip, 1 + clip)
            surrogate1 = probabilities_ratio * advantages
            surrogate2 = clipped_probabilities_ratio * advantages
            loss = -torch.min(surrogate1, surrogate2).mean() - c1 * entropy.mean()
            loss.backward()
            policy_optimizer.step()
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        epoches_losses.append(mean_loss)
    return epoches_losses
