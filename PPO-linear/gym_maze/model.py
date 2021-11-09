import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PolicyNetwork(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64,64),
            nn.LeakyReLU(),
            nn.Linear(64,64),
            nn.LeakyReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        ).to(device)
    
    def forward(self, x:np.array):
        if len(x.shape) == 1:
            x = torch.tensor(x).float().unsqueeze(0).to(device)
        else:
            x =torch.tensor(x).float().to(device)
        return self.network(x)

    def sample_action(self, x:np.array):
        action = self(x)
        dist = Categorical(action)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(),action_logprob.item()
    
    def best_action(self, x:np.array):
        action = self(x)
        dist = Categorical(action)
        action = dist.sample()
        return torch.argmax(action).item()
    
    def evaluate_action(self, state:np.array, action:np.array):
        y = self(state)
        dist = Categorical(y)
        entropy = dist.entropy()
        log_probabilities = dist.log_prob(action)
        return log_probabilities,entropy


class ValueNetwork(nn.Module):
    def __init__(self,state_dim):
        super(ValueNetwork,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64,64),
            nn.LeakyReLU(),
            nn.Linear(64,64),
            nn.LeakyReLU(),
            nn.Linear(64,1)
        )
    def forward(self, x:np.array):
        if len(x.shape) == 1:
            x = torch.tensor(x).float().unsqueeze(0).to(device)
        else:
            x =torch.tensor(x).float().to(device)
        return self.network(x)

    def state_value(self,state):
        return self(state).item()


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
            new_log_probabilities, entropy = policy_model.evaluate_action(observations, actions)

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
