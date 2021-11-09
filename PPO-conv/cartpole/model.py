import torch
import torch.nn as nn
import numpy as np
from config import device
from torch.distributions import Categorical
from torchsummary import summary


class PolicyNetwork(nn.Module):
    def __init__(self,feature_shape:np.array,action_dim):
        '''
        feature_shape: [channel, heigh, width]
        '''
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(feature_shape[0],64, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64,64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64,32,kernel_size=3,stride=2),
            nn.Flatten(),
            nn.Linear(14880, 128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.LeakyReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax()
        )
    
    def forward(self, x):
        if not x is torch.Tensor:
            x = torch.Tensor(x).float().to(device)
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        return self.network(x)

    def sample_action(self, x):
        action = self(x)
        dist = Categorical(action)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def best_action(self, x):
        action = self(x)
        action = torch.argmax(action)
        return action.item()
    
    
class ValueNetwork(nn.Module):
    def __init__(self, feature_shape):
        '''
        feature_dim: shape is: [channel, heigh, width]
        '''
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(feature_shape[0],64, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64,64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64,32,kernel_size=3,stride=2),
            nn.Flatten(),
            nn.Linear(14880, 128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self(x).squeeze(0)
    
    def state_value(self, x):
        if not x is torch.Tensor:
            state = torch.from_numpy(x).float().to(device)
        if len(state.size()) == 3:
            state = state.unsqueeze(1)
        y = self(state)
        return y.item()
        
def test_policyNetwork():
    policy = PolicyNetwork((3,128,256),5)
    summary(policy, (3,128,256))
    x = np.random.randn(3, 128,256)
    y = policy.sample_action(x)
    print(y)


    
if __name__ == '__main__':
    test_policyNetwork()


    