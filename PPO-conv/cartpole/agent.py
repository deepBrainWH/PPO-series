from model import PolicyNetwork, ValueNetwork
import gym
import numpy as np
import cv2

class Agent(object):
    def __init__(self) -> None:
        super().__init__()
        self.env = gym.make("CartPole-v1")
        self.action_dim = self.env.action_space.n
        self.observation_dim = self.env.reset()
        # observation = self.env.render(mode="rgb_array")
        # observation = observation.astype('uint8')
        # observation = cv2.resize(observation,(256,128))
        # cv2.imshow('image', observation)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # self.env.close()
        
        
def test_agent():
    agent = Agent()

if __name__ == '__main__':
    test_agent()