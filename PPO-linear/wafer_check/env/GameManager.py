import numpy as np
from env.Check_Envs_GA3C_v5 import wafer_check
from env.Config import Config

wafer = np.loadtxt('./env/envs.txt')
probe = np.loadtxt('./env/probe.txt')
location = (0, 0)

class GameManager:
    def __init__(self, display):
        self.env = wafer_check(wafer,probe,mode=display,training_time=Config.TRAINING_TIME,training_steps=Config.TRAINING_STEPS)
        self.reset()

    def reset(self):
        observation, available = self.env.reset()
        return observation, available

    def step(self, action):
        observation, reward, done, available, envs_mean, envs_std = self.env.step(action)
        return observation, reward, done, available, envs_mean, envs_std

    def get_num_actions(self):
        return self.env.action_space_num

    def get_num_state(self):
        return self.env.output.shape[0]