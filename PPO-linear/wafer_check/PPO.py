from pathlib import Path
from model import ValueNetwork, PolicyNetwork, train_policy_network, train_value_network, device
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import gym
from replay import Episode, History
import torch
import numpy as np
import cv2
from env.environment import Environment
import torch.nn as nn

def __init_weights(m):
    if isinstance(m ,nn.Conv2d):
        torch.nn.init.kaiming_normal(m.weight)
        m.bias.data.fill_(0.01)
        return
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

env_name = 'wafer_check'

def __to_gray_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_rgb_resized = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_CUBIC)
    return np.expand_dims(img_rgb_resized, axis=0)

def main(reward_scale=20.0, clip=0.2, log_dir="./logs", learning_rate=0.001, state_scale=1.0):
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=env_name, comment=env_name)
    
    env = Environment()
    observation = env.reset()[0]
    action_dim = env.get_num_actions()
    feature_dim = 1*24*24

    value_model = ValueNetwork(state_dim=feature_dim).to(device)
    value_model.apply(__init_weights)
    value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)

    policy_model = PolicyNetwork(state_dim=feature_dim, action_dim=action_dim).to(device)
    policy_model.apply(__init_weights)

    policy_optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)

    n_epoch = 10
    max_episodes = 20
    max_timesteps = 1000

    batch_size = 128

    max_iterations = 200

    history = History()
    epoch_ite = 0
    episode_ite = 0

    for ite in tqdm(range(max_iterations)):
        if ite % 50 == 0:
            torch.save(
                policy_model.state_dict(),
                Path(log_dir) / (env_name + f"_{str(ite)}_policy.pth"),
            )
            torch.save(
                value_model.state_dict(),
                Path(log_dir) / (env_name + f"_{str(ite)}_value.pth"),
            )

        for episode_i in range(max_episodes):
            observation = env.reset()[0]
            observation = np.reshape(observation,(1,-1))
            episode = Episode()
            for timestep in range(max_timesteps):
                action, log_probability = policy_model.sample_action(observation)
                value = value_model.state_value(observation)
                new_observation, reward, done, _,_,_ = env.step(action)
                episode.append(
                    observation=observation,
                    action=action,
                    reward=reward,
                    value=value,
                    log_probability=log_probability,
                    reward_scale=reward_scale,
                )

                observation = np.reshape(new_observation,(1,-1))
                if done:
                    episode.end_episode(last_value=0)
                    break

                if timestep == max_timesteps - 1:
                    value = value_model.state_value(observation)
                    episode.end_episode(last_value=value)

            episode_ite += 1
            writer.add_scalar(
                "Average Episode Reward",
                reward_scale * np.sum(episode.rewards),
                episode_ite,
            )
            writer.add_scalar(
                "Average Probabilities",
                np.exp(np.mean(episode.log_probabilities)),
                episode_ite,
            )

            history.add_episode(episode)

        history.build_dataset()
        data_loader = DataLoader(history, batch_size=batch_size, shuffle=True)

        policy_loss = train_policy_network(policy_model, policy_optimizer, data_loader, epoches=n_epoch, clip=clip)

        value_loss = train_value_network(
            value_model, value_optimizer, data_loader, epoches=n_epoch
        )

        for p_l, v_l in zip(policy_loss, value_loss):
            epoch_ite += 1
            writer.add_scalar("Policy Loss", p_l, epoch_ite)
            writer.add_scalar("Value Loss", v_l, epoch_ite)
        history.free_memory()


if __name__ == "__main__":

    main(
        reward_scale=20.0,
        clip=0.2,
        learning_rate=0.001,
        state_scale=1.0,
        log_dir="logs"
    )