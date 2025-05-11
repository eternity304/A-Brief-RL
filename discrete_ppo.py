import argparse
import os
from distutils.util import strtobool
import datetime

import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter 

import gymnasium as gym

from agent import *

def main():
    args = parse_args()
    print(args)

    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{datetime.datetime.now().strftime('%m-%d-%H%M')}"
    print(run_name)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameter",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{val}|" for key, val in vars(args).items()]))
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment
    envs = gym.vector.SyncVectorEnv([make_env(
        args.gym_id,
        args.seed + i,
        i,
        args.capture_video,
        run_name
    ) for i in range(args.num_envs)]) # vectorized environment, initialized by passing in an environment making function
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete, "Only support discrete action space")
    print("envs.single_observation_space.shape: ", envs.single_observation_space.shape)
    print("envs.single_action_space.n: ", envs.single_action_space.n)
    
    observation = envs.reset()
    episodic_return = 0
    for _ in range(args.total_timesteps):
        action = envs.action_space.sample()
        observation, reward, terminated, truncated, info = envs.step(action)
    
        if "episode" in info.keys():
            print(f"episodic return: {info['episode']['r'][0]}")

    envs.close()

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0: # flag for only capturing video in the first sub environment 
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda t: t % 1000 == 0, fps=16)
        env.reset(seed=seed)
        env.unwrapped.action_space.seed(seed)
        env.unwrapped.observation_space.seed(seed)
        return env
    return thunk


def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip("py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
        help='the learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='the seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=2000,
        help='the total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `--torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help = "whether or not to capture the agent's performance"
    )

    # Algorithm Specific
    parser.add_argument('--num-envs', type=int, default=4,
        help = 'the number of parallel game environment'
    )

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    main()