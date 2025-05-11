import argparse
import os
from distutils.util import strtobool
import datetime, time

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
    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only support discrete action space"
    print("envs.single_observation_space.shape:", envs.single_observation_space.shape)
    print("envs.single_action_space.n:", envs.single_action_space.n)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Value Storage Setup
    obs = torch.zeros((args.num_steps, args.num_envs, *envs.single_observation_space.shape)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, *envs.single_action_space.shape)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    star_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0)/ num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs  # increment by the number of vector  environment
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_value(next_obs)
                values[step] = value.flatten()
            actions[step]= action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.tensor(next_obs).to(device), torch.Tensor(terminated).to(device)
            
            if "episode" in info.keys():
                episode_mask = info['_episode']
                episode_reward = info['episode']['r'][episode_mask]
                episode_length = info['episode']['l'][episode_mask]

                print(f"global_step={global_step}, episodic_return={episode_reward[0]}")
                writer.add_scalar("charts/episodic_return", episode_reward[0], global_step)
                writer.add_scalar("charts/episodic_length", episode_length[0], global_step)



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
    parser.add_argument('--num-steps', type=int, default=128,
        help='the number of steps to run in each environment per policy rollout'
    )
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, learning rate is annealed for policy and value networks'
    )
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation'
    )
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor'
    )
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation'
    )
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)

    return args

if __name__ == "__main__":
    main()