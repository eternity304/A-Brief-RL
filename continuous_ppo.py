import argparse
import os
from distutils.util import strtobool
import datetime, time
from tqdm import tqdm
import random
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter 
from torch.distributions.normal import  Normal

import gymnasium as gym

from cts_agent import *

def main():
    args = parse_args()
    print(args)

    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{datetime.datetime.now().strftime('%m-%d-%H%M')}"
    print(run_name)

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True
        )

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
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only support continuous action space"
    print("envs.single_observation_space.shape:", envs.single_observation_space.shape)

    agent = Agent(envs).to(device).float()
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
    start_time = time.time()

    for update in tqdm(range(1, num_updates + 1), desc="Number of updates"):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0)/ num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in tqdm(range(0, args.num_steps), desc='Number of steps', leave=False):
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
            
            # TD estimation of reward with GAE
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values 
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten batch
            b_obs = obs.reshape((-1, *envs.single_observation_space.shape))
            b_logprobs = logprobs.reshape(-1)
            b_actions =  actions.reshape((-1, *envs.single_action_space.shape))
            b_advantages = advantages.reshape(-1)  
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Policy and Value network optimization
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in tqdm(range(args.update_epochs), desc="Epoch", leave=False):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end  = start + args.minibatch_size
                    mb_inds =  b_inds[start:end]

                    # forward pass on minibatch observatioons
                    _, newlogprob, entropy, newvalue = agent.get_action_value(
                        b_obs[mb_inds],  b_actions[mb_inds]
                    )
                    logratio = newlogprob -  b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs +=  [((ratio - 1.0).abs() > args.clip_obj).float().mean().item()]  
                    
                    # Advantage
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    
                    # Policy Loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_obj, 1 + args.clip_obj)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    newvalue = newvalue.view(-1)
                    if args.clip_valLoss:
                        v_loss_unclipped = (newvalue + b_returns[mb_inds]) ** 2
                        v_clipped =  b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_obj,
                            args.clip_obj
                        )
                        v_loss_clipped  = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss  = 0.5 *v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds])  **  2).mean()

                    entropy_loss =  entropy.mean()
                    loss = pg_loss - args.entropy_coef *entropy_loss + v_loss * args.valLoss_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    save_path = f"weight/{run_name}_weight.pth"
    torch.save(agent.state_dict(), save_path)

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode='rgb_array')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0: # flag for only capturing video in the first sub environment 
                env = gym.wrappers.RecordVideo(env, f"/scratch/ondemand28/harryscz/A-Brief-RL/videos/{run_name}", step_trigger=lambda t: t % 100000  == 0, fps=16)
                # env = AutoResetVideoWrapper(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10).astype(np.float32), observation_space=env.observation_space)
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.reset(seed=seed)
        env.unwrapped.action_space.seed(seed)
        env.unwrapped.observation_space.seed(seed)
        return env
    return thunk

# class AutoResetVideoWrapper(gym.Wrapper):
#     def step(self, action):
#         obs, reward, terminated, truncated, info = super().step(action)
#         # whenever the episode finishes, force a reset so RecordVideo dumps its file
#         if terminated or truncated:
#             super().reset()
#         return obs, reward, terminated, truncated, info

def parse_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip("py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default='HalfCheetah-v5',
        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
        help='the learning rate for the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='the seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=2000000,
        help='the total timesteps of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `--torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help = "whether or not to capture the agent's performance")
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True,
        help = "if toggled, updates are tracked with wandb")
    parser.add_argument('--wandb-project-name', type=str, default="RL Run",
        help="the wandb project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity name for wandb")

    # Algorithm Specific
    parser.add_argument('--num-envs', type=int, default=1,
        help = 'the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=2048 ,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, learning rate is annealed for policy and value networks')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=32 ,
        help='the number of minibatches')
    parser.add_argument('--update-epochs', type=int, default=10,
        help='the number of epochs to update the policy')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, advantage will be normalized')
    parser.add_argument('--clip-obj', type=float, default=0.2,
        help='coefficient for clipping surrogate objective')
    parser.add_argument('--clip-valLoss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, loss for value function will be clipped')
    parser.add_argument('--entropy-coef', type=float, default=0.0,
        help='the coefficient of the entropy')
    parser.add_argument('--valLoss-coef', type=float, default=0.5,
        help='the coefficient for the loss of value or actor network')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='the upperbound for gradient norm')
    parser.add_argument('--target-kl',  type=float,  default=None,
        help='the target KL divergence threshold')

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args

if __name__ == "__main__":
    main()