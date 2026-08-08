import random
from typing import Generator, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions import OneHotCategorical

from . import coroutine
from agent.world_models.world_model_env import WorldModelEnv


@coroutine
def make_env_loop(
    env: WorldModelEnv, model: nn.Module, num_agent: int, epsilon: float = 0.0
) -> Generator[Tuple[torch.Tensor, ...], int, None]:
    num_steps = yield

    # hx = torch.zeros(num_agent, env.num_envs, model.lstm_dim, device=model.device)
    # cx = torch.zeros(num_agent, env.num_envs, model.lstm_dim, device=model.device)
    actor, critic = model
    seed = random.randint(0, 2**31 - 1)
    obs, shared_obs, av_action = env.reset(seed=[seed + i for i in range(env.num_envs)])    # (b, n, obs_dim), (b, n, state_dim), (b, n, act_dim)

    while True:
        # hx, cx = hx.detach(), cx.detach()
        all_ = []
        infos = []
        av_actions = []
        n = 0

        while n < num_steps:
            if env.is_continuous_act:
                act, logits_act = actor(obs, deterministic = False)
            else:
                _, logits_act = actor(obs)

                if av_action is not None:
                    logits_act[av_action == 0] = -1e10
                    av_actions.append(av_action)

                action_dist = OneHotCategorical(logits=logits_act)
                act = action_dist.sample().squeeze(0)

            val = critic(shared_obs)

            next_obs, next_shared_obs, rew, end, trunc, info = env.step(act)
            next_av_action = info.get('av_action', None)

            if n > 0:
                val_bootstrap = val.detach().clone()
                if dead.any():
                    val_bootstrap[dead] = val_final_obs
                all_[-1][-1] = val_bootstrap

            if trunc.ndim == 2:
                dead = torch.logical_or(end, trunc).any(dim=-1) 
            else:
                dead = torch.logical_or(end, trunc)

            if dead.any():
                with torch.no_grad():
                    val_final_obs = critic(info["final_state"])

                    # if is_ma:
                    #     _, val_final_obs, _ = model.predict_act_value(info["final_state"], (hx[:, dead], cx[:, dead]))
                    # else:
                    #     _, val_final_obs, _ = model.predict_act_value(info["final_observation"], (hx[dead], cx[dead]))
                # reset_gate = 1 - dead.float().unsqueeze(1)
                # if is_ma:
                #     reset_gate = reset_gate.unsqueeze(0).repeat(num_agents, 1, 1)
                # hx = hx * reset_gate
                # cx = cx * reset_gate
                # if "burnin_obs" in info:
                #     burnin_obs = info["burnin_obs"]
                #     for i in range(burnin_obs.size(1)):
                #         if is_ma:
                #             _, _, (hx[:, dead], cx[:, dead]) = model.predict_act_value(burnin_obs[:, i], (hx[:, dead], cx[:, dead]))
                #         else:
                #             _, _, (hx[dead], cx[dead]) = model.predict_act_value(burnin_obs[:, i], (hx[dead], cx[dead]))

            all_.append([obs, shared_obs, act, rew, end, trunc, logits_act, val, None])
            infos.append(info)

            obs = next_obs
            shared_obs = next_shared_obs
            av_action = next_av_action
            n += 1

        with torch.no_grad():
            val_bootstrap = critic(next_shared_obs)
            # _, val_bootstrap, _ = model.predict_act_value(next_obs, (hx, cx))  # do not update hx/cx

        if dead.any():
            val_bootstrap[dead] = val_final_obs

        all_[-1][-1] = val_bootstrap

        all_av_actions = torch.stack(av_actions, dim=1) if len(av_actions) > 0 else None

        # (num_envs, trajectory_length)
        all_obs, all_shared_obs, act, rew, end, trunc, logits_act, val, val_bootstrap = (torch.stack(x, dim=1) for x in zip(*all_))

        num_steps = yield all_obs, all_shared_obs, act, rew, end, trunc, logits_act, val, val_bootstrap, all_av_actions, infos


from einops import rearrange

def rollout_policy_with_env(env, actor, critic, horizons):
    all_ = []
    infos = []
    av_actions = []
    n = 0

    # initialize wm_env
    seed = random.randint(0, 2**31 - 1)
    obs, shared_obs, av_action = env.reset(seed=[seed + i for i in range(env.num_envs)])
    
    while n < horizons:
        if getattr(env, 'use_stack_obs', False):
            obs = rearrange(obs.permute(0, 2, 1, 3), 'b n t d -> b n (t d)')

        if env.is_continuous_act:
            act, logits_act = actor(obs, deterministic = False)
        else:
            _, logits_act = actor(obs)

            if av_action is not None:
                logits_act[av_action == 0] = -1e10
                av_actions.append(av_action)
            
            action_dist = OneHotCategorical(logits=logits_act)
            act = action_dist.sample().squeeze(0)
        
        val = critic(shared_obs)

        next_obs, next_shared_obs, rew, pcont, end, trunc, info = env.step(act)
        next_av_action = info.get('av_action', None)
        
        if n > 0:
            val_bootstrap = val.detach().clone()
            if dead.any():
                val_bootstrap[dead] = val_final_obs
            all_[-1][-1] = val_bootstrap

        if trunc.ndim == 2:
            dead = torch.logical_or(end, trunc).any(dim=-1) # (num_envs, num_agents)
        else:
            dead = torch.logical_or(end, trunc)

        if dead.any():
            with torch.no_grad():
                val_final_obs = critic(info["final_state"])
        
        # all_.append([obs, shared_obs, act, rew, end, trunc, logits_act, val, None])
        all_.append([obs, shared_obs, act, rew, pcont, end, trunc, logits_act, val, None])
        infos.append(info)

        obs = next_obs
        shared_obs = next_shared_obs
        av_action = next_av_action
        n += 1
    
    if getattr(env, 'use_stack_obs', False):
        next_obs = rearrange(next_obs.permute(0, 2, 1, 3), 'b n t d -> b n (t d)')

    with torch.no_grad():
        val_bootstrap = critic(next_shared_obs)

    if dead.any():
        val_bootstrap[dead] = val_final_obs

    all_[-1][-1] = val_bootstrap
    all_av_actions = torch.stack(av_actions, dim=1) if len(av_actions) > 0 else None

    # (num_envs, trajectory_length)
    all_obs, all_shared_obs, act, rew, pcont, end, trunc, logits_act, val, val_bootstrap = (torch.stack(x, dim=1).detach() for x in zip(*all_))

    return all_obs, all_shared_obs, act, rew, pcont, end, trunc, logits_act, val, val_bootstrap, all_av_actions, infos


def rollout_policy_with_env_wo_reset(env, actor, critic, horizons):
    all_ = []
    infos = []
    av_actions = []
    n = 0

    # initialize wm_env
    seed = random.randint(0, 2**31 - 1)
    obs, shared_obs, av_action = env.reset(seed=[seed + i for i in range(env.num_envs)])
    
    while n < horizons:
        if getattr(env, 'use_stack_obs', False):
            obs = rearrange(obs.permute(0, 2, 1, 3), 'b n t d -> b n (t d)')

        if env.is_continuous_act:
            act, logits_act = actor(obs, deterministic = False)
        else:
            _, logits_act = actor(obs)

            if av_action is not None:
                logits_act[av_action == 0] = -1e10
                av_actions.append(av_action)
            
            action_dist = OneHotCategorical(logits=logits_act)
            act = action_dist.sample().squeeze(0)
        
        if getattr(critic, '_shape', (1,)):
            val = critic(shared_obs)
        else:
            val = critic(shared_obs).mean()

        next_obs, next_shared_obs, rew, pcont, end, trunc, info = env.step(act)
        next_av_action = info.get('av_action', None)
        
        if n > 0:
            val_bootstrap = val.detach().clone()
            all_[-1][-1] = val_bootstrap
        
        # all_.append([obs, shared_obs, act, rew, end, trunc, logits_act, val, None])
        all_.append([obs, shared_obs, act, rew, pcont, end, trunc, logits_act, val, None])
        infos.append(info)

        obs = next_obs
        shared_obs = next_shared_obs
        av_action = next_av_action
        n += 1
    
    if getattr(env, 'use_stack_obs', False):
        next_obs = rearrange(next_obs.permute(0, 2, 1, 3), 'b n t d -> b n (t d)')

    with torch.no_grad():
        if getattr(critic, '_shape', (1,)):
            val_bootstrap = critic(next_shared_obs)
        else:
            val_bootstrap = critic(next_shared_obs).mean()

    all_[-1][-1] = val_bootstrap
    all_av_actions = torch.stack(av_actions, dim=1) if len(av_actions) > 0 else None

    # (num_envs, trajectory_length)
    all_obs, all_shared_obs, act, rew, pcont, end, trunc, logits_act, val, val_bootstrap = (torch.stack(x, dim=1).detach() for x in zip(*all_))

    return all_obs, all_shared_obs, act, rew, pcont, end, trunc, logits_act, val, val_bootstrap, all_av_actions, infos
