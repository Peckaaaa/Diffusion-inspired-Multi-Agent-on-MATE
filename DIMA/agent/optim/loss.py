import numpy as np
import torch
import wandb
import torch.nn.functional as F

from agent.optim.utils import rec_loss, compute_return, state_divergence_loss, calculate_ppo_loss, \
    batch_multi_agent, log_prob_loss, info_loss, compute_lambda_returns
from agent.utils.params import FreezeParameters
from networks.dreamer.rnns import rollout_representation, rollout_policy
from utils import symexp

from agent.coroutines.env_loop import rollout_policy_with_env, rollout_policy_with_env_wo_reset
from agent.world_models.world_model_env import WorldModelEnv

import ipdb

from termcolor import cprint
from tb_logger import LOGGER


def model_loss(config, model, obs, action, av_action, reward, done, fake, last):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]

    embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)
    feat = torch.cat([post.stoch, deters], -1)
    feat_dec = post.get_features()

    reconstruction_loss, i_feat = rec_loss(model.observation_decoder,
                                           feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                           obs[:-1].reshape(-1, n_agents, obs.shape[-1]),
                                           1. - fake[:-1].reshape(-1, n_agents, 1))
    reward_loss = F.smooth_l1_loss(model.reward_model(feat), reward[1:]) # Using feat here seems unusual
    pcont_loss = log_prob_loss(model.pcont, feat, (1. - done[1:]))
    av_action_loss = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)

    dis_loss = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))
    div = state_divergence_loss(prior, post, config)

    model_loss = div + reward_loss + dis_loss + reconstruction_loss + pcont_loss + av_action_loss
    if np.random.randint(20) == 4:
        wandb.log({'Model/reward_loss': reward_loss, 'Model/div': div, 'Model/av_action_loss': av_action_loss,
                   'Model/reconstruction_loss': reconstruction_loss, 'Model/info_loss': dis_loss,
                   'Model/pcont_loss': pcont_loss})

    return model_loss

def actor_rollout(obs, action, last, model, actor, critic, config):
    n_agents = obs.shape[2]
    with FreezeParameters([model]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation(model.representation, obs.shape[0], embed, action,
                                                prev_state, last)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        items = rollout_policy(model.transition, model.av_action, config.HORIZON, actor, post) ### imagination rollouts
    imag_feat = items["imag_states"].get_features()
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    
    returns = critic_rollout(model, critic, imag_feat, imag_rew_feat, items["actions"],
                             items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])), config)
    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(), imag_feat[:-1].detach(), returns.detach()]
    return [batch_multi_agent(v, n_agents) for v in output]


def critic_rollout(model, critic, states, rew_states, actions, raw_states, config):
    with FreezeParameters([model, critic]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        value = critic(states)
        discount_arr = model.pcont(rew_states).mean
        wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                   'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                   'Value/Value': value.mean()})
    returns = compute_return(imag_reward, value[:-1], discount_arr, bootstrap=value[-1], lmbda=config.DISCOUNT_LAMBDA,
                             gamma=config.GAMMA)
    return returns


def calculate_reward(model, states, mask=None):
    imag_reward = model.reward_model(states)
    if mask is not None:
        imag_reward *= mask
    return imag_reward


def calculate_next_reward(model, actions, states):
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    next_state = model.transition(actions, states)
    imag_rew_feat = torch.cat([states.stoch, next_state.deter], -1)
    return calculate_reward(model, imag_rew_feat)


@torch.no_grad()
def _ppo_diagnostics(log_rho, rho, policy_loss, entropy_term, clip_param):
    """The three numbers that tell you whether PPO is doing anything.

    approx_kl : Schulman's k3 estimator, mean(rho - 1 - log rho). Unbiased, always >= 0,
                far lower variance than mean(-log rho). ~0 means the policy is not moving.
    clip_frac : share of samples whose ratio left the trust region. 0 means the updates
                are too small to ever hit the boundary; >0.3 means they are too large.
    ratio_*   : raw importance weights, for sanity-checking the two above.
    """
    log_rho = log_rho.detach()
    rho = rho.detach()
    return {
        'Policy/approx_kl':   (rho - 1.0 - log_rho).mean(),
        'Policy/clip_frac':   ((rho - 1.0).abs() > clip_param).float().mean(),
        'Policy/ratio_mean':  rho.mean(),
        'Policy/ratio_std':   rho.std(),
        'Policy/policy_loss': policy_loss.detach().mean(),
        'Policy/entropy_term': entropy_term.detach().mean(),
    }


def actor_loss(imag_states, actions, av_actions, old_policy, advantage, actor, ent_weight, clip_param):
    """Returns (loss, diagnostics).

    The loss value itself is uninformative: `advantage` arrives already normalised to
    zero mean, and on the first PPO epoch rho == 1 exactly (old_policy IS the policy
    being updated), so the policy term is -mean(A) == 0 by construction and only the
    minibatch sampling noise makes it wobble. The diagnostics are what actually say
    whether PPO is moving the policy.
    """
    _, new_policy = actor(imag_states)
    if av_actions is not None:
        new_policy[av_actions == 0] = -1e10
    actions = actions.argmax(-1, keepdim=True)
    log_rho = (F.log_softmax(new_policy, dim=-1).gather(2, actions) -
               F.log_softmax(old_policy, dim=-1).gather(2, actions))
    rho = log_rho.exp()
    ppo_loss, ent_loss = calculate_ppo_loss(new_policy, rho, advantage, clip_param)
    if np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': ent_loss.mean(), 'Policy/Mean action': actions.float().mean()})
        # print(f'in function actor loss, entropy is {ent_loss.detach().mean().item()}')
    loss = (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean()
    return loss, _ppo_diagnostics(log_rho, rho, ppo_loss, ent_loss * ent_weight, clip_param)

## update
def continuous_actor_loss(imag_states, actions, av_actions, old_log_probs, advantage, actor, ent_weight, clip_param):
    action_log_probs, dist_entropy, _ = actor.evaluate_actions(imag_states, actions)

    imp_weights = torch.prod(
        torch.exp(action_log_probs - old_log_probs),
        dim=-1,
        keepdim=True,
    )
    surr1 = imp_weights * advantage
    surr2 = torch.clamp(imp_weights, 1.0 - clip_param, 1.0 + clip_param) * advantage

    policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
    actor_loss = policy_loss - dist_entropy * ent_weight
    _diag = _ppo_diagnostics(torch.log(imp_weights.clamp_min(1e-12)), imp_weights,
                             policy_loss, -dist_entropy * ent_weight, clip_param)
    # (policy_loss - dist_entropy * ent_weight).backward()
    if np.random.randint(10) == 9:
        wandb.log({'Policy/Entropy': dist_entropy.detach().item(), 'Policy/Mean action': actions.detach().float().mean().item()})
        LOGGER.log_scalar_wo_step('Policy/Entropy', dist_entropy.detach().item())
        LOGGER.log_scalar_wo_step('Policy/Mean action', actions.detach().float().mean().item())
        # print(f'in function continuous actor loss, entropy is {dist_entropy.detach().item()}')
    return actor_loss, _diag

def value_loss(critic, imag_feat, targets):
    value_pred = critic(imag_feat)
    mse_loss = (targets - value_pred) ** 2 / 2.0
    return torch.mean(mse_loss)


# pylint: disable-next=invalid-name
def huber_loss(e, d):
    """Huber loss."""
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


# pylint: disable-next=invalid-name
def mse_loss(e):
    """MSE loss."""
    return e**2 / 2

## diffusion world model rollout
def rollout_diffusion_world_models(training_buffer, running_mean_std, state_decoder, denoiser, rew_end_model, actor, critic, config, env_type, **kwargs):
    with FreezeParameters([denoiser, denoiser, critic]):
        wm_env = WorldModelEnv(
            running_mean_std=running_mean_std,
            state_decoder=state_decoder,
            denoiser=denoiser,
            rew_end_model=rew_end_model,
            dataset=training_buffer,    # Check if this buffer needs sampling weight adjustment
            num_envs=config.ac_batch_size,
            cfg=config.worldmodel_env_cfg,
            return_denoising_trajectory=True,
            mode='non-ensemble',
            use_stack_obs=config.use_stack,
            num_stack_obs=config.stack_obs_num,
            env_type=env_type,
            state_decoder_type=config.state_decoder_type,
            should_reset_with_dead=config.compute_end_in_TD,
            #####
            device = config.DEVICE,
        )

        if config.compute_end_in_TD:
            # cprint('using reset in wm env', 'light_magenta')
            obs, shared_obs, act, rew, pcont, end, trunc, logits_act, val, val_bootstrap, av_actions, _ = rollout_policy_with_env(wm_env, actor, critic, config.horizon)
        else:
            # cprint('Disable reset in wm env', 'light_blue')
            obs, shared_obs, act, rew, pcont, end, trunc, logits_act, val, val_bootstrap, av_actions, _ = rollout_policy_with_env_wo_reset(wm_env, actor, critic, config.horizon)
        
        end = end.to(torch.float32)
        trunc = trunc.to(torch.float32)

    del wm_env

    return obs, shared_obs, act, rew, pcont, end, trunc, logits_act, val, val_bootstrap, av_actions