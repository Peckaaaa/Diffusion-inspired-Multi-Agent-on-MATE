"""Owns the world model: tokenizer, categorical diffusion, reward/termination.

Two responsibilities:

``update``    one gradient step on real transitions for all three components.
``imagine``   an H-step latent rollout that produces the batch MAPPO trains on.
"""

import torch

from models.categorical_diffusion import CategoricalDiffusion
from models.reward_termination import RewardTerminationModel
from models.state_autoencoder import StateAutoEncoder
from models.transformer_denoiser import JointCameraDenoiser


class WorldModelTrainer:
    def __init__(self, env_spec, config, device):
        self.device = torch.device(device)
        self.config = config

        wm = config['world_model']
        diffusion_config = config['diffusion']

        self.autoencoder = StateAutoEncoder(
            state_dim=env_spec['state_dim'],
            obs_dim=env_spec['obs_dim'],
            msg_dim=env_spec['msg_dim'],
            n_agents=env_spec['n_agents'],
            levels=tuple(wm['levels']),
            num_tokens=wm['num_tokens'],
            hidden_dim=wm['ae_hidden'],
        ).to(self.device)

        denoiser = JointCameraDenoiser(
            num_classes=self.autoencoder.codebook_size,
            num_tokens=wm['num_tokens'],
            n_agents=env_spec['n_agents'],
            action_dim=env_spec['action_dim'],
            msg_dim=env_spec['msg_dim'],
            hidden_dim=wm['tf_hidden'],
            n_layers=wm['tf_layers'],
            n_heads=wm['tf_heads'],
        )
        self.diffusion = CategoricalDiffusion(
            denoiser=denoiser,
            num_classes=self.autoencoder.codebook_size,
            num_tokens=wm['num_tokens'],
            num_steps=diffusion_config['num_steps'],
            sample_steps=diffusion_config['sample_steps'],
            noise_type=diffusion_config['noise_type'],
            schedule=diffusion_config['schedule'],
        ).to(self.device)

        self.reward_model = RewardTerminationModel(
            num_classes=self.autoencoder.codebook_size,
            num_tokens=wm['num_tokens'],
            n_agents=env_spec['n_agents'],
            action_dim=env_spec['action_dim'],
            hidden_dim=wm['tf_hidden'],
            n_layers=max(wm['tf_layers'] // 2, 1),
            n_heads=wm['tf_heads'],
            max_length=max(wm['reward_seq_len'], 1),
        ).to(self.device)

        self.optimizers = {
            'ae': torch.optim.Adam(self.autoencoder.parameters(), lr=wm['lr']),
            'diffusion': torch.optim.Adam(self.diffusion.parameters(), lr=wm['lr']),
            'reward': torch.optim.Adam(self.reward_model.parameters(), lr=wm['lr']),
        }
        self.grad_clip = wm['grad_clip']
        self.reward_seq_len = wm['reward_seq_len']

    # ------------------------------------------------------------------ training

    def _step(self, name, module, loss):
        self.optimizers[name].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip)
        self.optimizers[name].step()

    def update(self, buffer, batch_size):
        """One gradient step for the tokenizer, the diffusion model and the reward model."""

        batch = buffer.sample(batch_size, self.device)

        ae_loss, metrics = self.autoencoder.loss(
            batch['states'], batch['obs'], batch['messages_in']
        )
        self._step('ae', self.autoencoder, ae_loss)

        # Tokens are targets, not differentiable inputs: the diffusion and reward
        # models are trained against the codes the (just updated) tokenizer emits.
        with torch.no_grad():
            indices = self.autoencoder.encode_indices(batch['states'], batch['messages_in'])
            next_indices = self.autoencoder.encode_indices(
                batch['next_states'], batch['next_messages_in']
            )

        diffusion_loss, diffusion_metrics = self.diffusion.loss(
            clean_indices=next_indices,
            context_indices=indices,
            actions=batch['actions'],
            emissions=batch['messages_out'],
        )
        self._step('diffusion', self.diffusion, diffusion_loss)

        sequence = buffer.sample_sequences(batch_size, self.reward_seq_len, self.device)
        with torch.no_grad():
            flat_indices = self.autoencoder.encode_indices(
                sequence['states'].flatten(0, 1), sequence['messages_in'].flatten(0, 1)
            ).view(*sequence['states'].shape[:2], -1)

        reward_loss, reward_metrics = self.reward_model.loss(
            flat_indices, sequence['actions'], sequence['rewards'], sequence['dones']
        )
        self._step('reward', self.reward_model, reward_loss)

        metrics.update(diffusion_metrics)
        metrics.update(reward_metrics)
        return metrics

    # --------------------------------------------------------------- imagination

    @torch.no_grad()
    def imagine(self, policy, buffer, config):
        """Roll the world model forward from replayed start states.

        Each imagined step draws ``I_{h+1}`` with the categorical reverse process
        -- ``sample_steps`` denoiser passes, independent of the ``num_steps`` the
        model was trained with.
        """

        policy_config = config['policy']
        horizon = policy_config['horizon']
        n_agents = buffer.obs.shape[1]

        batch = buffer.sample(config['train']['imagine_batch_size'], self.device)
        indices = self.autoencoder.encode_indices(batch['states'], batch['messages_in'])
        size = indices.shape[0]

        obs_seq, msg_seq, sample_seq, logp_seq = [], [], [], []
        state_seq, value_seq, reward_seq, continue_seq = [], [], [], []

        for _ in range(horizon):
            hat_state, hat_obs, hat_msg = self.autoencoder.decode_indices(indices)

            flat_obs = hat_obs.reshape(size * n_agents, -1)
            flat_msg = hat_msg.reshape(size * n_agents, -1)
            action, emission, log_prob, sample = policy.actor(flat_obs, flat_msg)

            action = action.view(size, n_agents, -1).clamp(-1.0, 1.0)
            emission = emission.view(size, n_agents, -1)

            reward, done_prob = self.reward_model.predict(indices, action)

            obs_seq.append(flat_obs)
            msg_seq.append(flat_msg)
            sample_seq.append(sample)
            logp_seq.append(log_prob)
            state_seq.append(hat_state)
            value_seq.append(policy.critic(hat_state))
            reward_seq.append(reward)
            continue_seq.append(policy_config['gamma'] * (1.0 - done_prob))

            indices = self.diffusion.sample(
                context_indices=indices,
                actions=action,
                emissions=emission,
                temperature=config['diffusion']['sample_temperature'],
            )

        hat_state, _, _ = self.autoencoder.decode_indices(indices)
        bootstrap = policy.critic(hat_state)

        rewards = torch.stack(reward_seq)
        values = torch.stack(value_seq)
        continues = torch.stack(continue_seq)
        returns = policy.lambda_returns(rewards, values, continues, bootstrap)
        advantages = returns - values

        repeat = lambda x: x.unsqueeze(-1).expand(-1, -1, n_agents).reshape(-1)

        rollout = {
            'obs': torch.cat(obs_seq, dim=0),
            'messages': torch.cat(msg_seq, dim=0),
            'samples': torch.cat(sample_seq, dim=0),
            'old_log_probs': torch.cat(logp_seq, dim=0),
            'advantages': repeat(advantages),
            # The critic is fit on every (step, agent) copy of its state so the
            # actor and critic minibatch indices stay aligned; the duplication
            # only scales the value-loss gradient, which value_coef absorbs.
            'states': torch.stack(state_seq)
            .unsqueeze(2)
            .expand(-1, -1, n_agents, -1)
            .reshape(horizon * size * n_agents, -1),
            'returns': repeat(returns),
            'old_values': repeat(values),
        }
        stats = {
            'imag/reward_mean': rewards.mean().item(),
            'imag/value_mean': values.mean().item(),
            'imag/return_mean': returns.mean().item(),
            'imag/continue_mean': continues.mean().item(),
        }
        return rollout, stats

    # ---------------------------------------------------------------- serialize

    def state_dict(self):
        return {
            'autoencoder': self.autoencoder.state_dict(),
            'diffusion': self.diffusion.state_dict(),
            'reward_model': self.reward_model.state_dict(),
        }

    def load_state_dict(self, d):
        self.autoencoder.load_state_dict(d['autoencoder'])
        self.diffusion.load_state_dict(d['diffusion'])
        self.reward_model.load_state_dict(d['reward_model'])
