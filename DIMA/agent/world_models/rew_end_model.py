from dataclasses import dataclass
from typing import List, Optional, Tuple
from einops import rearrange, repeat
from termcolor import cprint

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torchvision.ops.focal_loss import sigmoid_focal_loss
# from torcheval.metrics.functional import multiclass_confusion_matrix

from .temporal_unet import Downsample1d
from .blocks import Conv3x3, Downsample, ResBlocks
from .layers import mlp, SimNorm
from .perceiver import SimpleActionEncoder
# from data import Batch
from utils import init_lstm, LossAndLogs, init_weights


## initialization
def weight_init(m):
	"""Custom weight initialization for TD-MPC2."""
	if isinstance(m, nn.Linear):
		nn.init.trunc_normal_(m.weight, std=0.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Embedding):
		nn.init.uniform_(m.weight, -0.02, 0.02)
	elif isinstance(m, nn.ParameterList):
		for i,p in enumerate(m):
			if p.dim() == 3: # Linear
				nn.init.trunc_normal_(p, std=0.02) # Weight
				nn.init.constant_(m[i+1], 0) # Bias


def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)


## pre-process continuation
def compute_con(is_terminal, contdisc=True, gamma=0.997):
    con = (~is_terminal).float()  # 取反，并转换为浮点数
    if contdisc:
        con *= gamma  # multiply a discount factor
    return con

@dataclass
class RewEndModelConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[int]
    num_actions: Optional[int] = None

@dataclass
class StateRewEndModelConfig:
    lstm_dim: 512
    cond_channels: int
    depths: List[int]
    dim: int
    dim_mults: List[int]
    attn_depths: List[bool]
    mlp_dim: int

class RewEndModel(nn.Module):
    def __init__(self, cfg: StateRewEndModelConfig,
                 num_agents: int,
                 state_dim: int,
                 action_dim: int,
                 is_continuous_act: bool,
                 ## related to environment
                 pred_shared_reward: bool,
                 pred_shared_continuation: bool,
                 pred_av_action: bool,
                 use_ce_for_cont: bool,
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.is_continuous_act = is_continuous_act
        self.cfg = cfg

        # self.encoder = RewEndEncoder1D(state_dim, cfg.cond_channels, cfg.depths, cfg.dim, cfg.dim_mults, cfg.attn_depths)
        self.encoder = RewEndEncoder1D_Chi(state_dim, cfg.cond_channels, 128, [1, 2], cond_predict_scale=True, kernel_size=3)
        self.act_emb = SimpleActionEncoder(
            num_agents=num_agents, action_dim=action_dim, is_continuous_act=is_continuous_act, output_dim=cfg.cond_channels,
            num_heads=8, embed_dim=256, attn_dropout=0.1, ff_dropout=0.1, depth=3
        )

        self.pred_shared_reward = pred_shared_reward
        self.pred_shared_continuation = pred_shared_continuation
        self.use_ce_for_cont = use_ce_for_cont
        self.pred_av_action = pred_av_action

        self.latent_dim = 256

        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()

        # self.lstm = nn.LSTM(self.latent_dim, cfg.lstm_dim, batch_first=True)
        self.gru = nn.GRUCell(self.latent_dim, cfg.lstm_dim)
        init_lstm(self.gru)

        ### 一个可能需要注意的点，目前的mlp沿用td-mpc2，直接使用的是Mish激活，而diamond里面默认都用SiLu
        # predict the item of next timestep
        if self.pred_shared_reward:
            self.reward_head = nn.Sequential(
                nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
                nn.LayerNorm(cfg.lstm_dim),
                nn.SiLU(),
                nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
                nn.LayerNorm(cfg.lstm_dim),
                nn.SiLU(),
                nn.Linear(cfg.lstm_dim, 1, bias=False),
            )
        else:
            self.reward_head = nn.Sequential(
                nn.Linear(cfg.lstm_dim + num_agents, cfg.lstm_dim),
                nn.LayerNorm(cfg.lstm_dim),
                nn.SiLU(),
                nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
                nn.LayerNorm(cfg.lstm_dim),
                nn.SiLU(),
                nn.Linear(cfg.lstm_dim, 1, bias=False),
            )
        
        ## credits to Dreamer V3
        # end head predict the logit of continuation
        con_head_output_dim = 2 if self.use_ce_for_cont else 1
        cprint(f"Using {'CE' if self.use_ce_for_cont else 'NLL'} for prediction of continuation.", color="cyan" if self.use_ce_for_cont else "yellow", attrs=["bold"])

        if self.pred_shared_continuation:
            self.con_head =nn.Sequential(
                nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
                nn.LayerNorm(cfg.lstm_dim),
                nn.SiLU(),
                nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
                nn.LayerNorm(cfg.lstm_dim),
                nn.SiLU(),
                nn.Linear(cfg.lstm_dim, con_head_output_dim, bias=False),
            )
        else:
            self.con_head = nn.Sequential(
                nn.Linear(cfg.lstm_dim + num_agents, cfg.lstm_dim),
                nn.LayerNorm(cfg.lstm_dim),
                nn.SiLU(),
                nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
                nn.LayerNorm(cfg.lstm_dim),
                nn.SiLU(),
                nn.Linear(cfg.lstm_dim, con_head_output_dim, bias=False),
            )
        ## ---------------------

        if self.pred_av_action:
            self.av_action_head = nn.Sequential(
                nn.Linear(cfg.lstm_dim + num_agents, cfg.lstm_dim),
                nn.LayerNorm(cfg.lstm_dim),
                nn.SiLU(),
                nn.Linear(cfg.lstm_dim, cfg.lstm_dim),
                nn.LayerNorm(cfg.lstm_dim),
                nn.SiLU(),
                nn.Linear(cfg.lstm_dim, 2 * action_dim, bias=False),
            )

        self.apply(init_weights)

    def predict_rew_end(
        self,
        state: Tensor,
        act: Tensor,
        next_state: Tensor,
        hx: Optional[Tensor] = None,
        done: Tensor = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert act.size(2) == self.num_agents
        b, t, n = act.shape[:3]
        device = act.device

        state, act, next_state = state.reshape(b * t, -1), act.reshape(b * t, n, -1), next_state.reshape(b * t, -1)
        if not self.is_continuous_act:
            act = act.argmax(-1)

        act_cond = self.act_emb(act)
        x = torch.stack((state, next_state), dim=1)
        x = rearrange(x, 'b h t -> b t h')
        x = self.encoder(x, act_cond)
        x = x.reshape(b, t, -1)

        seq_len = x.size(1)
        outputs = []
        for i in range(seq_len):
            hx = self.gru(x[:, i], hx)
            outputs.append(hx)
            if done is not None:
                # hx = hx * (1 - done[:, i].float()).unsqueeze(-1)
                hx = (hx * (1 - done[:, i].float()).unsqueeze(-1)).detach()

        x = torch.stack(outputs, dim=1)

        # predictition of every possible module
        agent_id = torch.eye(n, dtype=torch.float32, device=device).detach()
        agent_id = repeat(agent_id, 'n d -> b t n d', b = b, t = t).detach()
        agent_x = torch.cat([x.unsqueeze(-2).repeat(1, 1, n, 1), agent_id], dim=-1)

        if self.pred_shared_reward:
            pred_r = self.reward_head(x)
        else:
            pred_r = self.reward_head(agent_x)
        
        # output logits
        if self.pred_shared_continuation:
            logits_e = self.con_head(x)
        else:
            logits_e = self.con_head(agent_x)

        if self.pred_av_action:
            logits_av_action = self.av_action_head(agent_x)
            logits_av_action = logits_av_action.view(agent_x.shape[:-1] + (self.action_dim, 2))
        else:
            logits_av_action = None

        return pred_r, logits_e, logits_av_action, hx

    def forward(self, batch, gamma, contdisc, **kwargs) -> LossAndLogs:
        assert batch['action'].size(2) == self.num_agents

        state      = batch['shared_obs']
        act        = batch['action']
        next_state = batch['next_shared_obs']
        
        rew = batch['reward']
        end = batch['done']
        # mask = batch.mask_padding

        b, t = state.shape[:2]

        pred_r, logits_e, logits_av_action, _ = self.predict_rew_end(state, act, next_state, done=end.all(-2).squeeze(-1))

        # pred_r     = pred_r[mask]
        # logits_e   = logits_e[mask]
        # target_rew = rew[mask]
        # target_end = end[mask]

        if self.pred_shared_reward:
            loss_rew = F.smooth_l1_loss(pred_r, rew.mean(dim=2))
        else:
            loss_rew = F.smooth_l1_loss(pred_r, rew)
        
        if self.use_ce_for_cont:
            if self.pred_shared_continuation:
                loss_end = F.cross_entropy(logits_e.reshape(-1, 2), end.all(dim=2).to(torch.long).reshape(-1,))
            else:
                loss_end = F.cross_entropy(logits_e.reshape(-1, 2), end.to(torch.long).reshape(-1,))

        else:
            ### 使用Dreamer V3 的方法训练
            if self.pred_shared_continuation:
                target_con = compute_con(end.all(dim=2), contdisc, gamma)
            else:
                target_con = compute_con(end.to(torch.bool), contdisc, gamma)

            # using focal loss
            # loss_end = sigmoid_focal_loss(logits_e, target_con, reduction='mean')

            # using NLL loss
            # cont_pred = td.independent.Independent(td.bernoulli.Bernoulli(logits=logits_e), 1)
            # loss_end = -torch.mean(cont_pred.log_prob(target_con))
            loss_end = self.bce_with_logits_loss_func(logits_e, target_con)

        if self.pred_av_action:
            tmp = torch.roll(batch['done'], 1, dims=1).squeeze(-1)
            labels_av_actions = batch['av_action']
            labels_av_actions[tmp == True] = torch.ones_like(labels_av_actions[tmp == True])

            logits_av_actions = rearrange(logits_av_action[:, :-1], 'b l n a e -> (b l n) a e').reshape(-1, logits_av_action.size(-1))
            labels_av_actions = rearrange(labels_av_actions[:, 1:], 'b l n a -> (b l n) a').reshape(-1,).to(torch.long)
            loss_av_action = F.cross_entropy(logits_av_actions, labels_av_actions)

            # target_av_action = rearrange(target_av_action, 'b n a -> (b n a)').to(torch.long)
            # logits_av_action = rearrange(logits_av_action, 'b n a e -> (b n a) e')

            # loss_av_action = F.cross_entropy(logits_av_action, target_av_action)
        else:
            loss_av_action = torch.tensor(0.)

        loss = loss_rew + loss_end + loss_av_action

        metrics = {
            "loss_rew": loss_rew.detach(),
            "loss_end": loss_end.detach(),
            "loss_av_action": loss_av_action.detach() if self.pred_av_action else 0.,
            "loss_total": loss.detach(),
            # "confusion_matrix": {
            #     "rew": multiclass_confusion_matrix(logits_rew, target_rew, num_classes=3),
            #     "end": multiclass_confusion_matrix(logits_end, target_end, num_classes=2),
            # },
        }
        return loss, metrics


# implement RewEndModelEncoder1D with Chi's ResBlock
from .temporal_unet import ChiResidualBlock, GroupNorm1d
class RewEndEncoder1D_Chi(nn.Module):
    def __init__(
        self,
        in_dim: int,
        cond_channels: int,     # 128
        dim: int,               # 128
        dim_mults: List[int],   # [1, 2]
        cond_predict_scale: bool = True,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        dims = [in_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        self.downs = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ChiResidualBlock(
                    dim_in, dim_out, cond_channels, kernel_size, cond_predict_scale),
                ChiResidualBlock(
                    dim_out, dim_out, cond_channels, kernel_size, cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
        
        self.final_proj = nn.Flatten()
        # self.final_conv = nn.Sequential(
        #     nn.Conv1d(dims[-1], dims[-1], kernel_size, padding=kernel_size // 2),
        #     GroupNorm1d(dim, 8, 4), nn.Mish(),
        #     nn.Conv1d(dims[-1], dims[-1], 1))

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        for idx, (resnet1, resnet2, downsample) in enumerate(self.downs):
            x = resnet1(x, cond)
            x = resnet2(x, cond)
            x = downsample(x)

        x = self.final_proj(x)
        return x


class RewEndEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        depths: List[int],
        channels: List[int],
        attn_depths: List[int],
    ) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self.conv_in = Conv3x3(in_channels, channels[0])
        blocks = []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        blocks.append(
            ResBlocks(
                list_in_channels=[channels[-1]] * 2,
                list_out_channels=[channels[-1]] * 2,
                cond_channels=cond_channels,
                attn=True,
            )
        )
        self.blocks = nn.ModuleList(blocks)
        self.downsamples = nn.ModuleList([nn.Identity()] + [Downsample(c) for c in channels[:-1]] + [nn.Identity()])

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x = self.conv_in(x)
        for block, down in zip(self.blocks, self.downsamples):
            x = down(x)
            x, _ = block(x, cond)
        return x

from .transformer import Transformer, TransformerConfig, get_sinusoid_encoding_table
from .transformer import Perceiver, PerceiverConfig
from .slicer import Embedder, Head, Slicer, SpecialHead, DiscreteDist, GeneralEmbedder, AgentWiseHead
from .kv_caching import KeysValues
import dataclasses

from networks.tools import LOSSES_DICT


@dataclass
class TransRewEndModelOutput:
    output_sequence: torch.FloatTensor
    pred_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor
    pred_avail_action: torch.FloatTensor
    attn_output: List

class TransRewEndModel(nn.Module):
    def __init__(self, state_dim: int, act_vocab_size: int, num_agents: int,
                 config: TransformerConfig,
                 action_dim: int, is_discrete_action: bool,
                 ### options for setting prediction head
                 use_symlog: bool = False, use_ce_for_end: bool = False, use_ce_for_av_action: bool = True, enable_av_pred: bool = False,
                 use_ce_for_reward: bool = False, rewards_prediction_config: dict = None) -> None:
        super().__init__()
        self.state_dim, self.act_vocab_size = state_dim, act_vocab_size
        self.num_modalities = 2
        self.is_discrete_action = is_discrete_action

        self.config = config
        self.num_agents = num_agents

        # Here, config.tokens_per_block = 2
        self.num_action_tokens = 1  # for continuous task, this should be dimension of joint action (e.g. like ManiSkill2)
        self.num_obs_tokens = config.tokens_per_block - self.num_action_tokens

        self.transformer = Transformer(config)

        act_tokens_pattern = torch.zeros(config.tokens_per_block)
        act_tokens_pattern[-self.num_action_tokens:] = 1
        self.act_tokens_pattern = act_tokens_pattern

        obs_tokens_pattern = torch.zeros(config.tokens_per_block)
        obs_tokens_pattern[:self.num_obs_tokens] = 1
        self.obs_tokens_pattern = obs_tokens_pattern

        ### due to attention mask, the last token of transformer output is generated by all tokens of input
        all_but_last_pattern = torch.zeros(config.tokens_per_block)
        all_but_last_pattern[-1] = 1

        # self.perattn_slicer = Slicer(max_blocks=config.max_blocks, block_mask=perattn_pattern)

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        ## perceiver attention
        self.act_emb = SimpleActionEncoder(
            num_agents=num_agents, action_dim=action_dim, is_continuous_act=not self.is_discrete_action, output_dim=config.embed_dim,
            # num_heads=8, embed_dim=256, attn_dropout=0.1, ff_dropout=0.1, depth=3
            num_heads=8, embed_dim=256, attn_dropout=0., ff_dropout=0., depth=3  # 这里可能dropout为0.，会好一点
        )
        ## --------------------

        self.embedder = GeneralEmbedder(
            max_blocks=config.max_blocks,
            block_masks=[obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Sequential(
                nn.Linear(state_dim, config.embed_dim),
                nn.LayerNorm(config.embed_dim),
                nn.SiLU(),
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.LayerNorm(config.embed_dim),
                nn.SiLU(),
                nn.Linear(config.embed_dim, config.embed_dim),
            )]),
            embedding_dim=config.embed_dim
        )
        self.act_emb_slicer = Slicer(max_blocks=config.max_blocks, block_mask=act_tokens_pattern)

        ## 加入适合dense的reward predictor
        self.use_symlog = use_symlog  # whether to use symlog transformation
        self.use_ce_for_reward = use_ce_for_reward
        if use_ce_for_reward:
            print("Use cross-entropy to train the prediction of reward...")
        else:
            print("Use SmoothL1Loss to train the prediction of reward...")

        if not self.use_ce_for_reward:
            self.head_rewards = Head(
                max_blocks=config.max_blocks,
                block_mask=all_but_last_pattern,
                head_module=nn.Sequential(
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, 1),
                )
            )

        else:
            assert rewards_prediction_config is not None
            self.use_symlog = True
            bin_width = (rewards_prediction_config["max_v"] - rewards_prediction_config["min_v"]) / rewards_prediction_config["bins"]
            self.reward_loss = LOSSES_DICT[rewards_prediction_config["loss_type"]](
                min_value = rewards_prediction_config["min_v"],
                max_value = rewards_prediction_config["max_v"],
                num_bins = rewards_prediction_config["bins"],
                sigma = bin_width * 0.75
            )
            print(f'Use {self.reward_loss} for discrete labels...')

            self.head_rewards = Head(
                max_blocks=config.max_blocks,
                block_mask=all_but_last_pattern,
                head_module=nn.Sequential(
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, config.embed_dim),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim, self.reward_loss.output_dim),
                )
            )
        

        self.use_ce_for_end = use_ce_for_end
        if use_ce_for_end:
            print("Use cross-entropy to train the prediction of continuation...")
        else:
            print("Use log-prob to train the prediction of continuation...")

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2 if use_ce_for_end else 1),
            )
        )

        self.action_dim = action_dim
        self.enable_av_pred = enable_av_pred
        self.pred_av_action = enable_av_pred  # for the world model env
        self.use_ce_for_av_action = use_ce_for_av_action
        ## predict the avail action at next timestep (not current timestep)
        if self.enable_av_pred:
            if use_ce_for_av_action:
                print("Use cross-entropy to train the prediction of av_action...")
            else:
                print("Use log-prob to train the prediction of av_action...")

        else:
            print("Disable the prediction of av_action...")

        if self.enable_av_pred:
            if not self.use_ce_for_av_action:
                self.heads_avail_actions = Head(
                    max_blocks=config.max_blocks,
                    block_mask=all_but_last_pattern,
                    head_module=nn.Sequential(
                        nn.Linear(config.embed_dim, config.embed_dim),
                        nn.ReLU(),
                        nn.Linear(config.embed_dim, config.embed_dim),
                        nn.ReLU(),
                        nn.Linear(config.embed_dim, action_dim),
                    )
                )
            
            else:
                self.heads_avail_actions = AgentWiseHead(
                    max_blocks=config.max_blocks,
                    block_mask=all_but_last_pattern,
                    head_module=DiscreteDist(
                        config.embed_dim, self.act_vocab_size, 2, 256
                    ),
                    num_agents=self.num_agents,
                    embed_dim = config.embed_dim,
                )

        self.apply(init_weights)
        
        if self.use_symlog:
            print("Enable `symlog` to transform the reward targets...")
        else:
            print("Disable `symlog` to transform...")


    def forward(self, tokens: torch.LongTensor, perattn_out: torch.Tensor = None,
                past_keys_values: Optional[KeysValues] = None, return_attn: bool = False, attention_mask: torch.Tensor = None) -> TransRewEndModelOutput:
        bs = tokens.size(0)
        num_steps = tokens.size(1)  # (B, T)

        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        
        sequences = self.embedder(tokens, num_steps, prev_steps)

        indices = self.act_emb_slicer.compute_slice(num_steps, prev_steps)
        if perattn_out is not None:
            assert len(indices) != 0
            sequences[:, indices] = perattn_out
        else:
            assert len(indices) == 0

        sequences += self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        x, attn_output = self.transformer(sequences,
                                          past_keys_values,
                                          return_attn = return_attn,
                                          attention_mask = attention_mask)

        pred_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)

        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        logits_avail_action = self.heads_avail_actions(x, num_steps=num_steps, prev_steps=prev_steps) if self.enable_av_pred else None

        return TransRewEndModelOutput(x, pred_rewards, logits_ends, logits_avail_action, attn_output=attn_output)

    def get_act_emb(self, action):
        if self.is_discrete_action:
            action = action.argmax(-1)

        return self.act_emb(action)

    def compute_loss(self,
                     batch,
                     attention_mask: torch.Tensor = None,
                     **kwargs):
        device = batch['shared_obs'].device

        # batch['shared_obs'] = batch['shared_obs'].mean(dim=2)
        
        b, l, n = batch['action'].shape[:3]
        act = rearrange(batch['action'], 'b l n e -> (b l) n e')
        if self.is_discrete_action:
            act = act.argmax(-1)

        act_cond = self.act_emb(act)
        act_cond = rearrange(act_cond, '(b l) e -> b l e', b = b, l = l)
        tokens = torch.stack([batch['shared_obs'].clone(), torch.empty(b, l, batch['shared_obs'].size(-1), device=act_cond.device, dtype=act_cond.dtype)], dim=-2)            
        tokens = rearrange(tokens, 'b l m e -> b (l m) e')  # (B, L(K+N))

        outputs = self(tokens, perattn_out = act_cond, attention_mask = attention_mask)

        ### compute discount loss
        if not self.use_ce_for_end:
            pred_ends = td.independent.Independent(td.Bernoulli(logits=outputs.logits_ends), 1)
            loss_ends = -torch.mean(pred_ends.log_prob((1. - batch['done'].all(-2).to(torch.float32))))
        else:
            logits_ends = rearrange(outputs.logits_ends, 'b l e -> (b l) e')
            labels_ends = rearrange(batch['done'].all(-2), 'b l 1 -> (b l)').to(torch.long)
            loss_ends = F.cross_entropy(logits_ends, labels_ends)

        ### compute reward loss
        labels_rewards = batch['reward'].mean(-2)
        if self.use_symlog:
            labels_rewards = symlog(labels_rewards)
        
        if self.use_ce_for_reward:
            labels_rewards = rearrange(labels_rewards, 'b l 1 -> (b l 1)')
            logits_rewards = rearrange(outputs.pred_rewards, 'b l e -> (b l) e')
            loss_rewards = self.reward_loss(logits_rewards, labels_rewards)

        else:
            loss_rewards = F.smooth_l1_loss(outputs.pred_rewards, labels_rewards)

        ### compute av_action loss
        if self.enable_av_pred:
            tmp = torch.roll(batch['done'].all(-2), 1, dims=1).squeeze(-1)
            labels_av_actions = batch['av_action']
            # labels_av_actions[tmp == True] = F.one_hot(torch.tensor(1), num_classes=self.action_dim).to(device, dtype=labels_av_actions.dtype).expand_as(labels_av_actions[tmp == True])  # torch.zeros_like(labels_av_actions[tmp == True], device=device)
            labels_av_actions[tmp == True] = torch.ones_like(labels_av_actions[tmp == True])

            ## for cross-entropy loss
            if self.use_ce_for_av_action:
                logits_av_actions = rearrange(outputs.pred_avail_action[:, :-1], 'b l n a e -> (b l n a) e')
                labels_av_actions = labels_av_actions[:, 1:].reshape(-1,).to(torch.long)
                loss_av_actions = F.cross_entropy(logits_av_actions, labels_av_actions)
            
            else:
                raise NotImplementedError
                pred_av_actions = td.independent.Independent(td.Bernoulli(logits=outputs.pred_avail_action[:, :-1]), 1)
                labels_av_actions = rearrange(labels_av_actions, 'b l n e -> (b n) l e')
                loss_av_actions = -torch.mean(pred_av_actions.log_prob(labels_av_actions[:, 1:]))
        else:
            loss_av_actions = 0.

        info_loss = 0.

        loss = loss_ends + loss_rewards + loss_av_actions + info_loss

        loss_dict = {
            'loss_rew': loss_rewards.item(),
            'loss_end': loss_ends.item(),
            'loss_av_action': loss_av_actions.item() if self.enable_av_pred else 0.,
            'info_loss': 0.,
            'loss_total': loss.item(),
        }

        return loss, loss_dict
        # return LossWithIntermediateLosses(**loss_dict)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, filled: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(filled)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).unsqueeze(-1).expand_as(obs_tokens), -100).transpose(1, 2), 'b n l k -> (b n) (l k)')[:, 1:]
        
        labels_rewards = rewards.masked_fill(mask_fill.unsqueeze(-1).unsqueeze(-1).expand_as(rewards), 0.)

        labels_ends = ends.masked_fill(mask_fill.unsqueeze(-1).unsqueeze(-1).expand_as(ends), 1.).to(torch.long)
        
        return labels_observations.reshape(-1), labels_rewards, labels_ends
    
    def compute_labels_world_model_all_valid(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        labels_observations = rearrange(obs_tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')[:, 1:]
        labels_rewards = rearrange(rewards.transpose(1, 2), 'b n l 1 -> (b n) l 1')
        labels_ends = rearrange(ends.transpose(1, 2), 'b n l 1 -> (b n) l 1')
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1).to(torch.long)
    
    
    def get_perceiver_attn_out(self, obs_tokens, actions):
        device = obs_tokens.device
        shape = obs_tokens.shape
        
        obs_encodings = self.embedder.embedding_tables[1](obs_tokens)
        action_encodings = self.embedder.embedding_tables[0](actions)
        input_encodings = torch.cat([obs_encodings, action_encodings], dim=-2)

        n, m, e = input_encodings.shape[-3:]
        input_encodings = rearrange(input_encodings, '... n m e -> (...) (n m) e')
        agent_id_emb = repeat(self.agent_id_pos_emb[:, :n], '1 n e -> b (n m) e', b = input_encodings.size(0), n = n, m = m)

        input_encodings += agent_id_emb.detach().to(device)
        perattn_out = self.perattn(input_encodings)

        perattn_out = perattn_out.reshape(*shape[:-1], -1)
        return perattn_out
    
    def get_perceiver_cross_attn_w(self, obs_tokens, actions):
        device = obs_tokens.device
        shape = obs_tokens.shape
        
        obs_encodings = self.embedder.embedding_tables[1](obs_tokens)
        action_encodings = self.embedder.embedding_tables[0](actions)
        input_encodings = torch.cat([obs_encodings, action_encodings], dim=-2)

        n, m, e = input_encodings.shape[-3:]
        input_encodings = rearrange(input_encodings, '... n m e -> (...) (n m) e')
        agent_id_emb = repeat(self.agent_id_pos_emb[:, :n], '1 n e -> b (n m) e', b = input_encodings.size(0), n = n, m = m)

        input_encodings += agent_id_emb.detach().to(device)
        perattn_out, cross_attn_w = self.perattn(input_encodings, return_cross_attn = True)

        perattn_out = perattn_out.reshape(*shape[:-1], -1)
        return perattn_out, cross_attn_w
    
    
    ### visualize attention map
    @torch.no_grad()
    def visualize_attn(self, sample, tokenizer, save_dir):
        # preliminary
        device = sample["observation"].device
        n_agents = sample['observation'].shape[-2]
        horizon = sample['observation'].shape[-3]
        obs_token_indices = rearrange(repeat(self.obs_tokens_pattern, 'n -> h n', h=horizon), 'h n -> (h n)')
        obs_token_indices = (obs_token_indices == 1).nonzero().squeeze().numpy()
        act_token_indices = rearrange(repeat(self.act_tokens_pattern, 'n -> h n', h=horizon), 'h n -> (h n)')
        act_token_indices = (act_token_indices == 1).nonzero().squeeze().numpy()
        perattn_indices = rearrange(repeat(self.perattn_pattern, 'n -> h n', h=horizon), 'h n -> (h n)')
        perattn_indices = (perattn_indices == 1).nonzero().squeeze().numpy()
        
        save_dir.mkdir(parents=True, exist_ok=True)
        for agent_id in range(n_agents):
            tmp_dir = save_dir / f"agent_{agent_id}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            
        for horizon_idx in range(horizon):
            tmp_dir = save_dir / f"horizon_{horizon_idx}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
        
        _, obs_tokens = tokenizer.encode(sample['observation'], should_preprocess=True)
        obs_tokens = obs_tokens.to(torch.long)
        act_tokens = torch.argmax(sample['action'], dim=-1, keepdim=True)
        
        perattn_out = self.get_perceiver_attn_out(obs_tokens, act_tokens)
        b, l, n, e = perattn_out.shape
        perattn_out = rearrange(perattn_out, 'b l n e -> (b n) l e', b=b, l=l, n=n)
        
        tokens = torch.cat([obs_tokens, act_tokens, torch.empty_like(act_tokens, device=device, dtype=torch.long)], dim=-1)
        tokens = rearrange(tokens.transpose(1, 2), 'b n l k -> (b n) (l k)')  # (B, L(K+N))

        outputs = self(tokens, perattn_out = perattn_out, return_attn=True)
        
        ### visualize perceiver_cross_attn
        _, cross_attn_weight = self.get_perceiver_cross_attn_w(obs_tokens, act_tokens)
        cross_attn_weight = cross_attn_weight.cpu().numpy()
        
        attn_output = outputs.attn_output
        
        # define custom cmap
        # modality_colors = ["Blues", "Reds", "Oranges"]
        modality_colors = ["Blues", "Reds", "YlOrBr"]
        # modality_colors = ['#8DC7E3', '#FF988C', '#FFC995']
        colors = []
        for color in modality_colors:
            cmap = mpl.colormaps[color]
            colors.append(
                cmap(np.linspace(0., 1., 333))
            )
            
        white_cmap = LinearSegmentedColormap.from_list("white", [(0., 'white'), (1., 'white')], N=1)
        colors.append(
            white_cmap(np.linspace(0., 1., 1))
        )
            
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", np.vstack(colors))
        red_cmap = mpl.colormaps["Oranges"]
        
        def save_matrix_as_image(matrix, filename, custom_cmap):
            plt.imshow(matrix, cmap=custom_cmap, vmin=0, vmax=1)
            # plt.colorbar(orientation="horizontal")
            
            plt.axis("off")
            
            # indices = np.tril_indices_from(matrix)

            # min_row, max_row = min(indices[0]), max(indices[0])
            # min_col, max_col = min(indices[1]), max(indices[1])

            # vertices = [(min_col - 0.5, min_row - 0.5), (max_col + 0.5, min_row - 0.5), (max_col + 0.5, max_row + 0.5)]

            # triangle = plt.Polygon(vertices, edgecolor='black', linewidth=2, fill=None)
            # plt.gca().add_patch(triangle)
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, dpi=600)
            
            plt.close()
        
        import seaborn as sns
        import pandas as pd
        import matplotlib.patches as patches
        
        square_size = 20

        fig_width = cross_attn_weight.shape[-1] * square_size / 100
        fig_height = cross_attn_weight.shape[-2] * square_size / 100
        
        for horizon_idx in range(cross_attn_weight.shape[0]):
            for head_id in range(cross_attn_weight.shape[1]):
                matrix = cross_attn_weight[horizon_idx, head_id]
                
                df = pd.DataFrame(matrix, index=[None for i in range(1, n_agents + 1)])

                df.columns = [None] * len(df.columns)
                
                # plt.imshow(matrix, cmap='viridis')
                fig = plt.figure(figsize=(fig_width, fig_height)); heatmap = sns.heatmap(df, vmin=-0.05, vmax=1.05, cmap=sns.cubehelix_palette(as_cmap=True), square=True, cbar_kws={'aspect': 5})
                
                heatmap.set_xticks(np.arange(0.5, len(matrix[0]), 1))
                heatmap.set_yticks(np.arange(0.5, len(matrix), 1))
                
                plt.gca().patch.set_edgecolor('black'); plt.gca().patch.set_linewidth('1')
                
                cax = plt.gcf().axes[-1]; cax.set_frame_on(True); cax.patch.set_edgecolor('black'); cax.patch.set_linewidth('1') # cax.add_patch(patches.Rectangle((-0.05, -0.05), 1.05, 1.05, fill=False, edgecolor='black', linewidth=2))
                
                plt.savefig(save_dir / f"horizon_{horizon_idx}" / f"cross_attn_head{head_id}.png",
                            bbox_inches="tight", pad_inches=0.1, dpi=600)
                plt.close()
        
        ## save as image
        scale = 0.332
        for layer_id in range(len(attn_output)):
            attn_weight = attn_output[layer_id].cpu().numpy()
            attn_weight[:, :, obs_token_indices] *= scale
            
            attn_weight[:, :, act_token_indices] *= scale
            attn_weight[:, :, act_token_indices] += 0.3335
            
            attn_weight[:, :, perattn_indices] *= scale
            attn_weight[:, :, perattn_indices] += 0.6665
            
            attn_weight = np.where(np.tril(np.ones_like(attn_weight)) == 1, attn_weight, np.zeros_like(attn_weight) + 0.9995)
            
            for agent_id in range(attn_weight.shape[0]):
                for head_id in range(attn_weight.shape[1]):
                    save_matrix_as_image(attn_weight[agent_id, head_id],
                                         save_dir / f"agent_{agent_id}" / f"layer{layer_id}_head{head_id}.png",
                                         custom_cmap)
        
        print(f"Attention visualization has been saved to {str(save_dir)}.")