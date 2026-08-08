import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
# from torch_utils import persistence
from einops.layers.torch import Rearrange
from functools import partial
from typing import List, Optional
import einops
import math


# Reimplementation of the https://github.com/anuragajay/decision-diffuser/blob/main/code/diffuser/models/temporal.py.
# "Is Conditional Generative Modeling all you need for Decision-Making?"


# @persistence.persistent_class
class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# convs
Conv1x1 = partial(nn.Conv1d, kernel_size=1, stride=1, padding=0)
Conv3x1 = partial(nn.Conv1d, kernel_size=3, stride=1, padding=1)


# GroupNorm and conditional GroupNorm
## Settings for GroupNorm
GN_GROUP_SIZE = 32
GN_EPS = 1e-5

class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)           # Decision Diffuser 使用的是 n_groups = 8
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)
    
class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)      # Decision Diffuser 使用的是 n_groups = 8
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


# Embedding of the noise level
class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: Tensor) -> Tensor:
        assert input.ndim == 1
        f = 2 * math.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


# [Down|Up]sampling
class Downsample1d(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)
    
class Upsample1d(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

## A upsample module that resembles the one in DIAMOND
class Upsample1d_diamond(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")  # 这一行可能有问题，请仔细check x.shape
        return self.conv(x)
    

# Residual block (conditioning with AdaGroupNorm, no [down|up]sampling)
#### NOTE that this part is akin to the implementation in DIAMOND
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, attn: bool) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x1(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x1(out_channels, out_channels)
        # self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r
        # x = self.attn(x)
        return x


# Sequence of residual blocks (in_channels -> mid_channels -> ... -> mid_channels -> out_channels)
class ResBlocks(nn.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_ch, out_ch, cond_channels, attn)
                for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
            ]
        )

    def forward(self, x: Tensor, cond: Tensor, to_cat: Optional[List[Tensor]] = None) -> Tensor:
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x, cond)
            outputs.append(x)
        return x, outputs


# UNet1D
class UNet1D(nn.Module):
    def __init__(self,
                 cond_channels: int,
                 depths: List[int],
                 channels: List[int],
                 attn_depths: List[int]) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self._num_down = len(channels) - 1

        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):          # depths 每一个表明对应一次upsampling或者downsampling内这个resblocks有多少个block
            c1 = channels[max(0, i - 1)]        # input_channels
            c2 = channels[i]                    # input_channels
            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
            u_blocks.append(
                ResBlocks(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))

        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=False,  # 这里原来是True
        )

        downsamples = [nn.Identity()] + [Downsample1d(c) for c in channels[:-1]]
        upsamples = [nn.Identity()] + [Upsample1d(c) for c in reversed(channels[:-1])]
        self.downsamples = nn.ModuleList(downsamples)
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:        
        # *_, t = x.size()
        # n = self._num_down
        # padding_t = math.ceil(t / 2 ** n) * 2 ** n - t
        # padding_h = math.ceil(h / 2 ** n) * 2 ** n - h
        # padding_w = math.ceil(w / 2 ** n) * 2 ** n - w
        # x = F.pad(x, (0, padding_w, 0, padding_h))

        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append((x_down, *block_outputs))

        x, _ = self.mid_blocks(x, cond)
        
        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, block_outputs = block(x_up, cond, skip[::-1])
            u_outputs.append((x_up, *block_outputs))

        # x = x[..., :t]
        return x, d_outputs, u_outputs


### Decision Diffuser Implementation
class Conv1dBlock(torch.nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, mish=True, n_groups=8):
        super().__init__()

        if mish:
            act_fn = torch.nn.Mish()
        else:
            act_fn = torch.nn.SiLU()

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            torch.nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            act_fn,
        )

    def forward(self, x):
        return self.block(x)


# @persistence.persistent_class
class ResidualTemporalBlock(torch.nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, mish=True):
        super().__init__()

        self.blocks = torch.nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, mish),
            Conv1dBlock(out_channels, out_channels, kernel_size, mish),
        ])

        if mish:
            act_fn = torch.nn.Mish()
        else:
            act_fn = torch.nn.SiLU()

        self.time_mlp = torch.nn.Sequential(
            act_fn,
            torch.nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = torch.nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else torch.nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)
    

# @persistence.persistent_class
class TemporalUnet1D(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        cond_dim,
        horizon=100,
        dim=128,
        dim_mults=(1, 4, 8),
        returns_condition=False,
        condition_dropout=0.1,
        calc_energy=False,
        kernel_size=5,
    ):
        super().__init__()

        dims = [state_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        if calc_energy:
            mish = False
            act_fn = torch.nn.SiLU()
        else:
            mish = True
            act_fn = torch.nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPosEmb(dim),
            torch.nn.Linear(dim, dim * 4),
            act_fn,
            torch.nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        if self.returns_condition:
            self.returns_mlp = torch.nn.Sequential(
                        torch.nn.Linear(1, dim),
                        act_fn,
                        torch.nn.Linear(dim, dim * 4),
                        act_fn,
                        torch.nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = torch.distributions.Bernoulli(probs=1-self.condition_dropout)
            embed_dim = 2*dim
        else:
            embed_dim = dim

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(torch.nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Downsample1d(dim_out) if not is_last else torch.nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(torch.nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, mish=mish),
                Upsample1d(dim_in) if not is_last else torch.nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = torch.nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, mish=mish),
            torch.nn.Conv1d(dim, state_dim, 1),
        )

    # TODO: time -> cond
    # example: cond -> (n_agents * act_dim)
    # according to time, generate available cond mask
    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        if self.calc_energy:
            x_inp = x

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask * returns_embed
            if force_dropout:
                returns_embed = 0 * returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        if self.calc_energy:
            # Energy function
            energy = ((x - x_inp)**2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]
        else:
            return x

    def get_pred(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x


# Chi's implementation of Unet1d
class GroupNorm1d(nn.Module):
    def __init__(self, dim, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, dim // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x.unsqueeze(2),
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x.squeeze(2)

class ChiResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, emb_dim: int, kernel_size: int = 3, cond_predict_scale: bool = False):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            GroupNorm1d(out_dim, 8, 4), nn.Mish())
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            GroupNorm1d(out_dim, 8, 4), nn.Mish())

        cond_dim = 2 * out_dim if cond_predict_scale else out_dim
        self.cond_predict_scale = cond_predict_scale
        self.out_dim = out_dim
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(emb_dim, cond_dim))

        self.residual_conv = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, emb):
        out = self.conv1(x)
        embed = self.cond_encoder(emb)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_dim, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed.unsqueeze(-1)
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out