import math
import numpy as np
from torch import nn
import torch
import torchvision
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


###############################################################################
#
# Building blocks for transformers
#
###############################################################################


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.output_layer(out)


class TransformerFeedForwardNN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # Remember the residual connection
        layers = [
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, input_size, inv_freq_factor=10, factor_ratio=None):
        super().__init__()
        self.input_size = input_size
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_size
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (
            self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.0
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)

    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape

    def output_size(self, input_size):
        return input_size


###############################################################################
#
# Transformer Decoder (we only use transformer decoder for our policies)
#
###############################################################################



class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers,
        num_heads,
        head_output_size,
        mlp_hidden_size,
        dropout,
        **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        Attention(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                        ),
                        Norm(input_size),
                        TransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout
                        ),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
            (self.num_elements is None)
            or (self.seq_len is None)
            or (self.num_elements != input_shape[2])
            or (self.seq_len != input_shape[1])
        ):

            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                torch.triu(torch.ones(self.seq_len, self.seq_len))
                - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

    def forward(self, x, mask=None):
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(att(att_norm(x), mask))
            elif self.mask is not None:
                x = x + drop_path(att(att_norm(x), self.mask))
            else:  # no masking, just use full attention
                x = x + drop_path(att(att_norm(x)))

            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class Expert(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 4 * in_features),
            nn.ReLU(),
            nn.Linear(4 * in_features, in_features),
            nn.Dropout(0.1)
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(in_features, 4 * in_features),
        #     nn.GELU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(4 * in_features, in_features),
        #     nn.Dropout(0.1),
        # )

    def forward(self, x):
        return self.fc(x)

class SparseMoE2(nn.Module):

    def __init__(self, in_features, num_experts, top_k, is_multigate):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.is_multigate = is_multigate
        if self.is_multigate:
            self.gates = nn.ModuleList([nn.Linear(in_features, num_experts) for _ in range(10)])
        else:
            self.gate = nn.Linear(in_features, num_experts)
        self.noise_linear = nn.Linear(in_features, num_experts)

        self.experts = nn.ModuleList([Expert(in_features) for _ in range(self.num_experts)])
        self.shared_expert = Expert(in_features)

        self.experts_counts = torch.zeros(num_experts, dtype=torch.long)

    def forward(self, hidden_states, task_id=None):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        if self.is_multigate:
            gate_outputs = torch.stack([gate(hidden_states) for gate in self.gates], dim=1)
            router_logits = gate_outputs[torch.arange(batch_size), task_id]
        else:
            router_logits = self.gate(hidden_states)    # (bs, seq, num_expert)
        # noise_logits = self.noise_linear(hidden_states)
        #
        # # Adding scaled unit gaussian noise to the logits
        # noise = torch.randn_like(router_logits) * F.softplus(noise_logits)
        # router_logits = router_logits + noise


        hidden_states = hidden_states.view(-1, hidden_dim)  # bs*seq, hidden_dim
        routing_weights = F.softmax(router_logits, dim=-1)  # (bs, seq, num_expert)
        routing_weights = routing_weights.view(-1, self.num_experts)     # (bs*seq, num_expert)

        # 计算每个专家的概率
        prob_per_expert = routing_weights.mean(dim=0)  # P_i

        # select top2 experts
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)     # (bs*seq, 2)

        # calculate experts used num
        if not self.training:
            flattened = selected_experts.flatten()
            expert_count = torch.bincount(flattened, minlength=self.num_experts)
            self.experts_counts += expert_count.cpu()

        # fusing weight && add
        routing_weights = routing_weights / torch.sum(routing_weights, dim=-1, keepdim=True).to(hidden_states.dtype)
        #  init maxtrix to save result
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # for efficiency, calculate the result one time using the mask
        expert_mask = nn.functional.one_hot(selected_experts, num_classes=self.num_experts)     # (bs*seq, 2, num_expert)

        tokens_per_expert = expert_mask.sum(dim=(0, 1)).float()
        tokens_per_expert = tokens_per_expert / (routing_weights.size(0) * self.top_k)  # f_i

        # [20,2,8] ---> [8,2,20]
        expert_mask = expert_mask.permute(2, 1, 0)  # (num_expert, 2, bs*seq)
        for expert_index in range(self.num_experts):
            expert_layer = self.experts[expert_index]
            idx, top_x = torch.where(expert_mask[expert_index])     # (3,)
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)     # (3, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]   # (3, hidden_dim)
            current_hidden_states = current_hidden_states.to(hidden_states.dtype)

            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        final_hidden_states += self.shared_expert(hidden_states)

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)


        # 计算负载均衡损失
        load_balance_loss = self.num_experts * torch.sum(prob_per_expert * tokens_per_expert)

        return final_hidden_states, load_balance_loss

class MoeTransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers,
        num_heads,
        head_output_size,
        num_experts,
        top_k,
        dropout,
        is_multigate=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}
        self.is_multigate = is_multigate

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        Attention(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                        ),
                        Norm(input_size),
                        SparseMoE2(input_size, num_experts=num_experts, top_k=top_k, is_multigate=is_multigate),
                        # MoE(input_size, num_experts=num_experts, top_k=top_k),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
            (self.num_elements is None)
            or (self.seq_len is None)
            or (self.num_elements != input_shape[2])
            or (self.seq_len != input_shape[1])
        ):

            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                torch.triu(torch.ones(self.seq_len, self.seq_len))
                - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

    def forward(self, x, task_id=None, mask=None):
        aux_losses = 0.0
        for layer_idx, (att_norm, att, ff_norm, moe) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(att(att_norm(x), mask))
            elif self.mask is not None:
                x = x + drop_path(att(att_norm(x), self.mask))
            else:  # no masking, just use full attention
                x = x + drop_path(att(att_norm(x)))

            if not self.training:
                self.attention_output[layer_idx] = att.att_weights

            out, loss = moe(ff_norm(x), task_id)
            x = x + self.drop_path(out)
            aux_losses += loss
        # aux_losses /= len(self.layers)
        return x, aux_losses

    @property
    def device(self):
        return next(self.parameters()).device