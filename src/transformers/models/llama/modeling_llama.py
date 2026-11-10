# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, StaticCache
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_llama import LlamaConfig
from dynamic_prune.model.pruning_module import HiddenRepresentationPruning
from dynamic_prune.prune_context import prune_context as cfg
from .attentionmask import AttentionMaskConverter, _prepare_4d_causal_attention_mask

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

import torch.cuda.nvtx as nvtx

# prune stream
prune_stream = torch.cuda.Stream()

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)

    
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, num_heads, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )



class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor


        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]
        
        # Pruning support
        self.layer_idx = layer_idx
        self.pruning_module = HiddenRepresentationPruning(cfg, f"llama_mlp_{layer_idx}", config)

        self.selected_out_dim_indices = None

        self._weight_squared_cached = None

    # ================= CUDA Graph Compatible Methods =================

    def cache_weight_squared(self):
        """
        Pre-cache weight norms for metric calculation. Call once after model loading.

        Mathematical optimization
        Original computation: metric[j] = ||norm_val[j] * weight[:, j]||_2
                           = norm_val[j] * ||weight[:, j]||_2

        Only cache ||weight[:, j]||_2 weight 

        Memory savings
        - Original: [4096, 11008] = 90 MB/
        - Optimized: [11008] = 22 KB/
        - Savings: 99.97%
        """
        weight = self.down_proj.weight.data  # [hidden_size, intermediate_size] = [4096, 11008]
        # Only transpose if explicitly configured (matches original prune logic)
        if cfg.config.transpose_down_linear:
            weight = weight.t()

        weight_sq = weight.pow(2)
        self._weight_sq_l2_norm = torch.linalg.vector_norm(weight_sq, ord=2, dim=0).contiguous()  # [intermediate_size]

    def dense_forward(
        self,
        x: torch.Tensor,
        mlp_buffer: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Dense forward pass - no branches, no cfg access, CUDA Graph compatible.

        Args:
            x: [batch, seq_len, hidden_size]
            mlp_buffer: Optional [1, seq_len, intermediate_size] buffer for metric computation
                        If provided, computation is done in-place to avoid copy overhead.

        Returns:
            [batch, seq_len, hidden_size] output tensor
        """
        gate_out = self.act_fn(self.gate_proj(x))
        up_out = self.up_proj(x)

        # In-place computation to buffer (avoid copy overhead)
        if mlp_buffer is not None:
            torch.mul(gate_out, up_out, out=mlp_buffer)
            x_middle = mlp_buffer
        else:
            x_middle = gate_out * up_out

        return self.down_proj(x_middle)

    def sparse_forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Sparse forward pass - uses external indices, CUDA Graph compatible.
        
        Optimized: Uses fused Gate+Up+SiLU kernel for 2.06x speedup.

        Args:
            x: [batch, seq_len, hidden_size]
            indices: [keep_dims] pre-computed dimension indices from scheduler

        Returns:
            [batch, seq_len, hidden_size] output tensor
        """
        # Fused Gate+Up+SiLU kernel: ~2x faster than separate calls
        if cfg.config.use_fused_kernel: 
            from dynamic_prune.model.triton_linear import fused_gate_up_silu_sparse
            x_middle = fused_gate_up_silu_sparse(
                x,
                self.gate_proj.weight,
                self.up_proj.weight,
                indices
            )
            return self.down_proj(x_middle, in_dim_indices=indices)
        else: 
            down_proj = self.down_proj(
                self.act_fn(
                    self.gate_proj(
                        x, 
                        out_dim_indices=indices
                    )
                ) 
                * self.up_proj(
                    x, 
                    out_dim_indices=indices
                ), 
                in_dim_indices=indices
            )
            return down_proj

    @staticmethod
    def sort_metric_static(
        metric: torch.Tensor,
        prune_ratio: float,
        block_size: int = 64
    ) -> torch.Tensor:
        """
        Sort metric and return indices to keep.

        Args:
            metric: [intermediate_size] importance scores
            prune_ratio: ratio of dimensions to prune
            block_size: alignment for tensor core efficiency

        Returns:
            [keep_dims] sorted indices of dimensions to keep
        """
        num_total = metric.shape[0]
        num_prune = int(prune_ratio * num_total)
        # Round up to block_size
        num_prune = ((num_prune + block_size - 1) // block_size) * block_size
        # Ensure we keep at least block_size dims
        num_prune = min(num_prune, num_total - block_size)
        num_keep = num_total - num_prune

        # topk: O(n log k) vs argsort: O(n log n)
        topk_indices = torch.topk(metric, num_keep, largest=True, sorted=False).indices
        mask = torch.zeros(num_total, dtype=torch.bool, device=metric.device)
        mask.scatter_(0, topk_indices, True)
        return mask.nonzero(as_tuple=False).squeeze(-1)

    def compute_pruning_selection(self, x, block_size=16, reduction="mean", **kwargs):
        intermediate_activations = self.act_fn(self.gate_proj(x, cal_mlp_metric_scores=True)) * \
                    self.up_proj(x, cal_mlp_metric_scores=True)

        metric_scores = self.pruning_module.cal_mlp_prune_metric(
            intermediate_activations,
            self.down_proj.weight.data.t() if cfg.config.transpose_down_linear else self.down_proj.weight.data, 
            global_metric_score_distribution=self.down_proj.get_global_metric_score_distribution(),
        )
        selected_out_dim_indices, _ = self.pruning_module.sort_mlp_metric(
            metric_scores, cfg.derived.tc_multiple
        )

        return selected_out_dim_indices, intermediate_activations

    def prune_post_process(self, intermediate_activations):
        metric_scores = self.pruning_module.cal_mlp_prune_metric(
            intermediate_activations,
            self.down_proj.weight.data.t() if cfg.config.transpose_down_linear else self.down_proj.weight.data, 
            global_metric_score_distribution=self.down_proj.get_global_metric_score_distribution(),
        )
        selected_out_dim_indices, _ = self.pruning_module.sort_mlp_metric(
            metric_scores, cfg.derived.tc_multiple
        )
        
        cfg.stats.real_prune_ratio = len(selected_out_dim_indices) / 11008
        return selected_out_dim_indices
    
    def update_pruning_selection(self, intermediate_activations):
        selected_out_dim_indices = self.prune_post_process(intermediate_activations)
        self.selected_out_dim_indices = selected_out_dim_indices
    
    def forward(self, x, **kwargs):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # Baseline inference does not install dynamic-pruning Linear
            # modules, so standard nn.Linear must receive no pruning kwargs.
            if not cfg.config.prune_flag:
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            if cfg.state.calibration_stage:
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                return down_proj
            else:
                mlp_targets_enabled = any(
                    module in cfg.config.cust_tgt_modules
                    for module in ("down_proj", "up_proj", "gate_proj")
                )
                if mlp_targets_enabled and self.layer_idx not in cfg.derived.skip_layers:
                    # dynamic prune
                    if cfg.config.mode == "dynamic": 
                        # prune
                        if not cfg.config.async_prune:
                            if cfg.state.is_prune:
                                intermediate_activations = self.act_fn(self.gate_proj(x, cal_mlp_metric_scores=True)) * self.up_proj(x, cal_mlp_metric_scores=True)
                                down_proj = self.down_proj(intermediate_activations)
                                
                                self.update_pruning_selection(intermediate_activations)
                                return down_proj
                            
                            down_proj = self.down_proj(
                                self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                            ) if self.selected_out_dim_indices is None else self.down_proj(
                                self.act_fn(
                                    self.gate_proj(
                                        x, 
                                        out_dim_indices=self.selected_out_dim_indices
                                    )
                                ) 
                                * self.up_proj(
                                    x, 
                                    out_dim_indices=self.selected_out_dim_indices
                                ), 
                                in_dim_indices=self.selected_out_dim_indices
                            )
                            
                            return down_proj
                        else:
                            if cfg.state.is_prune:
                                # N-Batch mode
                                mlp_buffer = kwargs.get('mlp_buffer', None)
                                buffer_idx = kwargs.get('buffer_idx', None)
                                
                                if mlp_buffer is not None and buffer_idx is not None:
                                    gate_out = self.act_fn(self.gate_proj(x, cal_mlp_metric_scores=True))
                                    up_out = self.up_proj(x, cal_mlp_metric_scores=True)

                                    torch.mul(gate_out.squeeze(0), up_out.squeeze(0), out=mlp_buffer[buffer_idx])
                                    intermediate_activations = mlp_buffer[buffer_idx].unsqueeze(0)

                                    down_proj = self.down_proj(intermediate_activations)
                                    return down_proj
                                else:
                                    intermediate_activations = self.act_fn(self.gate_proj(x, cal_mlp_metric_scores=True)) * self.up_proj(x, cal_mlp_metric_scores=True)
                                    down_proj = self.down_proj(intermediate_activations)

                                    with torch.cuda.stream(prune_stream):
                                        self.update_pruning_selection(intermediate_activations)
                                    return down_proj

                            if self.selected_out_dim_indices is None:
                                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

                            down_proj = self.down_proj(
                                self.act_fn(
                                    self.gate_proj(
                                        x, 
                                        out_dim_indices=self.selected_out_dim_indices
                                    )
                                ) 
                                * self.up_proj(
                                    x, 
                                    out_dim_indices=self.selected_out_dim_indices
                                ), 
                                in_dim_indices=self.selected_out_dim_indices
                            )
                            
                            return down_proj
                else:
                    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
                    return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # self.head_dim = self.hidden_size // self.num_heads
        self.head_dim = 128
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        # Pruning support
        self.custom_duration = 0
        self.pruning_module = HiddenRepresentationPruning(cfg, f'llama_attention_{layer_idx}', config)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        
        # Pruning support
        self.layer_idx = layer_idx
        self.q_num_heads, self.k_num_heads, self.v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads
        self.heads_to_preserve = torch.arange(self.num_heads)


        self.selected_qk_out_dim_indices = None
        self.selected_vo_out_dim_indices = None
        self._init_rope()

    def is_GQA(self):
        return self.num_key_value_groups != 1

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.num_heads,
                self.head_dim,
                max_position_embeddings=cfg.config.max_seq_len,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
            
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def compute_pruning_selection(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs):
        vo_prune_way = cfg.derived.vo_prune_way
        sequence_length = hidden_states.shape[1]

        bsz, q_len, _ = hidden_states.size()
        self.q_num_heads, self.k_num_heads, self.v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads

        # Compute attention output for every token in the current sequence.
        query_states_full = self.q_proj(hidden_states, cal_attn_metric_scores=True)
        key_states_full = self.k_proj(hidden_states, cal_attn_metric_scores=True)
        value_states_full = self.v_proj(hidden_states, cal_attn_metric_scores=True)
        query_states_full = query_states_full.view(query_states_full.shape[0], q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_full = key_states_full.view(key_states_full.shape[0], q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states_full = value_states_full.view(value_states_full.shape[0], q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states_full.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states_full, seq_len=sequence_length)
        query_states, key_states = apply_rotary_pos_emb(query_states_full, key_states_full, cos, sin, position_ids)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states_full, self.num_key_value_groups)
        
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if causal_mask is not None:
            mask_length = causal_mask.shape[-1]
            last_row = causal_mask[:, :, -1, :].clone()
            new_mask = last_row.unsqueeze(2).expand(-1, -1, mask_length, -1)
            causal_mask = new_mask

        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(key_states.shape[0], q_len, self.num_heads * self.head_dim)
        metric_scores = self.pruning_module.cal_attn_prune_metric(
            attention_output,
            self.o_proj.weight.data.t() if cfg.config.transpose_down_linear else self.o_proj.weight.data, 
            global_metric_score_distribution=self.o_proj.get_global_metric_score_distribution()
        )
        selected_vo_out_dim_indices, self.remain_num_heads, self.heads_to_preserve = (
            self.pruning_module.sort_attn_metric(
                metric_scores, self.num_heads, self.head_dim, vo_prune_way, "vo", cfg.derived.tc_multiple
            )
        )

        if self.is_GQA():
            self.q_num_heads = self.remain_num_heads
            selected_qk_out_dim_indices = selected_vo_out_dim_indices
        else:
            self.q_num_heads, self.k_num_heads, self.v_num_heads = self.remain_num_heads, self.remain_num_heads, self.remain_num_heads
            selected_qk_out_dim_indices = selected_vo_out_dim_indices

        return selected_qk_out_dim_indices, selected_vo_out_dim_indices, None, attention_output, past_key_value, key_states_full, value_states_full
    
    # Pruning support
    def attention_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs
        ):
        # full inference
        bsz, q_len, _ = hidden_states.size()
        q_out = self.q_proj.weight.shape[0]  # = num_heads * head_dim
        k_out = self.k_proj.weight.shape[0]  # = num_kv_heads * head_dim
        num_heads = q_out // self.head_dim
        num_key_value_heads = k_out // self.head_dim
        num_key_value_groups = num_heads // num_key_value_heads
       
        if self.config.pretraining_tp > 1:
            key_value_slicing = (num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
        # key_states: bsz, self.num_key_value_heads, q_len, self.head_dim -> bsz, self.num_key_value_heads, self.head_dim, q_len
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # value_states: bsz, self.num_key_value_heads, q_len, self.head_dim
        # value_states: 1, self.num_key_value_heads, q_len, self.head_dim
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, num_heads * self.head_dim)
        if cfg.state.pad_tokens is not None:
            cfg.state.pad_tokens.to(attn_weights.device) 
            attn_output[cfg.state.pad_tokens] = 0

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        if not cfg.config.prune_flag:
            return self.attention_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
        if cfg.state.calibration_stage:
            return self.attention_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
        else:
            attn_targets_enabled = any(
                module in cfg.config.cust_tgt_modules
                for module in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
            if attn_targets_enabled and self.layer_idx not in cfg.derived.skip_layers:
                bsz, q_len, _ = hidden_states.size()
                selected_qk_out_dim_indices, selected_vo_out_dim_indices = None, None
                if cfg.config.mode == "dynamic":
                    if cfg.state.is_prune: 
                        nvtx.range_push("attn activation_input")
                        selected_qk_out_dim_indices, selected_vo_out_dim_indices, attn_weights, attn_output, past_key_value = self.compute_pruning_selection(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
                        self.selected_qk_out_dim_indices = selected_qk_out_dim_indices
                        self.selected_vo_out_dim_indices = selected_vo_out_dim_indices
                        nvtx.range_pop()

                    # partial inference
                    bsz, q_len, _ = hidden_states.size()
                    query_states = self.q_proj(hidden_states, out_dim_indices=self.selected_qk_out_dim_indices)
                    key_states = self.k_proj(hidden_states, out_dim_indices=self.selected_qk_out_dim_indices)
                    value_states = self.v_proj(hidden_states, out_dim_indices=self.selected_vo_out_dim_indices)

                    query_states = query_states.view(bsz, q_len, self.q_num_heads, self.head_dim).transpose(1, 2)
                    key_states = key_states.view(bsz, q_len, self.k_num_heads, self.head_dim).transpose(1, 2)
                    value_states = value_states.view(bsz, q_len, self.v_num_heads, self.head_dim).transpose(1, 2)

                    kv_seq_len = key_states.shape[-2]
                    if past_key_value is not None:
                        kv_seq_len += past_key_value[0].shape[-2]
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

                    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

                    if past_key_value is not None:
                        key_states = torch.cat([past_key_value[0], key_states], dim=2)
                        value_states = torch.cat([past_key_value[1], value_states], dim=2)

                    past_key_value = (key_states, value_states) if use_cache else None
                    if self.is_GQA():
                        key_states = repeat_kv(key_states, self.num_key_value_groups, self.heads_to_preserve)
                        value_states = repeat_kv(value_states, self.num_key_value_groups, self.heads_to_preserve)
                    else:
                        key_states = repeat_kv(key_states, self.num_key_value_groups)
                        value_states = repeat_kv(value_states, self.num_key_value_groups)

                    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                    if attn_weights.size() != (bsz, self.q_num_heads, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention weights should be of size {(bsz, self.q_num_heads, q_len, kv_seq_len)}, but is"
                            f" {attn_weights.size()}"
                        )

                    if attention_mask is not None:
                        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                            raise ValueError(
                                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                            )
                        attn_weights = attn_weights + attention_mask
                        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
                        
                    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                    attn_output = torch.matmul(attn_weights, value_states)
                    if attn_output.size() != (bsz, self.q_num_heads, q_len, self.head_dim):
                        raise ValueError(
                            f"`attn_output` should be of size {(bsz, self.q_num_heads, q_len, self.head_dim)}, but is"
                            f" {attn_output.size()}"
                        )
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, self.q_num_heads * self.head_dim)
                    if cfg.state.pad_tokens is not None:
                        cfg.state.pad_tokens.to(attn_weights.device) 
                        attn_output[cfg.state.pad_tokens] = 0

                    attn_output = self.o_proj(attn_output, in_dim_indices=self.selected_vo_out_dim_indices)
                            
                    if not output_attentions:
                        attn_weights = None
                    return attn_output, attn_weights, past_key_value
            else: 
                return self.attention_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self._weight_squared_cached = None

    # ================= CUDA Graph Compatible Methods =================

    def cache_weight_squared(self):
        """
        Pre-cache weight norms for metric calculation. Call once after model loading.

        Mathematical optimization MLP 
        Original computation weighted = norm_val * weight dim=(0,2)  L2 norm
         norm_val[j]  dim=0 
            metric[h] = ||norm_val_h * weight_sq_l2_norm_h||_2

        Only cache ||weight||_2  dim=0

        Memory savings
        - Original: [4096, 4096] = 33 MB/
        - Optimized: [4096] = 8 KB/
        - Savings: 99.97%
        """
        weight = self.o_proj.weight.data  # [hidden_size, hidden_size] = [4096, 4096]
        # Only transpose if explicitly configured
        if cfg.config.transpose_down_linear:
            weight = weight.t()

        weight_sq = weight.pow(2)
        self._weight_sq_l2_norm = torch.linalg.vector_norm(weight_sq, ord=2, dim=0).contiguous()  # [hidden_size]

    def dense_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_buffer: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Dense attention forward - no branches, no cfg access, CUDA Graph compatible.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len]
            position_ids: [batch, seq_len]
            cos_sin_cache: (cos, sin) pre-computed RoPE tensors
            attn_buffer: Optional [1, seq_len, hidden_size] buffer for metric computation
                        If provided, stores attn_output in-place to avoid copy.

        Returns:
            [batch, seq_len, hidden_size] output tensor
        """
        bsz, q_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = cos_sin_cache
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        # Make contiguous for SDPA
        if query.device.type == "cuda":
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # Apply attention mask modification for bidirectional attention
        # Match the non-CUDA-Graph path (line 919-923) for equivalent behavior:
        # Expand the last row to every row for bidirectional attention among action tokens
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, :key.shape[-2]]
            # last_row expand: allow all action tokens to attend to each other
            D = causal_mask.shape[-1]
            last_row = causal_mask[:, :, -1, :].clone()
            new_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)
            causal_mask = new_mask

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Store in buffer if provided (buffer shape matches attn_output)
        if attn_buffer is not None:
            attn_buffer.copy_(attn_output)  # Keep the copy because attn_output is already reshaped

        return self.o_proj(attn_output)

    def sparse_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin_cache: Tuple[torch.Tensor, torch.Tensor],
        qk_indices: torch.Tensor,
        vo_indices: torch.Tensor,
        num_heads: int
    ) -> torch.Tensor:
        """
        Sparse attention forward - uses external indices, CUDA Graph compatible.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len]
            position_ids: [batch, seq_len]
            cos_sin_cache: (cos, sin) pre-computed RoPE tensors
            qk_indices: [keep_dims] indices for Q/K projection
            vo_indices: [keep_dims] indices for V/O projection
            num_heads: number of remaining heads after pruning

        Returns:
            [batch, seq_len, hidden_size] output tensor
        """
        bsz, q_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states, out_dim_indices=qk_indices)
        key = self.k_proj(hidden_states, out_dim_indices=qk_indices)
        value = self.v_proj(hidden_states, out_dim_indices=vo_indices)

        query = query.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)

        cos, sin = cos_sin_cache
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        # Make contiguous for SDPA
        if query.device.type == "cuda":
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # Apply attention mask modification for bidirectional attention
        causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask[:, :, :, :key.shape[-2]]
            # last_row expand: allow all action tokens to attend to each other
            D = causal_mask.shape[-1]
            last_row = causal_mask[:, :, -1, :].clone()
            new_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)
            causal_mask = new_mask

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, num_heads * self.head_dim)

        return self.o_proj(attn_output, in_dim_indices=vo_indices)

    @staticmethod
    def sort_attn_metric_static(
        metric: torch.Tensor,
        num_heads: int,
        head_dim: int,
        prune_ratio: float
    ) -> Tuple[torch.Tensor, int]:
        """
        Sort attention metric and return head indices to keep.

        Args:
            metric: [num_heads] per-head importance scores (already aggregated by head)
                     [hidden_size] requires internal aggregation
            num_heads: total number of attention heads
            head_dim: dimension per head
            prune_ratio: ratio of heads to prune

        Returns:
            (keep_indices, remain_num_heads) where:
            - keep_indices: [keep_heads * head_dim] flattened dimension indices
            - remain_num_heads: number of heads kept
        """
        # Support two input formats:
        if metric.numel() == num_heads:
            head_scores = metric
        else:
            # Legacy format: [hidden_size] -> [num_heads, head_dim] -> sum -> [num_heads]
            metric_reshaped = metric.view(num_heads, head_dim)
            head_scores = metric_reshaped.sum(dim=1)  # [num_heads]

        # Calculate how many heads to keep
        num_prune_heads = int(prune_ratio * num_heads)
        remain_num_heads = num_heads - num_prune_heads

        topk_indices = torch.topk(head_scores, remain_num_heads, largest=True, sorted=False).indices
        mask = torch.zeros(num_heads, dtype=torch.bool, device=metric.device)
        mask.scatter_(0, topk_indices, True)
        heads_to_keep_sorted = mask.nonzero(as_tuple=False).squeeze(-1)

        # Expand to full dimension indices
        full_indices = (
            heads_to_keep_sorted.unsqueeze(1) * head_dim +
            torch.arange(head_dim, device=metric.device).unsqueeze(0)
        ).view(-1)

        return full_indices, remain_num_heads

    def compute_pruning_selection(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs):
        vo_prune_way = cfg.derived.vo_prune_way
        sequence_length = hidden_states.shape[1]

        bsz, q_len, _ = hidden_states.size()
        self.q_num_heads, self.k_num_heads, self.v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads

        query_states_full = self.q_proj(hidden_states, cal_attn_metric_scores=True)
        key_states_full = self.k_proj(hidden_states, cal_attn_metric_scores=True)
        value_states_full = self.v_proj(hidden_states, cal_attn_metric_scores=True)
        query_states_full = query_states_full.view(query_states_full.shape[0], q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_full = key_states_full.view(key_states_full.shape[0], q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states_full = value_states_full.view(value_states_full.shape[0], q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states_full.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states_full, seq_len=sequence_length)
        query_states, key_states = apply_rotary_pos_emb(query_states_full, key_states_full, cos, sin, position_ids)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states_full, self.num_key_value_groups)
        
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if causal_mask is not None:
            mask_length = causal_mask.shape[-1]
            last_row = causal_mask[:, :, -1, :].clone()
            new_mask = last_row.unsqueeze(2).expand(-1, -1, mask_length, -1)
            causal_mask = new_mask

        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(key_states.shape[0], q_len, self.num_heads * self.head_dim)
        metric_scores = self.pruning_module.cal_attn_prune_metric(
            attention_output,
            self.o_proj.weight.data.t() if cfg.config.transpose_down_linear else self.o_proj.weight.data, 
            global_metric_score_distribution=self.o_proj.get_global_metric_score_distribution(),
        )
        selected_vo_out_dim_indices, self.remain_num_heads, self.heads_to_preserve = (
            self.pruning_module.sort_attn_metric(
                metric_scores, self.num_heads, self.head_dim, vo_prune_way, "vo", cfg.derived.tc_multiple
            )
        )

        if self.is_GQA():
            self.q_num_heads = self.remain_num_heads
            selected_qk_out_dim_indices = selected_vo_out_dim_indices
        else:
            self.q_num_heads, self.k_num_heads, self.v_num_heads = self.remain_num_heads, self.remain_num_heads, self.remain_num_heads
            selected_qk_out_dim_indices = selected_vo_out_dim_indices

        return selected_qk_out_dim_indices, selected_vo_out_dim_indices, None, attention_output, past_key_value, key_states_full, value_states_full
    
    def pre_forward(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs):
        sequence_length = hidden_states.shape[1]

        _, q_len, _ = hidden_states.size()
        self.q_num_heads, self.k_num_heads, self.v_num_heads = self.num_heads, self.num_key_value_heads, self.num_key_value_heads

        query_states_full = self.q_proj(hidden_states, cal_attn_metric_scores=True)
        key_states_full = self.k_proj(hidden_states, cal_attn_metric_scores=True)
        value_states_full = self.v_proj(hidden_states, cal_attn_metric_scores=True)
        query_states_full = query_states_full.view(query_states_full.shape[0], q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states_full = key_states_full.view(key_states_full.shape[0], q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states_full = value_states_full.view(value_states_full.shape[0], q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states_full.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states_full, seq_len=sequence_length)
        query_states, key_states = apply_rotary_pos_emb(query_states_full, key_states_full, cos, sin, position_ids)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states_full, self.num_key_value_groups)
        
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if causal_mask is not None:
            mask_length = causal_mask.shape[-1]
            last_row = causal_mask[:, :, -1, :].clone()
            new_mask = last_row.unsqueeze(2).expand(-1, -1, mask_length, -1)
            causal_mask = new_mask

        attention_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(key_states.shape[0], q_len, self.num_heads * self.head_dim)
        return attention_output

    def prune_post_process(self, attention_output):
        vo_prune_way = cfg.derived.vo_prune_way

        metric_scores = self.pruning_module.cal_attn_prune_metric(
            attention_output,
            self.o_proj.weight.data.t() if cfg.config.transpose_down_linear else self.o_proj.weight.data, 
            global_metric_score_distribution=self.o_proj.get_global_metric_score_distribution(),
        )
        selected_vo_out_dim_indices, self.remain_num_heads, self.heads_to_preserve = (
            self.pruning_module.sort_attn_metric(
                metric_scores, self.num_heads, self.head_dim, vo_prune_way, "vo", cfg.derived.tc_multiple
            )
        )

        cfg.stats.real_attn_keep_ratio = self.remain_num_heads / self.num_heads

        if self.is_GQA():
            self.q_num_heads = self.remain_num_heads
            selected_qk_out_dim_indices = selected_vo_out_dim_indices
        else:
            self.q_num_heads, self.k_num_heads, self.v_num_heads = self.remain_num_heads, self.remain_num_heads, self.remain_num_heads
            selected_qk_out_dim_indices = selected_vo_out_dim_indices
        
        return selected_qk_out_dim_indices, selected_vo_out_dim_indices

    def update_pruning_selection(self, attention_output):
        selected_qk_out_dim_indices, selected_vo_out_dim_indices = self.prune_post_process(attention_output)
        self.selected_qk_out_dim_indices = selected_qk_out_dim_indices
        self.selected_vo_out_dim_indices = selected_vo_out_dim_indices
    
    # Adapted from LlamaAttention.forward
    def attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs, 
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()
        q_out = self.q_proj.weight.shape[0]  # = num_heads * head_dim
        k_out = self.k_proj.weight.shape[0]  # = num_kv_heads * head_dim
        num_heads = q_out // self.head_dim
        num_key_value_heads = k_out // self.head_dim
        num_key_value_groups = num_heads // num_key_value_heads

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # In case static cache is used, it is an instance attribute.
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if causal_mask is not None:
            D = causal_mask.shape[-1]
            last_row = causal_mask[:, :, -1, :].clone()
            new_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)
            causal_mask = new_mask

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, num_heads * attn_output.shape[-1])

        attn_output = self.o_proj(attn_output)        
        return attn_output, None, past_key_value

    def prune_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass used during pruning
        """
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, out_dim_indices=self.selected_qk_out_dim_indices)
        key_states = self.k_proj(hidden_states, out_dim_indices=self.selected_qk_out_dim_indices)
        value_states = self.v_proj(hidden_states, out_dim_indices=self.selected_vo_out_dim_indices)
        query_states = query_states.view(bsz, q_len, self.q_num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.k_num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.v_num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if causal_mask is not None:
            D = causal_mask.shape[-1]
            last_row = causal_mask[:, :, -1, :].clone()
            new_mask = last_row.unsqueeze(2).expand(-1, -1, D, -1)
            causal_mask = new_mask

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.q_num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output, in_dim_indices=self.selected_vo_out_dim_indices)
        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        if cfg.state.calibration_stage:
            return self.attention_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
        else:
            attn_targets_enabled = any(
                module in cfg.config.cust_tgt_modules
                for module in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
            if attn_targets_enabled and self.layer_idx not in cfg.derived.skip_layers:
                if cfg.config.mode == "dynamic": 
                    if not cfg.config.async_prune:
                        if cfg.state.is_prune: 
                            attention_output = self.pre_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
                            attn_output = self.o_proj(attention_output)

                            self.update_pruning_selection(attention_output)
                            return attn_output, None, past_key_value 
                        
                        attn_output, _, past_key_value = self.prune_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
                        return attn_output, None, past_key_value
                    else:
                        if cfg.state.is_prune:
                            # N-Batch mode
                            attn_buffer = kwargs.get('attn_buffer', None)
                            buffer_idx = kwargs.get('buffer_idx', None)

                            attention_output = self.pre_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)

                            if attn_buffer is not None and buffer_idx is not None:
                                attn_buffer[buffer_idx].copy_(attention_output.squeeze(0))
                            else:
                                with torch.cuda.stream(prune_stream):
                                    self.update_pruning_selection(attention_output)

                            attn_output = self.o_proj(attention_output)
                            return attn_output, None, past_key_value

                        current_qk_indices = self.selected_qk_out_dim_indices
                        if current_qk_indices is None:
                            attn_output, _, past_key_value = self.attention_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
                        else:
                            attn_output, _, past_key_value = self.prune_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)
                        
                        return attn_output, None, past_key_value
            else:
                return self.attention_forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, **kwargs)


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config, layer_idx)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Pruning support
        self.layer_idx = layer_idx
        self.llamaconfig = config
        
        self.cur_attn_inference_duration = 0
        self.cur_mlp_inference_duration = 0

        self.is_changing_gpu = False
        self.whole_batch_calculation = torch.cuda.Event(enable_timing=False, blocking=False)

        self.start_event = torch.cuda.Event(enable_timing=True, blocking=False)
        self.end_event = torch.cuda.Event(enable_timing=True, blocking=False)

    def dense_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin_cache: Tuple[torch.Tensor, torch.Tensor],
        mlp_buffer: torch.Tensor = None,
        attn_buffer: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Dense forward pass - no branches, no cfg access, CUDA Graph compatible.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len]
            position_ids: [batch, seq_len]
            cos_sin_cache: (cos, sin) pre-computed RoPE tensors
            mlp_buffer: Optional buffer for MLP activation values
            attn_buffer: Optional buffer for attention output

        Returns:
            [batch, seq_len, hidden_size] output tensor
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn.dense_forward(
            hidden_states, attention_mask, position_ids, cos_sin_cache, attn_buffer
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.dense_forward(hidden_states, mlp_buffer)
        hidden_states = residual + hidden_states

        return hidden_states

    def sparse_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin_cache: Tuple[torch.Tensor, torch.Tensor],
        mlp_indices: torch.Tensor,
        attn_qk_indices: torch.Tensor,
        attn_vo_indices: torch.Tensor,
        num_heads: int
    ) -> torch.Tensor:
        """
        Sparse forward pass - uses external indices, CUDA Graph compatible.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, 1, seq_len, seq_len]
            position_ids: [batch, seq_len]
            cos_sin_cache: (cos, sin) pre-computed RoPE tensors
            mlp_indices: [mlp_keep_dims] indices for MLP pruning
            attn_qk_indices: [attn_keep_dims] indices for Q/K projection
            attn_vo_indices: [attn_keep_dims] indices for V/O projection
            num_heads: number of remaining attention heads

        Returns:
            [batch, seq_len, hidden_size] output tensor
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn.sparse_forward(
            hidden_states, attention_mask, position_ids, cos_sin_cache,
            attn_qk_indices, attn_vo_indices, num_heads
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.sparse_forward(hidden_states, mlp_indices)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # N-Batch arguments
        mlp_buffer: Optional[torch.Tensor] = None,
        attn_buffer: Optional[torch.Tensor] = None,
        buffer_idx: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            respick=residual,
            attn_buffer=attn_buffer,       # N-Batch arguments
            buffer_idx=buffer_idx,         # N-Batch arguments
        )
        
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(
            hidden_states,
            respick=residual,
            mlp_buffer=mlp_buffer,         # N-Batch arguments
            buffer_idx=buffer_idx,         # N-Batch arguments
        )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, None


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if hasattr(self.config, "_pre_quantization_dtype"):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, device=device, dtype=dtype
            )

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.total_cuda_time = 0
        self.num_forward = 0

        # Async timing for new scheduler path (reports previous frame's latency)
        self._prev_start_event = None
        self._prev_end_event = None
        self._prev_is_prune = True  # Track previous frame's prune decision for logging
        self._async_timing_count = 0
        self._async_total_time = 0.0
        self.all_FLOPs = 0

        # Initialize weights and apply final processing
        self.post_init()
        self.shared_prune_stream = torch.cuda.Stream()
        for layer in self.layers:
            if hasattr(layer, 'self_attn'):
                layer.self_attn.prune_stream = self.shared_prune_stream
                layer.self_attn.prune_event = torch.cuda.Event() 
                layer.self_attn.is_selecting = False
                # Buffer for pending pruning selections.
                layer.self_attn.pending_qk = None
                layer.self_attn.pending_vo = None
            if hasattr(layer, 'mlp'):
                layer.mlp.prune_stream = self.shared_prune_stream
                layer.mlp.prune_event = torch.cuda.Event()
                layer.mlp.is_selecting = False
                layer.mlp.pending_indices = None

        self._nbatch_buffers_initialized = False
        self._nbatch_metrics_pending = False
        self._nbatch_activations_ready = torch.cuda.Event()
        self._nbatch_metrics_done = torch.cuda.Event()

        # CUDA Graph Infrastructure
        self._scheduler = None
        self._dense_graph = None
        self._sparse_graph = None
        self._graphs_initialized = False

        # Static buffers for CUDA Graph
        self._static_input = None
        self._static_output = None
        self._static_mask = None
        self._static_pos = None
        self._cos_sin_cache = None

        # Activation buffers
        self._mlp_activation_buffer = None
        self._attn_activation_buffer = None

        # Layer index mapping
        self._layer_to_buffer = {}

        # Metric computation stream
        self._metric_stream = None

    def set_scheduler(self, scheduler):
        """
        Set the pruning scheduler for CUDA Graph mode.

        Args:
            scheduler: PruningScheduler instance from pruning_scheduler
        """
        self._scheduler = scheduler

    def init_dual_cuda_graphs(self):
        """
        Initialize Dense Graph and Sparse Graph for CUDA Graph mode.

        Must be called after set_scheduler() and before model inference.
        This will:
        1. Allocate static buffers
        2. Pre-compute RoPE cache
        3. Pre-cache all weight tensors
        4. Capture Dense and Sparse CUDA Graphs
        """
        if self._graphs_initialized:
            return

        if self._scheduler is None:
            raise RuntimeError("Must call set_scheduler() before init_dual_cuda_graphs()")

        static_cfg = self._scheduler.static_cfg
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        seq_len = static_cfg.max_seq_len

        # Allocate static buffers
        self._static_input = torch.zeros(
            1, seq_len, static_cfg.hidden_size,
            device=device, dtype=dtype
        )
        self._static_output = torch.zeros(
            1, seq_len, static_cfg.hidden_size,
            device=device, dtype=dtype
        )
        self._static_mask = torch.zeros(
            1, 1, seq_len, seq_len,
            device=device, dtype=dtype
        )
        self._static_pos = torch.arange(seq_len, device=device).unsqueeze(0)

        # Pre-compute RoPE cos/sin cache
        self._cos_sin_cache = self._precompute_rope_cache(seq_len, device, dtype)

        # Activation buffers for metric computation
        num_prunable = static_cfg.num_prunable_layers
        self._mlp_activation_buffer = torch.zeros(
            num_prunable, 1, seq_len, static_cfg.intermediate_size,  # [29, 1, 611, 11008]
            device=device, dtype=dtype
        )
        self._attn_activation_buffer = torch.zeros(
            num_prunable, 1, seq_len, static_cfg.hidden_size,  # [29, 1, 611, 4096]
            device=device, dtype=dtype
        )

        # Build layer index mapping
        buffer_idx = 0
        for i in range(len(self.layers)):
            if i not in static_cfg.skip_layers:
                self._layer_to_buffer[i] = buffer_idx
                buffer_idx += 1

        for layer in self.layers:
            layer.mlp.cache_weight_squared()
            layer.self_attn.cache_weight_squared()

        # Create metric stream
        self._metric_stream = torch.cuda.Stream()

        # Capture Dense Graph
        # Warmup run
        with torch.no_grad():
            _ = self._dense_layers_forward()

        self._dense_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._dense_graph):
            out = self._dense_layers_forward()
            self._static_output.copy_(out)

        # Capture Sparse Graph
        # Warmup run
        with torch.no_grad():
            _ = self._sparse_layers_forward()

        self._sparse_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._sparse_graph):
            out = self._sparse_layers_forward()
            self._static_output.copy_(out)

        self._graphs_initialized = True

    def _precompute_rope_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pre-compute RoPE cos/sin tensors for the given sequence length.

        Returns:
            (cos, sin) tensors of shape [seq_len, head_dim]
        """
        # Use the rotary embedding from the first layer
        rotary_emb = self.layers[0].self_attn.rotary_emb

        # Get cos/sin from the rotary embedding
        # Create a dummy tensor to trigger cache computation
        dummy = torch.zeros(1, 1, 1, rotary_emb.dim, device=device, dtype=dtype)
        cos, sin = rotary_emb(dummy, seq_len=seq_len)

        return (cos.to(dtype), sin.to(dtype))

    def _dense_layers_forward(self) -> torch.Tensor:
        """
        Branch-free Dense forward - captured by CUDA Graph.

        All layers use full computation, with activation values
        written to buffers for later metric computation.

        Note: Does NOT include final RMSNorm (self.norm) - that's done outside
        the timing scope to measure only the VLM backbone layers.
        """
        hidden = self._static_input

        for i, layer in enumerate(self.layers):
            buf_idx = self._layer_to_buffer.get(i)

            if buf_idx is not None:
                # Prunable layer: write to activation buffer
                hidden = layer.dense_forward(
                    hidden,
                    self._static_mask,
                    self._static_pos,
                    self._cos_sin_cache,
                    self._mlp_activation_buffer[buf_idx],
                    self._attn_activation_buffer[buf_idx]
                )
            else:
                # Skip layer: no buffer
                hidden = layer.dense_forward(
                    hidden,
                    self._static_mask,
                    self._static_pos,
                    self._cos_sin_cache
                )

        # Return without norm - norm is applied after timing
        return hidden

    def _sparse_layers_forward(self) -> torch.Tensor:
        """
        Branch-free Sparse forward - captured by CUDA Graph.

        Uses indices from scheduler.state (updated by metric computation).
        """
        hidden = self._static_input
        state = self._scheduler.state

        for i, layer in enumerate(self.layers):
            buf_idx = self._layer_to_buffer.get(i)

            if buf_idx is not None:
                # Prunable layer: use sparse forward
                # Note: Use static_cfg.attn_keep_heads instead of int(remain_num_heads[buf_idx])
                # which is not allowed during CUDA Graph capture.
                # The value is always the same (fixed by prune_ratio), only indices vary.
                hidden = layer.sparse_forward(
                    hidden,
                    self._static_mask,
                    self._static_pos,
                    self._cos_sin_cache,
                    state.mlp_indices_buffer[buf_idx],
                    state.attn_qk_indices_buffer[buf_idx],
                    state.attn_vo_indices_buffer[buf_idx],
                    self._scheduler.static_cfg.attn_keep_heads
                )
            else:
                # Skip layer: use dense forward
                hidden = layer.dense_forward(
                    hidden,
                    self._static_mask,
                    self._static_pos,
                    self._cos_sin_cache
                )

        # Return without norm - norm is applied after timing scope
        return hidden

    def _launch_async_metrics(self):
        """
        Launch metric computation asynchronously on metric_stream.

        Called after dense_graph.replay() to compute indices for next sparse frame.
        Does NOT block the main stream.
        """
        main_stream = torch.cuda.current_stream()
        copy_done = torch.cuda.Event()
        copy_done.record(main_stream)

        self._metric_stream.wait_event(copy_done)

        with torch.cuda.stream(self._metric_stream):
            # Metric computation timing (on metric_stream)
            metric_start = torch.cuda.Event(enable_timing=True)
            metric_end = torch.cuda.Event(enable_timing=True)
            metric_start.record()

            self._compute_all_metrics_graph()

            metric_end.record()

            # Store events for async timing report
            if not hasattr(self, '_metric_timing_events'):
                self._metric_timing_events = []
                self._metric_timing_count = 0
                self._metric_total_time = 0.0
            self._metric_timing_events.append((metric_start, metric_end))

    def _compute_all_metrics_graph(self):
        """
        Optimized metric  v4Preallocate buffer +  + 

        Optimizations
        1. x  Metric  EMA
        2. Preallocate buffer ~2.2 GB/
        3.  out= 
        4.  in-place 

        Required operation order, matching Linear._aggregate_squared_activation:
        Step 1: buffer <- x
        Step 2: clamp each token's squared activation in-place
        Step 3: local_norm <- sum(buffer, dim=sequence)
        Step 4: update EMA from the same local_norm
        """
        static_cfg = self._scheduler.static_cfg
        data_type_max = static_cfg.data_type_max
        num_layers = len(self._layer_to_buffer)

        # Step 0: Prepare activations
        all_mlp_activations = self._mlp_activation_buffer.squeeze(1)  # [29, 611, 11008]
        all_attn_activations = self._attn_activation_buffer.squeeze(1)  # [29, 611, 4096]

        # Lazily initialize weight norms
        if not hasattr(self, '_stacked_mlp_weight_sq_norm'):
            self._init_stacked_weight_norms()

        if not hasattr(self, '_mlp_squared_buffer'):
            self._init_squared_buffers()

        # Square into preallocated buffers, then clamp per token before reducing
        torch.pow(all_mlp_activations, 2, out=self._mlp_squared_buffer)
        torch.pow(all_attn_activations, 2, out=self._attn_squared_buffer)
        self._mlp_squared_buffer.clamp_(max=data_type_max)
        self._attn_squared_buffer.clamp_(max=data_type_max)

        history_metric_ratio = cfg.config.history_metric_ratio

        # These reductions now exactly follow the normal path's per-token clamp.
        mlp_local_norm = torch.sum(self._mlp_squared_buffer, dim=1)  # [29, 11008]

        all_mlp_metrics = self._cal_batched_mlp_metric(
            mlp_local_norm,
            self._stacked_mlp_weight_sq_norm,
            self._stacked_mlp_global_metric,
            history_metric_ratio,
            data_type_max
        ).clone()

        attn_local_norm = torch.sum(self._attn_squared_buffer, dim=1)  # [29, 4096]

        all_attn_metrics = self._cal_batched_attn_metric(
            attn_local_norm,
            self._stacked_attn_weight_sq_norm,
            self._stacked_attn_global_metric,
            history_metric_ratio,
            data_type_max,
            static_cfg.num_attention_heads,
            static_cfg.head_dim,
        ).clone()

        # Reuse the same per-token-clamped reductions for history EMA.
        if cfg.config.history_update and self._stacked_mlp_global_metric is not None:
            momentum = cfg.derived.ema_momentum
            self._stacked_mlp_global_metric.mul_(momentum).add_(
                mlp_local_norm, alpha=1 - momentum
            )
            self._stacked_attn_global_metric.mul_(momentum).add_(
                attn_local_norm, alpha=1 - momentum
            )

        # Step 4: Sort indices
        for buf_idx in range(num_layers):
            mlp_indices = LlamaMLP.sort_metric_static(
                all_mlp_metrics[buf_idx],
                static_cfg.mlp_prune_ratio,
                static_cfg.tc_multiple
            )

            attn_indices, num_heads = LlamaSdpaAttention.sort_attn_metric_static(
                all_attn_metrics[buf_idx],
                static_cfg.num_attention_heads,
                static_cfg.head_dim,
                static_cfg.attn_prune_ratio
            )

            self._scheduler.update_all_indices(buf_idx, mlp_indices, attn_indices, num_heads)

    def _init_stacked_weight_norms(self):
        """
         weight norms  global metric metric 

         weight_sq_norm  global_metric_score_distribution Batch-compute
        """
        sorted_layer_indices = sorted(self._layer_to_buffer.keys())
        static_cfg = self._scheduler.static_cfg
        seq_len = static_cfg.max_seq_len

        # Stack MLP weight norms: [num_layers, intermediate_size]
        mlp_norms = []
        for layer_idx in sorted_layer_indices:
            layer = self.layers[layer_idx]
            norm = layer.mlp._weight_sq_l2_norm
            mlp_norms.append(norm)
        self._stacked_mlp_weight_sq_norm = torch.stack(mlp_norms, dim=0)  # [29, 11008]

        # Stack Attention weight norms: [num_layers, hidden_size]
        attn_norms = []
        for layer_idx in sorted_layer_indices:
            layer = self.layers[layer_idx]
            norm = layer.self_attn._weight_sq_l2_norm
            attn_norms.append(norm)
        self._stacked_attn_weight_sq_norm = torch.stack(attn_norms, dim=0)  # [29, 4096]

        # Stack aggregated MLP global metrics: [num_layers, intermediate_size]
        mlp_global_metrics = []
        for layer_idx in sorted_layer_indices:
            layer = self.layers[layer_idx]
            global_metric = layer.mlp.down_proj.get_global_metric_score_distribution()
            if global_metric is not None:
                mlp_global_metrics.append(global_metric)

        if mlp_global_metrics and len(mlp_global_metrics) == len(sorted_layer_indices):
            self._stacked_mlp_global_metric = torch.stack(mlp_global_metrics, dim=0)  # [29, 11008]
        else:
            self._stacked_mlp_global_metric = None

        # Stack aggregated Attention global metrics: [num_layers, hidden_size]
        attn_global_metrics = []
        for layer_idx in sorted_layer_indices:
            layer = self.layers[layer_idx]
            global_metric = layer.self_attn.o_proj.get_global_metric_score_distribution()
            if global_metric is not None:
                attn_global_metrics.append(global_metric)

        if attn_global_metrics and len(attn_global_metrics) == len(sorted_layer_indices):
            self._stacked_attn_global_metric = torch.stack(attn_global_metrics, dim=0)  # [29, 4096]
        else:
            self._stacked_attn_global_metric = None

    def _init_squared_buffers(self):
        """
        Preallocate squared activation buffers metric 

        Avoid per-frame memory allocationzero-copy writes
         ~2.2 GB 
        """
        mlp_shape = self._mlp_activation_buffer.squeeze(1).shape  # [29, 611, 11008]
        attn_shape = self._attn_activation_buffer.squeeze(1).shape  # [29, 611, 4096]

        device = self._mlp_activation_buffer.device

        self._mlp_squared_buffer = torch.empty(
            mlp_shape,
            device=device,
            dtype=self._mlp_activation_buffer.dtype
        )

        self._attn_squared_buffer = torch.empty(
            attn_shape,
            device=device,
            dtype=self._attn_activation_buffer.dtype
        )

    @staticmethod
    @torch.compile(mode="reduce-overhead", fullgraph=True)
    def _cal_batched_mlp_metric(
        local_norm: torch.Tensor,
        weight_sq_norms: torch.Tensor,  # [L, D]
        global_metrics: torch.Tensor,   # [L, D] or None
        history_metric_ratio: float,
        data_type_max: float
    ) -> torch.Tensor:
        """
        Optimized MLP metric precomputed local_norm

        Mathematically equivalent _cal_batched_mlp_metricavoid duplicate computation x
        Optimizationslocal_norm  EMA  x 
        """
        # Global Metric Combine
        if global_metrics is not None:
            global_sum = global_metrics.clamp(max=data_type_max)  # [L, D]
            combined = history_metric_ratio * global_sum + (1.0 - history_metric_ratio) * local_norm
        else:
            combined = local_norm

        # Combine with weights
        metrics = combined * weight_sq_norms

        return torch.clamp(metrics, max=data_type_max)

    @staticmethod
    @torch.compile(mode="reduce-overhead", fullgraph=True)
    def _cal_batched_attn_metric(
        local_norm: torch.Tensor,
        weight_sq_norms: torch.Tensor,  # [L, H]
        global_metrics: torch.Tensor,   # [L, H] or None
        history_metric_ratio: float,
        data_type_max: float,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """
        Optimized Attention metric precomputed local_norm

        Mathematically equivalent _cal_batched_attn_metricavoid duplicate computation x
        Optimizationslocal_norm  EMA  x 
        """
        num_layers = local_norm.shape[0]

        # Global Metric Combine
        if global_metrics is not None:
            global_sum = global_metrics.clamp(max=data_type_max)  # [L, H]
            combined = history_metric_ratio * global_sum + (1.0 - history_metric_ratio) * local_norm
        else:
            combined = local_norm

        # Combine with weights
        weighted = combined * weight_sq_norms  # [L, H]

        # aggregate by head: [L, H] -> [L, num_heads, head_dim] -> [L, num_heads]
        weighted_reshaped = weighted.view(num_layers, num_heads, head_dim)
        metrics = torch.linalg.vector_norm(weighted_reshaped, ord=2, dim=2)

        return torch.clamp(metrics, max=data_type_max)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _init_nbatch_buffers(self, device, dtype):
        if self._nbatch_buffers_initialized:
            return
        if not cfg.config.nbatch_enabled:
            return

        seq_len = cfg.config.max_seq_len
        intermediate_size = self.config.intermediate_size  # 11008
        hidden_size = self.config.hidden_size  # 4096
        skip_layers = cfg.derived.skip_layers

        num_prunable_layers = len(self.layers) - len(skip_layers)

        self.mlp_activation_buffer = torch.empty(
            num_prunable_layers, seq_len, intermediate_size,
            device=device, dtype=dtype
        )
        self.attn_activation_buffer = torch.empty(
            num_prunable_layers, seq_len, hidden_size,
            device=device, dtype=dtype
        )

        self.prunable_layer_indices = []
        self.layer_to_buffer_idx = {}

        buffer_idx = 0
        for layer_idx in range(len(self.layers)):
            if layer_idx in skip_layers:
                continue  # skip non-prunable layers
            self.prunable_layer_indices.append(layer_idx)
            self.layer_to_buffer_idx[layer_idx] = buffer_idx
            buffer_idx += 1

        for layer_idx in self.prunable_layer_indices:
            self.layers[layer_idx].mlp.cache_weight_squared()
            self.layers[layer_idx].self_attn.cache_weight_squared()

        self._nbatch_buffers_initialized = True

    def _launch_nbatch_metrics(self):
        self._nbatch_activations_ready.record(torch.cuda.current_stream())
        self.shared_prune_stream.wait_event(self._nbatch_activations_ready)

        with torch.cuda.stream(self.shared_prune_stream):
            for buffer_idx, layer_idx in enumerate(self.prunable_layer_indices):
                layer = self.layers[layer_idx]
                layer.mlp.update_pruning_selection(
                    self.mlp_activation_buffer[buffer_idx].unsqueeze(0)
                )
                layer.self_attn.update_pruning_selection(
                    self.attn_activation_buffer[buffer_idx].unsqueeze(0)
                )
            self._nbatch_metrics_done.record(self.shared_prune_stream)

        self._nbatch_metrics_pending = True

    def layers_forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        output_attentions,
        use_cache,
        output_hidden_states
    ):
        """N=29 unified computation across all layers

        Design points
        1.  forward contiguous buffer
        2.  forward  metric 
        3. Do not wait; overlap metric computation with the Action expert
        4. Use results for frame N+22
        """
        if self._nbatch_metrics_pending:
            torch.cuda.current_stream().wait_event(self._nbatch_metrics_done)
            self._nbatch_metrics_pending = False

        nbatch_mode = cfg.config.nbatch_enabled and cfg.state.is_prune
        if nbatch_mode:
            self._init_nbatch_buffers(hidden_states.device, hidden_states.dtype)

        last_layer_residual = None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            buffer_idx = self.layer_to_buffer_idx.get(idx) if nbatch_mode else None

            

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
                last_layer_residual = None
            else:
                layer_outputs, last_layer_residual = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    last_layer_residual=last_layer_residual,
                    # N-Batch arguments
                    mlp_buffer=self.mlp_activation_buffer if nbatch_mode else None,
                    attn_buffer=self.attn_activation_buffer if nbatch_mode else None,
                    buffer_idx=buffer_idx,
                )

            hidden_states = layer_outputs[0]

        if nbatch_mode and not cfg.state.calibration_stage:
            self._launch_nbatch_metrics()

        return hidden_states

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).to(device)

        # for pad token, set to 0
        inputs_embeds[attention_mask == 0] = 0
        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # padding_mask = attention_mask
            cfg.state.first_one_indices = torch.argmax((attention_mask == 1).int(), dim=1).unsqueeze_(1)
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds
        # hidden_states = torch.zeros_like(inputs_embeds)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # record CUDA time
        # Check if using new scheduler path (will skip synchronization)
        _using_scheduler_init = hasattr(self, '_scheduler') and self._scheduler is not None and getattr(self, '_graphs_initialized', False)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Async timing: check if previous frame's events are ready (non-blocking)
        if _using_scheduler_init and self._prev_end_event is not None:
            if self._prev_end_event.query():  # Non-blocking check
                cuda_time_prev = self._prev_start_event.elapsed_time(self._prev_end_event)
                self._async_timing_count += 1
                if self._async_timing_count > 10:  # Skip warmup
                    self._async_total_time += cuda_time_prev
                    avg_time = self._async_total_time / (self._async_timing_count - 10)
                    # Use saved previous frame's is_prune decision for correct logging
                    frame_type = "Prune(Dense)" if self._prev_is_prune else "Sparse"
                    print(f"[Async] {frame_type} latency: {cuda_time_prev:.2f} ms | Avg: {avg_time:.2f} ms")

        # Metric computation timing: check completed events (non-blocking)
        if _using_scheduler_init and hasattr(self, '_metric_timing_events'):
            completed = []
            for start_evt, end_evt in self._metric_timing_events:
                if end_evt.query():  # Completed
                    metric_time = start_evt.elapsed_time(end_evt)
                    self._metric_timing_count += 1
                    if self._metric_timing_count > 10:  # Skip warmup
                        self._metric_total_time += metric_time
                        avg_metric = self._metric_total_time / (self._metric_timing_count - 10)
                        print(f"[Async] Metric computation: {metric_time:.2f} ms | Avg: {avg_metric:.2f} ms")
                    completed.append((start_evt, end_evt))
            # Remove completed events
            for evt in completed:
                self._metric_timing_events.remove(evt)

        if not _using_scheduler_init:
            # Only synchronize for legacy paths
            torch.cuda.current_stream().synchronize()

        start_event.record(torch.cuda.default_stream())
        nvtx.range_push("kernel time")

        # New scheduler-based CUDA Graph path (metric on async stream)
        # Decision is made by similarity logic in modeling_prismatic.py via cfg.state.is_prune
        if hasattr(self, '_scheduler') and self._scheduler is not None and self._graphs_initialized:
            # Copy inputs to static buffers
            self._static_input.copy_(hidden_states)
            self._static_mask.copy_(attention_mask)
            self._static_pos.copy_(position_ids)

            # Replay appropriate Graph based on cfg.state.is_prune (set by similarity decision)
            is_prune_decision = cfg.state.is_prune

            if is_prune_decision:
                self._dense_graph.replay()
            else:
                self._sparse_graph.replay()

            hidden_states = self._static_output

            # async metric launch moved AFTER end_event.record() to exclude launch overhead from timing

        # Original layers_forward path (no CUDA Graph)
        else:
            hidden_states = self.layers_forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states
            )

        # record CUDA time
        # Skip synchronization for new scheduler path to avoid blocking main stream
        _using_scheduler = hasattr(self, '_scheduler') and self._scheduler is not None and self._graphs_initialized

        end_event.record(torch.cuda.default_stream())

        # Save events and prune decision for async timing (will be read in next frame)
        if _using_scheduler:
            self._prev_start_event = start_event
            self._prev_end_event = end_event
            self._prev_is_prune = cfg.state.is_prune  # Save current prune decision

            # Launch async metric computation AFTER timing (to exclude launch overhead)
            # Only launch when this is a prune frame (need activations from dense forward)
            if self._prev_is_prune:
                self._launch_async_metrics()

        if not _using_scheduler:
            # Only synchronize for legacy paths (needed for accurate timing)
            torch.cuda.current_stream().synchronize()

        nvtx.range_pop()
        self.num_forward += 1

        # do not count the first inference (legacy path only)
        if not _using_scheduler and self.num_forward - 1 >= 60:
            cuda_time_current = start_event.elapsed_time(end_event)
            self.total_cuda_time += cuda_time_current
            cuda_time_avg = self.total_cuda_time / (self.num_forward - 60)
            print(f"Current CUDA latency: {cuda_time_current:.6f} ms | Average CUDA latency: {cuda_time_avg:.6f} ms")

        if not _using_scheduler:
            torch.cuda.synchronize()
        
        import time
        sync_start = time.perf_counter()
        torch.cuda.synchronize()
        sync_elapsed = (time.perf_counter() - sync_start) * 1000
        print(f"post-forward device sync: {sync_elapsed:.3f} ms")

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_seen_tokens: int,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        if self.config._attn_implementation == "sdpa":
            # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument,
            # in order to dispatch on Flash Attention 2.
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask, inputs_embeds=input_tensor, past_key_values_length=past_seen_tokens
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):  # static cache
            target_length = self.config.max_position_embeddings
        else:  # dynamic cache
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
