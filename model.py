"""
ETHOS Model Implementation

Contains all model components: attention, routing, expert generation, and main model.

Copyright (C) 2025 Wesley Medford, Chris McCormick, Eve Callicoat

This program is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
For commercial licensing, contact: wryanmedford@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class DeepseekV3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cos_cached = None
        self.sin_cached = None
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, seq_len=None):
        if self.cos_cached is None or seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepseekV3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.num_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.rotary_emb = DeepseekV3RotaryEmbedding(
            self.qk_rope_head_dim, 
            max_position_embeddings=config.max_seq_len, 
            base=config.rope_theta
        )
        
        self.is_causal = True
        self.attention_dropout = 0.0
        self.softmax_scale = self.q_head_dim ** (-0.5)

    def forward(self, hidden_states, attention_mask, position_ids):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe.transpose(1, 2), k_pe.transpose(1, 2), cos, sin, position_ids)

        query_states = torch.cat([q_nope.transpose(1, 2), q_pe], dim=-1)
        key_states = torch.cat([k_nope.transpose(1, 2), k_pe.expand(bsz, self.num_heads, q_len, self.qk_rope_head_dim)], dim=-1)
        value_states = value_states.transpose(1, 2)

        # Use Flash Attention if available
        try:
            from flash_attn import flash_attn_func
            
            query_states = query_states.transpose(1, 2).contiguous()
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            
            if self.q_head_dim != self.v_head_dim:
                value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
            
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=self.is_causal
            )
            
            if self.q_head_dim != self.v_head_dim:
                attn_output = attn_output[:, :, :, :self.v_head_dim]
            
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        except ImportError:
            # Fallback to standard attention
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.softmax_scale
            attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        
        return self.o_proj(attn_output)


class ExpertGenerationNetwork(nn.Module):
    def __init__(self, d_latent, d_model, d_intermediate):
        super().__init__()
        self.d_expert_params = 2 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_intermediate, bias=False),
            nn.GELU(),
            nn.Linear(d_intermediate, self.d_expert_params, bias=False)
        )
    
    def forward(self, latent_vector):
        return self.net(latent_vector)


class ProductKeyRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.d_query = config.d_query
        self.num_heads = config.num_routing_heads

        self.num_sub_keys = int(math.sqrt(self.num_experts))
        assert self.num_sub_keys**2 == self.num_experts, "num_experts must be a perfect square"

        # Multi-head query projections
        self.query_projs = nn.ModuleList([
            nn.Linear(self.d_model, self.d_query) for _ in range(self.num_heads)
        ])
        
        # Batch normalization per head
        self.query_norms = nn.ModuleList([
            nn.BatchNorm1d(self.d_query) for _ in range(self.num_heads)
        ])
        
        # Shared sub-keys across heads
        self.sub_keys_1 = nn.Embedding(self.num_sub_keys, self.d_query // 2)
        self.sub_keys_2 = nn.Embedding(self.num_sub_keys, self.d_query // 2)

    def forward(self, x_flat):
        batch_seq_len = x_flat.shape[0]
        
        all_scores = []
        all_indices = []
        
        for head_idx in range(self.num_heads):
            query = self.query_projs[head_idx](x_flat)
            query = self.query_norms[head_idx](query)
            q1, q2 = query.chunk(2, dim=-1)

            # Calculate scores against each sub-key table
            scores1 = q1 @ self.sub_keys_1.weight.t()
            scores2 = q2 @ self.sub_keys_2.weight.t()

            # Find top candidates from each sub-key set
            k_cand = self.top_k * 2
            top_scores1, top_indices1 = torch.topk(scores1, k_cand, dim=-1)
            top_scores2, top_indices2 = torch.topk(scores2, k_cand, dim=-1)

            # Combine scores of candidate pairs
            combined_scores = top_scores1.unsqueeze(2) + top_scores2.unsqueeze(1)
            combined_scores = combined_scores.view(batch_seq_len, -1)

            # Find the final top_k from the candidate pairs
            final_scores, top_combined_indices = torch.topk(combined_scores, self.top_k, dim=-1)
            final_scores = F.softmax(final_scores, dim=-1, dtype=torch.float).to(x_flat.dtype)

            # Decode the combined indices
            idx_from_1 = top_combined_indices // k_cand
            idx_from_2 = top_combined_indices % k_cand

            # Gather the final sub-key indices
            final_indices_1 = top_indices1.gather(1, idx_from_1)
            final_indices_2 = top_indices2.gather(1, idx_from_2)

            # Calculate the final 1D expert index
            final_expert_indices = final_indices_1 * self.num_sub_keys + final_indices_2
            
            all_scores.append(final_scores)
            all_indices.append(final_expert_indices)
        
        # Stack results: [batch*seq, num_heads, top_k]
        all_scores = torch.stack(all_scores, dim=1)
        all_indices = torch.stack(all_indices, dim=1)
        
        return all_scores, all_indices


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ffn_intermediate, bias=False)
        self.w2 = nn.Linear(config.d_ffn_intermediate, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ffn_intermediate, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SimplifiedHypernetMoE(nn.Module):
    """
    Simplified pure PyTorch implementation of the hypernetwork MoE layer.
    This version doesn't use the reordered execution pattern and is easier
    to understand, though less efficient than the Triton kernel version
    Only suitable for toy scale models.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.num_heads = config.num_routing_heads
        self.d_latent = config.d_latent
        self.d_hidden = config.d_intermediate_hypernet
        
        self.expert_latents = nn.Embedding(config.num_experts, config.d_latent)
        self.generation_network = ExpertGenerationNetwork(
            config.d_latent, config.d_model, config.d_intermediate_hypernet
        )
        self.router = ProductKeyRouter(config)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [B*S, d_model]
        n_tokens = x_flat.shape[0]
        
        # Get routing decisions
        scores, expert_indices = self.router(x_flat)  # [B*S, num_heads, top_k]
        
        # Process all heads at once
        # Reshape to process all experts across all heads
        all_indices = expert_indices.view(-1)  # [B*S*H*K]
        all_scores = scores.view(n_tokens, -1)  # [B*S, H*K]
        
        # Get all latent codes at once
        all_latents = self.expert_latents(all_indices)  # [B*S*H*K, d_latent]
        
        # Generate all expert parameters at once
        all_params = self.generation_network(all_latents)  # [B*S*H*K, 2*d_model]
        
        # Reshape to separate w_in and w_out
        all_params = all_params.view(n_tokens, self.num_heads * self.top_k, 2 * d_model)
        w_in = all_params[:, :, :d_model]  # [B*S, H*K, d_model]
        w_out = all_params[:, :, d_model:]  # [B*S, H*K, d_model]
        
        # Compute activations for all experts at once
        # x_flat: [B*S, d_model] -> [B*S, 1, d_model]
        # w_in: [B*S, H*K, d_model] -> [B*S, H*K, d_model]
        x_expanded = x_flat.unsqueeze(1)  # [B*S, 1, d_model]
        
        # Batch matrix multiply: [B*S, H*K, d_model] @ [B*S, d_model, 1] -> [B*S, H*K, 1]
        activations = torch.bmm(w_in, x_expanded.transpose(1, 2)).squeeze(-1)  # [B*S, H*K]
        activations = F.gelu(activations)
        
        # Apply routing scores
        weighted_activations = activations * all_scores  # [B*S, H*K]
        
        # Compute output: w_out * weighted_activations
        # w_out: [B*S, H*K, d_model]
        # weighted_activations: [B*S, H*K] -> [B*S, H*K, 1]
        weighted_activations = weighted_activations.unsqueeze(-1)  # [B*S, H*K, 1]
        expert_outputs = w_out * weighted_activations  # [B*S, H*K, d_model]
        
        # Sum over all experts (across heads and top-k)
        output = expert_outputs.sum(dim=1)  # [B*S, d_model]
        
        # Average across heads
        output = output / self.num_heads
        
        return output.view(batch_size, seq_len, d_model)

class TransformerBlock(nn.Module):
    def __init__(self, config, is_moe_layer):
        super().__init__()
        self.self_attn = DeepseekV3Attention(config)
        # Import FusedLowRankMoE_Reordered from kernels.py when needed
        if is_moe_layer:
            if config.use_triton:
                from kernels import FusedLowRankMoE_Reordered
                self.mlp = FusedLowRankMoE_Reordered(config)
            else:
                self.mlp = SimplifiedHypernetMoE(config)
        # If not a MoE layer, use standard FFN
        else:
            self.mlp = FFN(config)
        self.input_layernorm = DeepseekV3RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, attention_mask, position_ids):
        h = x + self.self_attn(self.input_layernorm(x), attention_mask, position_ids)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class CompressedMoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        layers = []
        for i in range(config.num_layers):
            is_moe_layer = i >= config.num_dense_layers
            layers.append(TransformerBlock(config, is_moe_layer))
        self.layers = nn.ModuleList(layers)
        
        self.norm = DeepseekV3RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        h = self.tok_embeddings(input_ids)

        attention_mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device, dtype=h.dtype)
        attention_mask = torch.triu(attention_mask, diagonal=1)
        
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)

        for layer in self.layers:
            h = layer(h, attention_mask, position_ids)

        return self.output(self.norm(h))