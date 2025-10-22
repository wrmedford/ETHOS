"""
ETHOS MoE Kernels and Components

Contains Triton kernels and PyTorch modules for the ETHOS low-rank MoE architecture.

Copyright (C) 2025 Wesley Medford, Chris McCormick, Eve Callicoat

This program is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
For commercial licensing, contact: wryanmedford@gmail.com
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def moe_reorder_kernel(
    # Inputs
    x_ptr,               # [B*S, d_model]
    latent_ptr,          # [num_experts, d_latent]
    indices_ptr,         # [B*S, top_k]
    scores_ptr,          # [B*S, top_k]

    wu_ptr,              # [d_model, d_hidden] - W_u for token→hidden projection
    w1_ptr,              # [d_latent, d_hidden]
    wv_ptr,              # [d_hidden, d_model] - W_v for hidden→token projection

    out_ptr,             # [B*S, d_model]

    # Dimensions
    batch_seq_size, d_model, d_latent, d_hidden, top_k,

    # Strides
    stride_x_bs, stride_x_d,
    stride_idx_bs, stride_idx_k,
    stride_score_bs, stride_score_k,
    stride_out_bs, stride_out_d,
    stride_latent_n, stride_latent_d,

    stride_wu_row, stride_wu_col,
    stride_w1_row, stride_w1_col,
    stride_wv_row, stride_wv_col,

    # Block sizes (compile-time constants)
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DHIDDEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= batch_seq_size:
        return

    # Step 1: Project token from d_model → d_hidden using W_u
    d_offs = tl.arange(0, BLOCK_DMODEL)
    h_offs = tl.arange(0, BLOCK_DHIDDEN)
    h_mask = h_offs < d_hidden

    x_proj = tl.zeros([BLOCK_DHIDDEN], dtype=tl.float32)

    # Compute x_proj = x @ W_u
    for d_start in range(0, d_model, BLOCK_DMODEL):
        d_chunk = d_start + d_offs
        x_mask = d_chunk < d_model

        x_chunk = tl.load(
            x_ptr + pid * stride_x_bs + d_chunk * stride_x_d,
            mask=x_mask,
            other=0.0,
        )

        w_chunk = tl.load(
            wu_ptr + d_chunk[:, None] * stride_wu_row + h_offs[None, :] * stride_wu_col,
            mask=x_mask[:, None] & h_mask[None, :],
            other=0.0,
        )

        x_proj += tl.sum(x_chunk[:, None] * w_chunk, axis=0)

    # Step 2: Process each expert
    output = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for k in range(top_k):
        expert_idx = tl.load(indices_ptr + pid * stride_idx_bs + k * stride_idx_k)
        score = tl.load(scores_ptr + pid * stride_score_bs + k * stride_score_k)

        # Compute recovery activations: a^r = GELU(latent @ W1)
        h = tl.zeros([BLOCK_DHIDDEN], dtype=tl.float32)
        for l in range(d_latent):
            latent_val = tl.load(latent_ptr + expert_idx * stride_latent_n + l * stride_latent_d)
            w1_row = tl.load(
                w1_ptr + l * stride_w1_row + h_offs * stride_w1_col,
                mask=h_mask,
                other=0.0,
            )
            h += latent_val * w1_row

        # Apply GELU activation
        h = h * tl.sigmoid(1.702 * h)

        # Compute scalar activation: a^e = GELU(h · x_proj) * score
        dot = tl.sum(h * x_proj)
        activation = dot * tl.sigmoid(1.702 * dot) * score

        # Project back to token space: output += activation * (h @ W_v)
        for d_start in range(0, d_model, BLOCK_DMODEL):
            d_chunk = d_start + d_offs
            x_mask = d_chunk < d_model

            wv_block = tl.load(
                wv_ptr + h_offs[:, None] * stride_wv_row + d_chunk[None, :] * stride_wv_col,
                mask=h_mask[:, None] & x_mask[None, :],
                other=0.0,
            )

            proj = tl.sum(h[:, None] * wv_block, axis=0)

            out_ptr_offset = pid * stride_out_bs + d_chunk * stride_out_d
            old = tl.load(out_ptr + out_ptr_offset, mask=x_mask, other=0.0)
            tl.store(out_ptr + out_ptr_offset, old + activation * proj, mask=x_mask)


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
    def __init__(self, d_model, num_experts, top_k, d_query, num_routing_heads):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_query = d_query
        self.num_heads = num_routing_heads

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


class FusedLowRankMoE_Reordered(nn.Module):
    """
    Reordered Low-Rank MoE that uses the more efficient execution pattern:
    1. Project tokens to hidden dimension once
    2. Compute expert activations in hidden dimension
    3. Project back to token dimension at the end
    """
    def __init__(self, d_model, num_experts, top_k, d_latent, d_intermediate_hypernet, d_query, num_routing_heads):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_heads = num_routing_heads
        self.d_latent = d_latent
        self.d_hidden = d_intermediate_hypernet

        self.expert_latents = nn.Embedding(num_experts, d_latent)
        self.generation_network = ExpertGenerationNetwork(
            d_latent, d_model, d_intermediate_hypernet
        )
        self.router = ProductKeyRouter(d_model, num_experts, top_k, d_query, num_routing_heads)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)
        n_tokens = x_flat.shape[0]

        # Get routing scores and indices for all heads
        scores, expert_indices = self.router(x_flat)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Get shared weight matrices
        W1 = self.generation_network.net[0].weight.t().contiguous()
        W2 = self.generation_network.net[2].weight.t().contiguous()

        # Split W2 into W_u and W_v
        W_u = W2[:, :self.d_model].t().contiguous()
        W_v = W2[:, self.d_model:].contiguous()

        expert_latents_contig = self.expert_latents.weight.contiguous()

        # Process each routing head
        for head_idx in range(self.num_heads):
            head_scores = scores[:, head_idx, :].contiguous()
            head_indices = expert_indices[:, head_idx, :].contiguous()
            head_output = torch.zeros_like(x_flat)

            # Block sizes
            BLOCK_DMODEL = min(1024, self.d_model)
            BLOCK_DHIDDEN = min(64, self.d_hidden)

            # Launch kernel
            grid = (n_tokens,)

            moe_reorder_kernel[grid](
                # Inputs
                x_flat,
                expert_latents_contig,
                head_indices,
                head_scores,

                # Weight matrices
                W_u,
                W1,
                W_v,

                # Output
                head_output,

                # Dimensions
                n_tokens,
                self.d_model,
                self.d_latent,
                self.d_hidden,
                self.top_k,

                # Strides
                x_flat.stride(0), x_flat.stride(1),
                head_indices.stride(0), head_indices.stride(1),
                head_scores.stride(0), head_scores.stride(1),
                head_output.stride(0), head_output.stride(1),
                expert_latents_contig.stride(0), expert_latents_contig.stride(1),

                W_u.stride(0), W_u.stride(1),
                W1.stride(0), W1.stride(1),
                W_v.stride(0), W_v.stride(1),

                # Block sizes
                BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_DHIDDEN=BLOCK_DHIDDEN,
            )

            output += head_output

        # Average across heads
        output = output / self.num_heads
        return output.view(batch_size, seq_len, self.d_model)
