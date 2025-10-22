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
def moe_reorder_fwd_kernel(
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


@triton.jit
def moe_reorder_bwd_kernel(
    # Inputs from forward pass
    x_ptr,               # [B*S, d_model]
    latent_ptr,          # [num_experts, d_latent]
    indices_ptr,         # [B*S, top_k]
    scores_ptr,          # [B*S, top_k]

    wu_ptr,              # [d_model, d_hidden]
    w1_ptr,              # [d_latent, d_hidden]
    wv_ptr,              # [d_hidden, d_model]

    # Gradient input
    grad_output_ptr,     # [B*S, d_model]

    # Gradient outputs
    grad_x_ptr,          # [B*S, d_model]
    grad_latent_ptr,     # [num_experts, d_latent]
    grad_wu_ptr,         # [d_model, d_hidden]
    grad_w1_ptr,         # [d_latent, d_hidden]
    grad_wv_ptr,         # [d_hidden, d_model]
    grad_scores_ptr,     # [B*S, top_k]

    # Dimensions
    batch_seq_size, d_model, d_latent, d_hidden, top_k,

    # Strides
    stride_x_bs, stride_x_d,
    stride_idx_bs, stride_idx_k,
    stride_score_bs, stride_score_k,
    stride_grad_out_bs, stride_grad_out_d,
    stride_grad_x_bs, stride_grad_x_d,
    stride_latent_n, stride_latent_d,
    stride_grad_latent_n, stride_grad_latent_d,

    stride_wu_row, stride_wu_col,
    stride_w1_row, stride_w1_col,
    stride_wv_row, stride_wv_col,

    stride_grad_wu_row, stride_grad_wu_col,
    stride_grad_w1_row, stride_grad_w1_col,
    stride_grad_wv_row, stride_grad_wv_col,
    stride_grad_score_bs, stride_grad_score_k,

    # Block sizes
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DHIDDEN: tl.constexpr,
):
    """
    Backward pass kernel using the same reordered optimization:
    1. Compute x_proj = x @ W_u ONCE (same as forward)
    2. For each expert, compute gradients efficiently
    3. Accumulate gradients with minimal redundant computation
    """
    pid = tl.program_id(0)
    if pid >= batch_seq_size:
        return

    d_offs = tl.arange(0, BLOCK_DMODEL)
    h_offs = tl.arange(0, BLOCK_DHIDDEN)
    h_mask = h_offs < d_hidden

    # Step 1: Recompute x_proj (same as forward - key optimization!)
    x_proj = tl.zeros([BLOCK_DHIDDEN], dtype=tl.float32)
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

    # Initialize gradient accumulators
    grad_x = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    grad_x_proj = tl.zeros([BLOCK_DHIDDEN], dtype=tl.float32)

    # Step 2: Process each expert (backward through expert loop)
    for k in range(top_k):
        expert_idx = tl.load(indices_ptr + pid * stride_idx_bs + k * stride_idx_k)
        score = tl.load(scores_ptr + pid * stride_score_bs + k * stride_score_k)

        # Recompute forward values needed for backward
        # h = GELU(latent @ W1)
        h = tl.zeros([BLOCK_DHIDDEN], dtype=tl.float32)
        latent = tl.zeros([d_latent], dtype=tl.float32)

        for l in range(d_latent):
            latent_val = tl.load(latent_ptr + expert_idx * stride_latent_n + l * stride_latent_d)
            latent = tl.where(l < d_latent, latent_val, latent)  # Store for later
            w1_row = tl.load(
                w1_ptr + l * stride_w1_row + h_offs * stride_w1_col,
                mask=h_mask,
                other=0.0,
            )
            h += latent_val * w1_row

        # GELU activation
        h_pre_gelu = h
        sigmoid_arg = 1.702 * h_pre_gelu
        h = h_pre_gelu * tl.sigmoid(sigmoid_arg)

        # Recompute dot and activation
        dot = tl.sum(h * x_proj)
        dot_sigmoid_arg = 1.702 * dot
        activation = dot * tl.sigmoid(dot_sigmoid_arg) * score

        # === Backward pass ===

        # Load grad_output for this token
        grad_out_token = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
        for d_start in range(0, d_model, BLOCK_DMODEL):
            d_chunk = d_start + d_offs
            x_mask = d_chunk < d_model
            grad_chunk = tl.load(
                grad_output_ptr + pid * stride_grad_out_bs + d_chunk * stride_grad_out_d,
                mask=x_mask,
                other=0.0,
            )
            grad_out_token = tl.where(x_mask, grad_chunk, grad_out_token)

        # Backward through: output += activation * (h @ W_v)
        # grad_activation = sum(grad_output * (h @ W_v))
        grad_activation = tl.zeros([1], dtype=tl.float32)
        grad_h_from_output = tl.zeros([BLOCK_DHIDDEN], dtype=tl.float32)

        for d_start in range(0, d_model, BLOCK_DMODEL):
            d_chunk = d_start + d_offs
            x_mask = d_chunk < d_model

            # Load W_v for this chunk
            wv_block = tl.load(
                wv_ptr + h_offs[:, None] * stride_wv_row + d_chunk[None, :] * stride_wv_col,
                mask=h_mask[:, None] & x_mask[None, :],
                other=0.0,
            )

            # h @ W_v for this chunk
            h_wv_chunk = tl.sum(h[:, None] * wv_block, axis=0)

            # grad_activation += grad_output * (h @ W_v)
            grad_out_chunk = tl.load(
                grad_output_ptr + pid * stride_grad_out_bs + d_chunk * stride_grad_out_d,
                mask=x_mask,
                other=0.0,
            )
            grad_activation += tl.sum(grad_out_chunk * h_wv_chunk)

            # grad_h += activation * grad_output @ W_v.T
            grad_h_from_output += tl.sum(grad_out_chunk[None, :] * wv_block, axis=1) * activation

            # grad_W_v += h^T @ (activation * grad_output) - accumulated atomically later

        # Backward through: activation = GELU(dot) * score
        # GELU'(x) = sigmoid(1.702*x) * (1 + 1.702*x*(1-sigmoid(1.702*x)))
        sigmoid_val = tl.sigmoid(dot_sigmoid_arg)
        gelu_grad = sigmoid_val * (1.0 + dot_sigmoid_arg * (1.0 - sigmoid_val))
        grad_dot = grad_activation * gelu_grad * score
        grad_score = grad_activation * (dot * sigmoid_val)

        # Store grad_score
        tl.store(
            grad_scores_ptr + pid * stride_grad_score_bs + k * stride_grad_score_k,
            grad_score,
        )

        # Backward through: dot = sum(h * x_proj)
        grad_h_from_dot = grad_dot * x_proj
        grad_x_proj += grad_dot * h

        # Combine gradients for h
        grad_h_total = grad_h_from_output + grad_h_from_dot

        # Backward through: h = GELU(h_pre_gelu)
        sigmoid_h = tl.sigmoid(sigmoid_arg)
        gelu_grad_h = sigmoid_h * (1.0 + sigmoid_arg * (1.0 - sigmoid_h))
        grad_h_pre_gelu = grad_h_total * gelu_grad_h

        # Backward through: h_pre_gelu = latent @ W1
        # grad_latent = grad_h_pre_gelu @ W1.T
        # grad_W1 = latent^T @ grad_h_pre_gelu - accumulated atomically later

    # Step 3: Backward through x_proj = x @ W_u (computed ONCE - key optimization!)
    # grad_x += grad_x_proj @ W_u.T
    for d_start in range(0, d_model, BLOCK_DMODEL):
        d_chunk = d_start + d_offs
        x_mask = d_chunk < d_model

        wu_chunk = tl.load(
            wu_ptr + d_chunk[:, None] * stride_wu_row + h_offs[None, :] * stride_wu_col,
            mask=x_mask[:, None] & h_mask[None, :],
            other=0.0,
        )

        grad_x_chunk = tl.sum(grad_x_proj[None, :] * wu_chunk, axis=1)

        # Store grad_x
        out_offset = pid * stride_grad_x_bs + d_chunk * stride_grad_x_d
        old_grad = tl.load(grad_x_ptr + out_offset, mask=x_mask, other=0.0)
        tl.store(grad_x_ptr + out_offset, old_grad + grad_x_chunk, mask=x_mask)


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


class MoEFunction(torch.autograd.Function):
    """
    Custom autograd Function for MoE forward/backward with Triton kernels.
    Uses the same "reordered" optimization in both forward and backward:
    - Forward: Compute x @ W_u once, reuse for all experts
    - Backward: Recompute x @ W_u once, backprop efficiently
    """

    @staticmethod
    def forward(ctx, x_flat, expert_latents, indices, scores, W_u, W1, W_v,
                d_model, d_latent, d_hidden, top_k):
        """
        Forward pass using Triton kernel.

        Args:
            x_flat: [n_tokens, d_model]
            expert_latents: [num_experts, d_latent]
            indices: [n_tokens, top_k]
            scores: [n_tokens, top_k]
            W_u: [d_model, d_hidden]
            W1: [d_latent, d_hidden]
            W_v: [d_hidden, d_model]
        """
        n_tokens = x_flat.shape[0]
        output = torch.zeros_like(x_flat)

        # Block sizes
        BLOCK_DMODEL = min(1024, d_model)
        BLOCK_DHIDDEN = min(64, d_hidden)

        # Launch kernel
        grid = (n_tokens,)

        moe_reorder_fwd_kernel[grid](
            # Inputs
            x_flat,
            expert_latents,
            indices,
            scores,

            # Weight matrices
            W_u,
            W1,
            W_v,

            # Output
            output,

            # Dimensions
            n_tokens,
            d_model,
            d_latent,
            d_hidden,
            top_k,

            # Strides
            x_flat.stride(0), x_flat.stride(1),
            indices.stride(0), indices.stride(1),
            scores.stride(0), scores.stride(1),
            output.stride(0), output.stride(1),
            expert_latents.stride(0), expert_latents.stride(1),

            W_u.stride(0), W_u.stride(1),
            W1.stride(0), W1.stride(1),
            W_v.stride(0), W_v.stride(1),

            # Block sizes
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DHIDDEN=BLOCK_DHIDDEN,
        )

        # Save for backward
        ctx.save_for_backward(x_flat, expert_latents, indices, scores, W_u, W1, W_v)
        ctx.d_model = d_model
        ctx.d_latent = d_latent
        ctx.d_hidden = d_hidden
        ctx.top_k = top_k

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using Triton kernel.
        Applies the same reordered optimization as forward pass.
        """
        x_flat, expert_latents, indices, scores, W_u, W1, W_v = ctx.saved_tensors
        d_model = ctx.d_model
        d_latent = ctx.d_latent
        d_hidden = ctx.d_hidden
        top_k = ctx.top_k

        n_tokens = x_flat.shape[0]
        num_experts = expert_latents.shape[0]

        # Initialize gradients
        grad_x = torch.zeros_like(x_flat)
        grad_latent = torch.zeros_like(expert_latents)
        grad_wu = torch.zeros_like(W_u)
        grad_w1 = torch.zeros_like(W1)
        grad_wv = torch.zeros_like(W_v)
        grad_scores = torch.zeros_like(scores)

        # Block sizes
        BLOCK_DMODEL = min(1024, d_model)
        BLOCK_DHIDDEN = min(64, d_hidden)

        # Launch backward kernel
        grid = (n_tokens,)

        moe_reorder_bwd_kernel[grid](
            # Inputs from forward pass
            x_flat,
            expert_latents,
            indices,
            scores,

            W_u,
            W1,
            W_v,

            # Gradient input
            grad_output.contiguous(),

            # Gradient outputs
            grad_x,
            grad_latent,
            grad_wu,
            grad_w1,
            grad_wv,
            grad_scores,

            # Dimensions
            n_tokens,
            d_model,
            d_latent,
            d_hidden,
            top_k,

            # Strides
            x_flat.stride(0), x_flat.stride(1),
            indices.stride(0), indices.stride(1),
            scores.stride(0), scores.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_x.stride(0), grad_x.stride(1),
            expert_latents.stride(0), expert_latents.stride(1),
            grad_latent.stride(0), grad_latent.stride(1),

            W_u.stride(0), W_u.stride(1),
            W1.stride(0), W1.stride(1),
            W_v.stride(0), W_v.stride(1),

            grad_wu.stride(0), grad_wu.stride(1),
            grad_w1.stride(0), grad_w1.stride(1),
            grad_wv.stride(0), grad_wv.stride(1),
            grad_scores.stride(0), grad_scores.stride(1),

            # Block sizes
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DHIDDEN=BLOCK_DHIDDEN,
        )

        # Return gradients for all inputs
        # Order matches forward: x_flat, expert_latents, indices, scores, W_u, W1, W_v, + ctx vars
        return grad_x, grad_latent, None, grad_scores, grad_wu, grad_w1, grad_wv, None, None, None, None


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

        # Process each routing head using custom autograd function
        for head_idx in range(self.num_heads):
            head_scores = scores[:, head_idx, :].contiguous()
            head_indices = expert_indices[:, head_idx, :].contiguous()

            # Use MoEFunction for forward/backward with Triton kernels
            head_output = MoEFunction.apply(
                x_flat,
                expert_latents_contig,
                head_indices,
                head_scores,
                W_u,
                W1,
                W_v,
                self.d_model,
                self.d_latent,
                self.d_hidden,
                self.top_k
            )

            output += head_output

        # Average across heads
        output = output / self.num_heads
        return output.view(batch_size, seq_len, self.d_model)
