"""
Test script for verifying the backward kernel implementation.

This tests:
1. Gradient shapes are correct
2. Gradients flow through all components
3. Backward kernel produces non-zero gradients
"""

import torch
import torch.nn as nn
from ethos_kernels import FusedLowRankMoE_Reordered

def test_backward_kernel():
    """Test that backward kernel computes gradients correctly."""

    # Setup
    torch.manual_seed(42)
    device = 'cuda'

    # Small model for testing
    d_model = 256
    num_experts = 16  # 4x4 for product key
    top_k = 4
    d_latent = 16
    d_intermediate_hypernet = 128
    d_query = 32
    num_routing_heads = 2
    batch_size = 2
    seq_len = 128

    print("=" * 80)
    print("Testing ETHOS MoE Backward Kernel")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_experts: {num_experts}")
    print(f"  top_k: {top_k}")
    print(f"  num_routing_heads: {num_routing_heads}")
    print(f"  batch_size: {batch_size}, seq_len: {seq_len}")
    print()

    # Create model
    moe = FusedLowRankMoE_Reordered(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        d_latent=d_latent,
        d_intermediate_hypernet=d_intermediate_hypernet,
        d_query=d_query,
        num_routing_heads=num_routing_heads
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    print("Forward pass...")
    # Forward pass
    output = moe(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == x.shape, "Output shape mismatch!"

    # Create dummy loss
    loss = output.sum()

    print("\nBackward pass...")
    # Backward pass
    loss.backward()

    # Check gradients
    print("\nChecking gradients...")

    # 1. Input gradient
    assert x.grad is not None, "Input gradient is None!"
    assert x.grad.shape == x.shape, "Input gradient shape mismatch!"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN!"
    assert (x.grad.abs() > 0).any(), "Input gradient is all zeros!"
    print(f"  ✓ Input gradient: shape {x.grad.shape}, norm {x.grad.norm():.4f}")

    # 2. Expert latent gradients
    assert moe.expert_latents.weight.grad is not None, "Expert latent gradient is None!"
    assert not torch.isnan(moe.expert_latents.weight.grad).any(), "Expert latent gradient contains NaN!"
    assert (moe.expert_latents.weight.grad.abs() > 0).any(), "Expert latent gradient is all zeros!"
    print(f"  ✓ Expert latent gradient: shape {moe.expert_latents.weight.grad.shape}, norm {moe.expert_latents.weight.grad.norm():.4f}")

    # 3. Generation network gradients
    gen_net = moe.generation_network

    # W1 gradient
    w1 = gen_net.net[0].weight
    assert w1.grad is not None, "W1 gradient is None!"
    assert not torch.isnan(w1.grad).any(), "W1 gradient contains NaN!"
    assert (w1.grad.abs() > 0).any(), "W1 gradient is all zeros!"
    print(f"  ✓ W1 gradient: shape {w1.grad.shape}, norm {w1.grad.norm():.4f}")

    # W2 gradient (contains W_u and W_v)
    w2 = gen_net.net[2].weight
    assert w2.grad is not None, "W2 gradient is None!"
    assert not torch.isnan(w2.grad).any(), "W2 gradient contains NaN!"
    assert (w2.grad.abs() > 0).any(), "W2 gradient is all zeros!"
    print(f"  ✓ W2 gradient: shape {w2.grad.shape}, norm {w2.grad.norm():.4f}")

    # 4. Router gradients
    for i, qp in enumerate(moe.router.query_projs):
        assert qp.weight.grad is not None, f"Router query_proj {i} gradient is None!"
        assert not torch.isnan(qp.weight.grad).any(), f"Router query_proj {i} gradient contains NaN!"
        print(f"  ✓ Router query_proj[{i}] gradient: shape {qp.weight.grad.shape}, norm {qp.weight.grad.norm():.4f}")

    print("\n" + "=" * 80)
    print("✓ All gradient checks passed!")
    print("=" * 80)

    return True

if __name__ == "__main__":
    try:
        test_backward_kernel()
        print("\n✅ Test PASSED!")
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
