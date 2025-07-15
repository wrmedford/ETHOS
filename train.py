"""
ETHOS Training Script

Main training loop with DeepSpeed integration.

Copyright (C) 2025 Wesley Medford, Chris McCormick, Eve Callicoat

This program is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
For commercial licensing, contact: wryanmedford@gmail.com
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed
import argparse
import yaml
from tqdm import tqdm
from datetime import datetime
import numpy as np

from model import CompressedMoEModel
from data import get_data_loaders
from monitor import TrainingMonitor


class Config:
    """Configuration class from YAML or defaults"""
    def __init__(self, config_path=None):
        # Default configuration
        self.vocab_size = 100277
        self.d_model = 1024
        self.num_layers = 16
        self.num_heads = 16
        self.num_dense_layers = 2
        self.num_moe_layers = 14
        self.q_lora_rank = 768
        self.kv_lora_rank = 256
        self.v_head_dim = 64
        self.qk_nope_head_dim = 64
        self.qk_rope_head_dim = 64
        self.num_experts = 512**2
        self.d_latent = 128
        self.d_intermediate_hypernet = 512
        self.top_k = 16
        self.num_routing_heads = 8
        self.d_query = 512
        self.d_ffn_intermediate = 4096
        self.max_seq_len = 4096
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10000.0
        
        # Training defaults
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.num_epochs = 3
        self.data_percentage = 0.01
        
        # Load from YAML if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                for key, value in config_dict.items():
                    setattr(self, key, value)


def setup_deepspeed():
    """Initialize DeepSpeed configuration"""
    ds_config = {
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 2,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 1e-5,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 1000
            }
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 1e8,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }
    return ds_config


def calculate_params(config):
    """Calculate model parameter counts"""
    # Dense layer params
    dense_active = config.num_dense_layers * (
        config.d_model * config.q_lora_rank + 
        config.q_lora_rank * config.num_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim) +
        config.d_model * (config.kv_lora_rank + config.qk_rope_head_dim) + 
        config.kv_lora_rank * config.num_heads * (config.qk_nope_head_dim + config.v_head_dim) +
        config.num_heads * config.v_head_dim * config.d_model +
        3 * config.d_model * config.d_ffn_intermediate
    )
    
    # MoE layer params
    moe_active = config.num_moe_layers * (
        config.d_model * config.q_lora_rank + 
        config.q_lora_rank * config.num_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim) +
        config.d_model * (config.kv_lora_rank + config.qk_rope_head_dim) + 
        config.kv_lora_rank * config.num_heads * (config.qk_nope_head_dim + config.v_head_dim) +
        config.num_heads * config.v_head_dim * config.d_model +
        config.num_routing_heads * config.d_model * config.d_query +
        config.num_routing_heads * config.top_k * 2 * config.d_model
    )
    
    embedding_params = config.vocab_size * config.d_model
    total_active = dense_active + moe_active + embedding_params
    
    return total_active


def train():
    """Main training function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Initialize distributed training
    if args.local_rank == -1:
        # Single GPU training
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['LOCAL_RANK'] = '0'
        
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=0, world_size=1)
    
    # Initialize model
    model = CompressedMoEModel(config)
    
    # Calculate and print parameter counts
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active_params = calculate_params(config)
    print(f"Total Trainable Parameters: {total_params / 1e9:.2f}B")
    print(f"Active Parameters per Token: ~{active_params / 1e6:.1f}M")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(config)
    
    # Initialize DeepSpeed
    ds_config = setup_deepspeed()
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # Initialize monitoring
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    monitor = TrainingMonitor(run_id)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=100277)  # EOT token
    
    # Training loop
    for epoch in range(config.num_epochs):
        model_engine.train()
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(train_progress):
            inputs = batch['input_ids'].to(model_engine.device)
            batch_size, seq_len = inputs.shape
            labels = inputs.clone()
            
            # Forward pass
            logits = model_engine(inputs)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
            
            # Backward pass
            model_engine.backward(loss)
            model_engine.step()
            
            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            global_step = model_engine.global_steps
            
            log_data = monitor.log_iteration(
                epoch=epoch + 1,
                batch=batch_idx,
                global_step=global_step,
                loss=loss.item(),
                lr=current_lr,
                batch_size=batch_size,
                seq_len=seq_len
            )
            
            train_progress.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ppl': f"{np.exp(loss.item()):.2f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Update plots periodically
            if global_step % 50 == 0:
                monitor.plot_metrics(update_frequency=1)
        
        # Validation
        print(f"\nRunning validation for epoch {epoch + 1}...")
        model_engine.eval()
        val_loss_total = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs = batch['input_ids'].to(model_engine.device)
                labels = inputs.clone()
                
                logits = model_engine(inputs)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
                
                val_loss_total += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss_total / val_steps
        val_perplexity = np.exp(avg_val_loss)
        
        monitor.log_validation(epoch + 1, avg_val_loss, val_perplexity)
        
        print(f"\nEpoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        
        # Save checkpoint
        checkpoint_dir = f"checkpoints/{run_id}/epoch_{epoch+1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_engine.save_checkpoint(checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    print("\nTraining complete!")
    stats = monitor.get_summary_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    train()