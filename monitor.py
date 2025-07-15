"""
Training monitoring and visualization utilities.

Copyright (C) 2025 Wesley Medford, Chris McCormick, Eve Callicoat

This program is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
For commercial licensing, contact: wryanmedford@gmail.com
"""

import os
import json
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
from IPython.display import display, clear_output


class TrainingMonitor:
    """Combined logger and visualizer for training metrics"""
    
    def __init__(self, run_id, log_dir="./training_logs", plot_window=500):
        self.run_id = run_id
        self.log_dir = log_dir
        self.plot_window = plot_window
        
        # Setup logging files
        os.makedirs(log_dir, exist_ok=True)
        self.json_log_path = os.path.join(log_dir, f"training_{run_id}.jsonl")
        self.csv_log_path = os.path.join(log_dir, f"metrics_{run_id}.csv")
        
        # Initialize CSV with headers
        with open(self.csv_log_path, 'w') as f:
            f.write("timestamp,epoch,batch,global_step,loss,perplexity,lr,gpu_memory_mb,avg_loss_100\n")
        
        # Initialize metrics storage
        self.metrics = {
            'steps': deque(maxlen=plot_window),
            'loss': deque(maxlen=plot_window),
            'perplexity': deque(maxlen=plot_window),
            'lr': deque(maxlen=plot_window),
            'gpu_memory': deque(maxlen=plot_window),
            'avg_loss': deque(maxlen=plot_window)
        }
        
        self.recent_losses = deque(maxlen=100)
        
        # Setup matplotlib
        self.fig = None
        self.axes = None
        self.plot_initialized = False
        
        # Validation metrics
        self.val_history = {
            'epoch': [],
            'loss': [],
            'perplexity': []
        }
        
        print(f"Logging to: {self.json_log_path}")
        print(f"Metrics CSV: {self.csv_log_path}")
    
    def log_iteration(self, epoch, batch, global_step, loss, lr, 
                     batch_size=None, seq_len=None, extra_metrics=None):
        """Log a training iteration"""
        timestamp = datetime.now().isoformat()
        
        # Calculate derived metrics
        perplexity = np.exp(loss) if loss < 50 else float('inf')
        gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        # Update running average
        self.recent_losses.append(loss)
        avg_loss = np.mean(self.recent_losses)
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'run_id': self.run_id,
            'epoch': epoch,
            'batch': batch,
            'global_step': global_step,
            'loss': float(loss),
            'perplexity': float(perplexity),
            'lr': float(lr),
            'gpu_memory_mb': float(gpu_memory_mb),
            'avg_loss_100': float(avg_loss)
        }
        
        if batch_size:
            log_entry['batch_size'] = batch_size
        if seq_len:
            log_entry['seq_len'] = seq_len
        if extra_metrics:
            log_entry.update(extra_metrics)
        
        # Write to JSON log
        with open(self.json_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Write to CSV
        csv_row = f"{timestamp},{epoch},{batch},{global_step},{loss:.6f}," \
                  f"{perplexity:.2f},{lr:.8f},{gpu_memory_mb:.1f},{avg_loss:.6f}\n"
        with open(self.csv_log_path, 'a') as f:
            f.write(csv_row)
        
        # Update metrics for plotting
        self.metrics['steps'].append(global_step)
        self.metrics['loss'].append(loss)
        self.metrics['perplexity'].append(perplexity)
        self.metrics['lr'].append(lr)
        self.metrics['gpu_memory'].append(gpu_memory_mb)
        self.metrics['avg_loss'].append(avg_loss)
        
        return log_entry
    
    def log_validation(self, epoch, val_loss, val_perplexity):
        """Log validation metrics"""
        self.val_history['epoch'].append(epoch)
        self.val_history['loss'].append(val_loss)
        self.val_history['perplexity'].append(val_perplexity)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_id,
            'type': 'validation',
            'epoch': epoch,
            'val_loss': float(val_loss),
            'val_perplexity': float(val_perplexity)
        }
        
        with open(self.json_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def plot_metrics(self, update_frequency=50):
        """Update real-time plots in notebook"""
        if len(self.metrics['steps']) < 2:
            return
        
        current_step = self.metrics['steps'][-1]
        
        # Only update plot every N steps
        if current_step % update_frequency != 0 and current_step > update_frequency:
            return
        
        if not self.plot_initialized:
            self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
            self.fig.suptitle(f'Training Metrics - Run {self.run_id}', fontsize=16)
            self.plot_initialized = True
        
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        steps = list(self.metrics['steps'])
        
        # Plot 1: Loss
        ax = self.axes[0, 0]
        ax.plot(steps, list(self.metrics['loss']), 'b-', alpha=0.3, label='Raw Loss')
        ax.plot(steps, list(self.metrics['avg_loss']), 'r-', linewidth=2, label='Avg Loss (100)')
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.set_ylim(bottom=0)
        
        # Plot 2: Perplexity
        ax = self.axes[0, 1]
        perplexity_values = [p for p in self.metrics['perplexity'] if p < 1000]
        if perplexity_values:
            ax.plot(steps[-len(perplexity_values):], perplexity_values, 'g-')
            ax.set_xlabel('Global Step')
            ax.set_ylabel('Perplexity')
            ax.set_title('Perplexity')
            ax.set_ylim(bottom=0)
        
        # Plot 3: Learning Rate
        ax = self.axes[1, 0]
        ax.plot(steps, list(self.metrics['lr']), 'orange')
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Plot 4: GPU Memory
        ax = self.axes[1, 1]
        ax.plot(steps, list(self.metrics['gpu_memory']), 'purple')
        ax.set_xlabel('Global Step')
        ax.set_ylabel('GPU Memory (MB)')
        ax.set_title('GPU Memory Usage')
        
        # Add validation points if available
        if self.val_history['epoch']:
            val_steps = [epoch * (steps[-1] // max(1, self.val_history['epoch'][-1])) 
                        for epoch in self.val_history['epoch']]
            
            self.axes[0, 0].scatter(val_steps, self.val_history['loss'], 
                                  color='red', s=100, marker='*', 
                                  label='Validation', zorder=5)
            self.axes[0, 0].legend()
        
        plt.tight_layout()
        
        # Display in notebook or save
        try:
            clear_output(wait=True)
            display(self.fig)
        except:
            # If not in notebook, save to file
            plt.savefig(os.path.join(self.log_dir, f'metrics_{self.run_id}.png'))
        
        plt.pause(0.01)
    
    def get_summary_stats(self):
        """Get summary statistics for the current run"""
        if not self.metrics['loss']:
            return {}
        
        recent_losses = list(self.metrics['loss'])[-100:]
        
        return {
            'total_steps': len(self.metrics['steps']),
            'current_loss': self.metrics['loss'][-1],
            'avg_recent_loss': np.mean(recent_losses),
            'min_loss': min(self.metrics['loss']),
            'current_lr': self.metrics['lr'][-1],
            'current_gpu_mb': self.metrics['gpu_memory'][-1] if self.metrics['gpu_memory'] else 0,
            'best_val_perplexity': min(self.val_history['perplexity']) if self.val_history['perplexity'] else None
        }