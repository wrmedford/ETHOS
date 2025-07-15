"""
Data loading utilities for ETHOS training.

Copyright (C) 2025 Wesley Medford, Chris McCormick, Eve Callicoat

This program is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
For commercial licensing, contact: wryanmedford@gmail.com
"""

import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
import tiktoken


def get_tokenizer():
    """Get the tiktoken cl100k_base tokenizer"""
    return tiktoken.get_encoding("cl100k_base")


def tokenize_function(examples, tokenizer, max_seq_len=4096):
    """Tokenize text examples"""
    token_ids = tokenizer.encode(examples["text"], allowed_special="all")
    token_ids = token_ids[:max_seq_len]
    return {"input_ids": token_ids}


def collate_batch(batch, pad_token_id):
    """Collate function for DataLoader"""
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        input_ids, 
        batch_first=True, 
        padding_value=pad_token_id
    )
    return {'input_ids': padded_inputs}


def get_data_loaders(config):
    """Get train and validation data loaders"""
    # Initialize tokenizer
    enc = get_tokenizer()
    
    # Cache directory
    cache_dir = "./c4_tokenized_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    train_cache_path = os.path.join(cache_dir, f"train_{config.data_percentage}")
    val_cache_path = os.path.join(cache_dir, f"val_{config.data_percentage}")
    
    # Load or create tokenized datasets
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print("Loading tokenized datasets from cache...")
        train_tokenized = Dataset.load_from_disk(train_cache_path)
        val_tokenized = Dataset.load_from_disk(val_cache_path)
        print(f"Loaded from cache - Train: {len(train_tokenized):,}, Val: {len(val_tokenized):,}")
    else:
        print("Downloading C4 dataset...")
        if config.data_percentage < 1.0:
            split_percentage = int(config.data_percentage * 100)
            train_dataset = load_dataset("allenai/c4", "en", split=f"train[:{split_percentage}%]")
            val_dataset = load_dataset("allenai/c4", "en", split=f"validation[:{split_percentage}%]")
        else:
            train_dataset = load_dataset("allenai/c4", "en", split="train")
            val_dataset = load_dataset("allenai/c4", "en", split="validation")
        
        print(f"Train samples: {len(train_dataset):,}, Val samples: {len(val_dataset):,}")
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_tokenized = train_dataset.map(
            lambda x: tokenize_function(x, enc, config.max_seq_len),
            remove_columns=["text", "timestamp", "url"],
            num_proc=64,
            desc="Tokenizing train set"
        )
        val_tokenized = val_dataset.map(
            lambda x: tokenize_function(x, enc, config.max_seq_len),
            remove_columns=["text", "timestamp", "url"],
            num_proc=64,
            desc="Tokenizing validation set"
        )
        
        # Save to cache
        print("Saving tokenized datasets to cache...")
        train_tokenized.save_to_disk(train_cache_path)
        val_tokenized.save_to_disk(val_cache_path)
        print("Cache saved successfully!")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_tokenized,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, enc.eot_token),
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_tokenized,
        batch_size=config.batch_size,
        collate_fn=lambda batch: collate_batch(batch, enc.eot_token),
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader