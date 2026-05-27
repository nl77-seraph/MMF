import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
import argparse
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

# Add module path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
def setup_distributed_training(rank, world_size, config):
    """
    Set up the distributed training environment.
    
    Args:
        rank: Rank of the current process.
        world_size: Total number of processes.
        config: Configuration dictionary.
    """
    # Set CUDA device.
    torch.cuda.set_device(rank)
    
    # Initialize the process group.
    os.environ['MASTER_ADDR'] = config.get('master_addr', 'localhost')
    os.environ['MASTER_PORT'] = config.get('master_port', '12355')
    
    dist.init_process_group(
        backend=config.get('dist_backend', 'nccl'),
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    print(f"Distributed training initialized - Rank: {rank}/{world_size}")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def cleanup_distributed_training():
    """Clean up the distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check whether this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0

def validate_gpu_config(config):
    """
    Validate the GPU configuration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        validated_config: Validated configuration.
        is_valid: Whether the configuration is valid.
        error_msg: Error message.
    """
    try:
        # Check CUDA availability.
        if not torch.cuda.is_available():
            if config.get('use_distributed', False):
                return config, False, "CUDA is unavailable, so GPU training cannot run"
            else:
                print("CUDA is unavailable; using CPU training")
                config['use_distributed'] = False
                return config, True, ""
        
        # Check the number of GPUs.
        available_gpus = torch.cuda.device_count()
        print(f"Detected {available_gpus} GPUs")
        
        # Validate requested GPU IDs.
        requested_gpus = config.get('gpus', [0])
        if not isinstance(requested_gpus, list):
            return config, False, "'gpus' must be a list"
        
        for gpu_id in requested_gpus:
            if gpu_id >= available_gpus:
                return config, False, f"GPU {gpu_id} does not exist; only {available_gpus} GPUs are available"
        
        # Set distributed training mode.
        num_gpus = len(requested_gpus)
        if num_gpus > 1:
            config['use_distributed'] = True
            print(f"Using {num_gpus} GPUs for distributed training: {requested_gpus}")
        else:
            config['use_distributed'] = False
            print(f"Using single-GPU training: GPU {requested_gpus[0]}")
        
        return config, True, ""
        
    except Exception as e:
        return config, False, f"GPU configuration validation failed: {e}"


def get_args():
    parser = argparse.ArgumentParser(description='Enhanced Training with Mixed Scheme C')
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to the config file')
    
    
    args = parser.parse_args()
    return args

def get_final_config():
    config = {}
    # Load config file.
    args = get_args()
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    else:
        print('Please check the config file')
    
    # Validate GPU configuration.
    config, is_valid, error_msg = validate_gpu_config(config)
    if not is_valid:
        print(error_msg)
        return

    print("Starting training...")
    print(f"Configuration:")
    print(f"  - Distributed: {config['use_distributed']}")
    print(f"  - GPU: {config['gpus']}")
    print(f"  - DF uses SE: {config['use_se_in_df']}")
    
    return config
if __name__ == '__main__':
    print("test_config")
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(config)




