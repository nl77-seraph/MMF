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

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
def setup_distributed_training(rank, world_size, config):
    """
    设置分布式训练环境
    
    Args:
        rank: 当前进程的rank
        world_size: 总进程数
        config: 配置字典
    """
    # 设置CUDA设备
    torch.cuda.set_device(rank)
    
    # 初始化进程组
    os.environ['MASTER_ADDR'] = config.get('master_addr', 'localhost')
    os.environ['MASTER_PORT'] = config.get('master_port', '12355')
    
    dist.init_process_group(
        backend=config.get('dist_backend', 'nccl'),
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    print(f"🚀 分布式训练初始化完成 - Rank: {rank}/{world_size}")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True



def cleanup_distributed_training():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """检查是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

def validate_gpu_config(config):
    """
    验证GPU配置的合法性
    
    Args:
        config: 配置字典
        
    Returns:
        validated_config: 验证后的配置
        is_valid: 是否有效
        error_msg: 错误信息
    """
    try:
        # 检查是否有CUDA支持
        if not torch.cuda.is_available():
            if config.get('use_distributed', False):
                return config, False, "❌ CUDA不可用，无法进行GPU训练"
            else:
                print("⚠️ CUDA不可用，将使用CPU训练")
                config['use_distributed'] = False
                return config, True, ""
        
        # 检查GPU数量
        available_gpus = torch.cuda.device_count()
        print(f"🔍 检测到 {available_gpus} 个GPU")
        
        # 验证请求的GPU是否存在
        requested_gpus = config.get('gpus', [0])
        if not isinstance(requested_gpus, list):
            return config, False, "❌ 'gpus' 配置必须是列表"
        
        for gpu_id in requested_gpus:
            if gpu_id >= available_gpus:
                return config, False, f"❌ GPU {gpu_id} 不存在（只有 {available_gpus} 个GPU）"
        
        # 设置分布式训练模式
        num_gpus = len(requested_gpus)
        if num_gpus > 1:
            config['use_distributed'] = True
            print(f"✅ 将使用 {num_gpus} 个GPU进行分布式训练: {requested_gpus}")
        else:
            config['use_distributed'] = False
            print(f"✅ 将使用单GPU训练: GPU {requested_gpus[0]}")
        
        return config, True, ""
        
    except Exception as e:
        return config, False, f"❌ GPU配置验证失败: {e}"


def get_args():
    parser = argparse.ArgumentParser(description='Enhanced Training with Mixed Scheme C')
    parser.add_argument('--config', type=str, default='configs/config.json',help='配置文件路径')
    
    
    args = parser.parse_args()
    return args

def get_final_config():
    config = {}
    # 加载配置文件
    args = get_args()
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    else:
        print('请检查配置文件')
    
    # 验证GPU配置
    config, is_valid, error_msg = validate_gpu_config(config)
    if not is_valid:
        print(error_msg)
        return

    print("开始训练...")
    print(f"📋 配置:")
    print(f"  - 分布式: {config['use_distributed']}")
    print(f"  - GPU: {config['gpus']}")
    print(f"  - DF使用SE: {config['use_se_in_df']}")
    
    return config
if __name__ == '__main__':
    print("test_config")
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(config)




