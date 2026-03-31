

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
import sys
import os

# 添加路径以导入自定义模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from meta_traffic_dataset import QueryTrafficDataset, SupportTrafficDataset

import torch.distributed as dist
def is_main_process():
    """检查是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0

class MetaTrafficDataLoader:
    """
    Meta学习数据加载器
    结合查询集和支持集，输出格式与MultiMetaFingerNet.forward()完全兼容
    """
    
    def __init__(self,
                 query_json_path: str,
                 query_files_dir: str,
                 support_root_dir: str,
                 activated_classes: List[int] = None,
                 support_target_length: int = 10000,
                 query_target_length: int = 20000,
                 shots_per_class: int = 1,
                 batch_size: int = 4,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 random_sampling: bool = False):
        """
        Args:
            query_json_path: 查询集索引JSON文件路径
            query_files_dir: 查询集数据文件目录
            support_root_dir: 支持集根目录
            activated_classes: 激活的类别列表，默认0-59
            target_length: 目标序列长度
            shots_per_class: 每个类别的支持样本数
            batch_size: 批大小
            shuffle: 是否打乱
            num_workers: 数据加载进程数
            random_sampling: 是否使用随机采样模式（用于训练）
        """
        self.activated_classes = activated_classes if activated_classes else list(range(60))  # 0-59
        self.support_target_length = support_target_length
        self.query_target_length = query_target_length
        self.shots_per_class = shots_per_class
        self.batch_size = batch_size
        self.random_sampling = random_sampling
        if is_main_process():
            print(f"MetaTrafficDataLoader初始化...")
            print(f"  - 激活类别: {len(self.activated_classes)}个 (0-{max(self.activated_classes)})")
            print(f"  - 每类样本数: {shots_per_class}")
            print(f"  - 批大小: {batch_size}")
            print(f"  - 随机采样: {random_sampling}")
        
        # 初始化查询集数据集
        self.query_dataset = QueryTrafficDataset(
            json_index_path=query_json_path,
            query_files_dir=query_files_dir,
            target_length=query_target_length,
            activated_classes=self.activated_classes
        )
        
        # 初始化支持集数据集
        self.support_dataset = SupportTrafficDataset(
            support_root_dir=support_root_dir,
            activated_classes=self.activated_classes,
            target_length=support_target_length,
            shots_per_class=shots_per_class,
            random_sampling=random_sampling
        )
        
        # 创建查询集DataLoader
        self.query_loader = DataLoader(
            self.query_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._query_collate_fn
        )
        
        if not self.random_sampling:
            # 固定采样模式：预加载所有支持集数据
            self.support_data, self.support_masks, self.class_order = self.support_dataset.get_all_support_data()
            if is_main_process():
                print(f"  - 支持集形状: {self.support_data.shape}")
        else:
            # 随机采样模式：每次迭代时动态生成支持集
            self.class_order = sorted(self.activated_classes)
            if is_main_process():
                print(f"  - 支持集: 动态随机采样模式")
        if is_main_process():
            print(f"  - 查询集样本数: {len(self.query_dataset)}")
            print(f"  - 数据加载器初始化完成！")
    
    def _query_collate_fn(self, batch):
        """查询集的collate函数"""
        query_data_list = []
        query_labels_list = []
        metadata_list = []
        
        for query_data, query_labels, metadata in batch:
            query_data_list.append(query_data)
            query_labels_list.append(query_labels)
            metadata_list.append(metadata)
        
        # 堆叠成batch
        batch_query_data = torch.stack(query_data_list)  # (batch_size, target_length)
        batch_query_labels = torch.stack(query_labels_list)  # (batch_size, num_classes)
        
        return batch_query_data, batch_query_labels, metadata_list
    
    def get_support_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取支持集数据
        
        Returns:
            support_data: (num_classes, shots_per_class, target_length)
            support_masks: (num_classes, shots_per_class, target_length)
        """
        if self.random_sampling:
            # 随机采样模式：每次调用都生成新的随机样本
            support_data, support_masks, _ = self.support_dataset.get_all_support_data()
            return support_data, support_masks
        else:
            # 固定采样模式：返回预加载的数据
            return self.support_data, self.support_masks
    
    def __iter__(self):
        """返回数据迭代器"""
        return MetaTrafficIterator(self)
    
    def __len__(self):
        """返回batch数量"""
        return len(self.query_loader)


class MetaTrafficIterator:
    """
    Meta Traffic数据迭代器
    输出格式完全兼容MultiMetaFingerNet.forward()
    """
    
    def __init__(self, dataloader: MetaTrafficDataLoader):
        self.dataloader = dataloader
        self.query_iter = iter(dataloader.query_loader)
        self.support_data, self.support_masks = dataloader.get_support_data()
        
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        返回下一个batch，格式兼容MultiMetaFingerNet
        
        Returns:
            query_data: (batch_size, target_length) 查询集数据
            support_data: (num_classes, shots_per_class, target_length) 支持集数据
            support_masks: (num_classes, shots_per_class, target_length) 支持集mask
            batch_info: Dict 包含查询标签和元数据
        """
        try:
            # 获取查询集batch
            query_data, query_labels, metadata = next(self.query_iter)
            #self.support_data, self.support_masks = self.dataloader.get_support_data()
            # 组织batch信息
            batch_info = {
                'query_labels': query_labels,  # (batch_size, num_classes)
                'metadata': metadata,
                'class_order': self.dataloader.class_order,
                'num_classes': len(self.dataloader.activated_classes)
            }
            
            return query_data, self.support_data, self.support_masks, batch_info
            
        except StopIteration:
            raise StopIteration


def test_meta_dataloader():
    """测试整合的数据加载器"""
    print("="*60)
    print("测试MetaTrafficDataLoader")
    print("="*60)
    
    # 设置路径
    query_json_path = "/home/ubuntu22/multi-tab-work/meta-finger/data/3tab_task/3tab_train.json"
    query_files_dir = "/home/ubuntu22/multi-tab-work/meta-finger/data/3tab_task/3tab_files"
    support_root_dir = "/home/ubuntu22/multi-tab-work/meta-finger/data/3tab_task/CW_single_tab/train"
    
    # 检查路径
    paths_to_check = [query_json_path, query_files_dir, support_root_dir]
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"路径存在: {path}")
        else:
            print(f"路径不存在: {path}")
    
    if not all(os.path.exists(p) for p in paths_to_check):
        print("\n数据路径不存在，跳过测试")
        return
    
    try:
        # 创建数据加载器
        print(f"\n创建MetaTrafficDataLoader...")
        dataloader = MetaTrafficDataLoader(
            query_json_path=query_json_path,
            query_files_dir=query_files_dir,
            support_root_dir=support_root_dir,
            activated_classes=list(range(60)),  # 0-59
            target_length=30000,
            shots_per_class=1,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        print(f"\n 测试数据格式兼容性...")
        # 测试几个batch
        for i, (query_data, support_data, support_masks, batch_info) in enumerate(dataloader):
            print(f"\nBatch {i+1}:")

            print(f"  查询集数据: {query_data.shape}")
            print(f"  查询集标签: {batch_info['query_labels'].shape}")
            print(f"  支持集数据: {support_data.shape}")
            print(f"  支持集掩码: {support_masks.shape}")
            print(f"  类别数量: {batch_info['num_classes']}")
            print(f"  标签和: {batch_info['query_labels']}")
            #print(f"  支持集数据: {support_data[0,0,:100]}")

            # 测试与MultiMetaFingerNet的兼容性
            print(f"\n🔧 MultiMetaFingerNet兼容性检查:")
            print(f"  query_data形状: {query_data.shape} ← 应为(batch_size, 30000)")
            print(f"  support_data形状: {support_data.shape} ← 应为(num_classes, shots, 30000)")
            print(f"  support_masks形状: {support_masks.shape} ← 应为(num_classes, shots, 30000)")
            
            # 只测试前2个batch
            if i >= 1:
                break
        
        print(f"\nMetaTrafficDataLoader测试完成！")
        print(f"数据格式完全兼容MultiMetaFingerNet.forward()接口")
        
    except Exception as e:
        print(f"\n 测试出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_meta_dataloader() 