"""
网站指纹识别数据集定义和加载工具。

特别说明：
1. 支持样本(support samples)长度固定为10000
2. 查询样本(query samples)长度为 num_tabs * 10000
3. 使用nested_tensor_from_tensor处理样本，记录填充部分的掩码
4. 数据集加载时即进行长度处理，以便批处理更高效
5. 新版本：按需加载数据以减少内存占用
"""

import os
import random
import torch
import torch.utils.data
import numpy as np
import pickle
import json

from util.misc import read_pkl_file, NestedTensor
from util.misc import nested_tensor_from_tensor


class TrafficDataset():
    def __init__(self, args, root_dir, ann_file, transforms, support_transforms, return_masks, activated_class_ids, with_support):
        self.with_support = with_support
        self.activated_class_ids = activated_class_ids
        self.transforms = transforms
        self.support_transforms = support_transforms
        self.prepare = None
        self.num_tabs = args.num_tabs

        # 新的数据加载方式：从JSON文件读取查询样本文件名列表
        # root_dir 现在应该是JSON文件的路径，例如 "3tab_train.json"
        self.json_file_path = root_dir
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"JSON文件不存在：{self.json_file_path}")
            
        # 读取JSON文件获取查询样本文件名列表
        print(f"加载JSON索引文件: {self.json_file_path}")
        with open(self.json_file_path, 'r') as f:
            self.query_file_names = json.load(f)
        
        # 设置查询样本文件所在目录
        # 默认为 /home/ubuntu22/multi-tab-work/meta-finger/data/3tab_task/3tab_files/
        self.query_files_dir = "/home/ubuntu22/multi-tab-work/meta-finger/data/3tab_task/3tab_files"
        if not os.path.exists(self.query_files_dir):
            raise FileNotFoundError(f"查询样本文件目录不存在：{self.query_files_dir}")
        
        self.data_length = len(self.query_file_names)
        print(f"JSON索引加载完成，查询样本数量: {self.data_length}")
        
        if self.with_support:
            # 支持样本相关参数
            self.NUM_SUPP = args.total_num_support
            self.NUM_MAX_POS_SUPP = args.max_pos_support
            
            # 构建支持样本索引（不立即加载数据）
            self.build_support_dataset()

    def parse_labels_from_filename(self, filename):
        """
        从文件名中解析标签
        文件名格式: "类别1_类别2_类别3_随机文件名.pkl"
        """
        # 移除.pkl扩展名
        basename = os.path.splitext(filename)[0]
        # 按下划线分割
        parts = basename.split('_')
        
        # 提取标签部分（除了最后一个随机文件名部分）
        # 假设随机文件名部分不包含纯数字，而类别ID是纯数字
        labels = []
        for part in parts:
            try:
                label = int(part)
                if label in self.activated_class_ids:
                    labels.append(label)
            except ValueError:
                # 如果无法转换为整数，则认为是随机文件名部分，停止解析
                break
        
        return labels

    def __getitem__(self, idx):
        # 读取数据 - 按需从磁盘加载
        query_file_name = self.query_file_names[idx]
        query_file_path = os.path.join(self.query_files_dir, query_file_name)
        if not os.path.exists(query_file_path):
            raise FileNotFoundError(f"查询样本文件不存在：{query_file_path}")
            
        # 从pkl文件加载数据
        sample_data = self.safe_load_pickle(query_file_path)

        
        trace_data = sample_data
        
        # 从文件名解析标签
        labels = self.parse_labels_from_filename(query_file_name)
        if not labels:
            raise ValueError(f"无法从文件名 {query_file_name} 中解析出有效标签")
            
        # 构建输入数据 [channel, seq_len] 格式
        query_data = torch.tensor(trace_data, dtype=torch.float32).unsqueeze(0)  # [1, seq_len]
        
        # 创建掩码，用于标记有效数据区域
        mask = torch.zeros(query_data.shape[1], dtype=torch.bool, device=query_data.device)
        query_nested = nested_tensor_from_tensor(query_data, num_tabs=self.num_tabs, is_support=False)

        # 构建target字典
        target = {
            'image_id': torch.tensor([idx]),
            'labels': torch.tensor(labels, dtype=torch.long),  # 多标签
            'annotations': None,
            'boxes': torch.zeros((0, 4)),
            'area': torch.tensor([0.0]),
            'iscrowd': torch.tensor([0])
        }
        
        # 应用数据增强（空接口）
        if self.transforms is not None:
            query_nested, target = self.transforms(query_nested, target)
        
        if self.with_support:
            # 根据query样本的标签采样support样本
            support_samples, support_class_ids, support_targets = self.sample_support_samples(target)
            return query_nested, target, support_samples, support_class_ids, support_targets
        else:
            return query_nested, target

    def __len__(self):
        return self.data_length

    def build_support_dataset(self):
        """
        构建支持样本索引 - 只建立类别到文件名列表的映射，不立即加载数据
        """
        self.supp_samples_by_class_files = {}  # 改名以更清楚地表示只存储文件路径
        
        # 获取single_tab数据的根目录
        support_root = '/home/ubuntu22/multi-tab-work/meta-finger/data/3tab_task/CW_single_tab/train'
        print(f"建立support数据索引: {support_root}")
        
        if not os.path.exists(support_root):
            print(f"警告: support数据目录不存在: {support_root}")
            return
        
        # 遍历support_root下的所有条目 (这些条目应该是代表类别的文件夹)
        for class_name_str in os.listdir(support_root):
            class_dir = os.path.join(support_root, class_name_str)
            
            if not os.path.isdir(class_dir):
                # 如果不是目录，则跳过 (例如，support_root下可能有非类别文件夹的其他文件)
                print(f"提示: {class_dir} 不是一个目录，已跳过。")
                continue
            
            # 将文件夹名 (字符串) 转换为整数类别ID
            numeric_class_id = int(class_name_str)

            # 收集当前类别的所有有效样本文件路径
            class_files = [
                os.path.join(class_dir, sample_name) 
                for sample_name in os.listdir(class_dir)
                if sample_name.endswith('.pkl')
            ]
            
            # 确保每个类别至少有1个样本
            if len(class_files) == 0:
                print(f"警告: 类别目录 '{class_name_str}' (ID={numeric_class_id}) 中没有有效样本。路径: {class_dir}")
                continue
            
            self.supp_samples_by_class_files[numeric_class_id] = class_files
            print(f"类别 '{class_name_str}' (ID={numeric_class_id}) 索引了 {len(class_files)} 个support样本文件")
        
        if not self.supp_samples_by_class_files:
            print("错误: 没有找到任何可用的support样本")
            raise ValueError("没有找到任何可用的support样本，请确保single_tab目录中包含有效的样本数据")

    def safe_load_pickle(self, file_path):
        """安全加载pickle文件"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"无法加载文件 {file_path}: {str(e)}")
        return data
    def sample_support_samples(self, target):
        """
        采样支持样本，处理多标签情况
        参考Meta-DETR中的实现，确保正负样本采样合理
        """
        if not self.supp_samples_by_class_files:
            raise ValueError("没有可用的support样本索引，请确保single_tab目录中包含有效的样本数据")

        # 获取当前query的所有唯一标签
        positive_labels = target['labels'].unique()
        num_positive_labels = positive_labels.shape[0]
        positive_labels_list = positive_labels.tolist()

        # 获取负类别列表（所有可用类别中不在positive_labels中的类别）
        available_classes = list(self.supp_samples_by_class_files.keys())
        negative_labels_list = list(set(available_classes) - set(positive_labels_list))

        # 严格处理空负样本情况（需根据实际情况调整）
        if not negative_labels_list and self.NUM_SUPP > num_positive_labels:
            raise ValueError("负样本类别为空且需要补充负样本，无法完成采样")

        # 根据正样本数量决定采样策略
        if num_positive_labels <= self.NUM_MAX_POS_SUPP:
            sampled_labels_list = positive_labels_list.copy()
            num_neg_needed = self.NUM_SUPP - num_positive_labels
            if num_neg_needed > 0:
                # 直接采样不重复类别
                sampled_neg = random.sample(negative_labels_list, k=num_neg_needed)
                sampled_labels_list += sampled_neg
        else:
            # 正样本数量过多时采样
            sampled_positive_labels = random.sample(
                positive_labels_list, 
                k=self.NUM_MAX_POS_SUPP
            )
            num_neg_needed = self.NUM_SUPP - self.NUM_MAX_POS_SUPP
            sampled_negative_labels = random.sample(
                negative_labels_list,
                k=num_neg_needed
            )
            sampled_labels_list = sampled_positive_labels + sampled_negative_labels

        # 加载支持样本
        support_samples = []
        support_targets = []
        support_class_ids = []
        for class_id in sampled_labels_list:
            # 从该类别中随机选择一个样本
            class_files = self.supp_samples_by_class_files[class_id]
            if not class_files:
                raise ValueError(f"类别 {class_id} 中没有有效的support样本文件")

            sample_file = random.choice(class_files)
            sample_data = self.safe_load_pickle(sample_file)

            # 确保数据有效性
            if sample_data is None or 'data' not in sample_data:
                raise ValueError(f"文件 {sample_file} 中没有有效数据")

            # 构建输入数据
            trace_data = sample_data['data']  # 从pkl文件读取数据
            support_tensor = torch.tensor(trace_data, dtype=torch.float32).unsqueeze(0)
            
            # 创建掩码
            mask = torch.zeros(support_tensor.shape[1], dtype=torch.bool, device=support_tensor.device)
            support_nested = nested_tensor_from_tensor(support_tensor, num_tabs=1, is_support=True)
            
            # 应用数据增强
            if self.support_transforms is not None:
                support_nested = self.support_transforms(support_nested)
            
            # 添加样本
            support_samples.append(support_nested)
            support_class_ids.append(class_id)
            
            # 构建目标字典
            unique_id = torch.tensor([hash(sample_file) % (2**63 - 1)])
            support_target = {
                'image_id': unique_id,
                'labels': torch.tensor([class_id]),
                'boxes': torch.zeros((0, 4)),
                'area': torch.tensor([0.0]),
                'iscrowd': torch.tensor([0])
            }
            support_targets.append(support_target)


        return support_samples, torch.tensor(support_class_ids), support_targets

class EmptyTransforms:
    """空的数据增强接口"""
    def __call__(self, data, target):
        return data, target


def make_transforms(phase):
    """创建数据增强函数"""
    return EmptyTransforms()


def make_support_transforms():
    """创建支持样本数据增强函数"""
    return EmptyTransforms()


def build(args, root_dir, ann_file, phase, activated_class_ids, with_support):
    """构建数据集"""
    return TrafficDataset(
        args, 
        root_dir, 
        ann_file,
        transforms=None, 
        support_transforms=None,
        return_masks=False,
        activated_class_ids=activated_class_ids,
        with_support=with_support
    )