"""
1D动态卷积实现
参考Few-shot Detection项目的dynamic_conv.py，适配1D时序数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dynamic_conv1d(is_first, partial=None):
    """
    创建1D动态卷积类
    参考dynamic_conv.py的设计思想
    
    Args:
        is_first: 是否是第一个动态卷积层
        partial: 部分预测参数（可选）
    """
    
    class DynamicConv1d(nn.Module):
        is_first = None
        partial = None
        
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=False):
            super(DynamicConv1d, self).__init__()
            
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            
            # 如果有partial参数，创建共享权重
            if self.partial is not None:
                assert self.partial <= self.out_channels
                self.weight = nn.Parameter(torch.Tensor(self.partial, kernel_size))
                self._init_weights()
            else:
                self.register_parameter('weight', None)
            
            # 不使用bias（与原始实现一致）
            self.register_parameter('bias', None)
        
        def _init_weights(self):
            """初始化权重"""
            if self.partial is not None:
                n = self.partial * self.kernel_size
                stdv = 1. / (n ** 0.5)
                self.weight.data.uniform_(-stdv, stdv)
        
        def forward(self, inputs):
            """
            1D动态卷积前向传播
            
            Args:
                inputs: tuple (input_features, dynamic_weight)
                    input_features: (batch, channels, length)
                    dynamic_weight: (num_classes, channels, kernel_size)
            
            Returns:
                卷积结果: (batch*num_classes, out_channels, output_length)
            """
            assert self.is_first is not None, 'Please set the state of DynamicConv1d first.'
            
            input_features, dynamic_weight = inputs
            
            # 验证动态权重的kernel_size
            assert dynamic_weight.size(-1) == self.kernel_size, \
                f"Dynamic weight kernel size {dynamic_weight.size(-1)} != {self.kernel_size}"
            
            # 验证通道匹配
            assert dynamic_weight.size(1) % input_features.size(1) == 0, \
                "Dynamic weight channels must be divisible by input channels"
            
            n_cls = dynamic_weight.size(0)  # 类别数
            
            # 处理partial权重（如果有）
            if self.partial is not None:
                # 重复共享权重到所有类别
                shared_weight = self.weight.repeat(n_cls, 1, 1)
                # 拼接共享权重和动态权重
                dynamic_weight = torch.cat([shared_weight, dynamic_weight], dim=1)
            
            if self.is_first:
                # 第一层：输入需要重复到所有类别
                batch_size = input_features.size(0)
                n_channels = input_features.size(1)
                # input tensor (N, C, L) -> (N, C*n_cls, L)
                input_features = input_features.repeat(1, n_cls, 1)
            else:
                # 后续层：输入已经包含类别维度
                assert input_features.size(0) % n_cls == 0, \
                    "Input batch size does not match with n_cls"
                batch_size = input_features.size(0) // n_cls
                n_channels = input_features.size(1)
                input_length = input_features.size(-1)
                input_features = input_features.view(batch_size, n_cls * n_channels, input_length)
            
            # 计算groups参数
            group_size = dynamic_weight.size(1) // n_channels
            groups = n_cls * n_channels // group_size
            
            # 重整形动态权重: (n_cls, channels, kernel_size) -> (n_cls*channels, 1, kernel_size)
            dynamic_weight = dynamic_weight.view(-1, group_size, dynamic_weight.size(-1))
            
            # 执行1D分组卷积
            conv_result = F.conv1d(input_features, dynamic_weight, self.bias, 
                                 self.stride, self.padding, self.dilation, groups)
            
            # 重整形输出
            feat_length = conv_result.size(-1)
            conv_result = conv_result.view(-1, n_channels, feat_length)
            
            return conv_result
    
    # 设置类属性
    DynamicConv1d.is_first = is_first
    DynamicConv1d.partial = partial
    return DynamicConv1d


class FeatureReweightingModule(nn.Module):
    """
    特征重加权模块
    将查询集特征和支持集权重进行融合
    """
    
    def __init__(self, feature_dim=256, kernel_size=1):
        super(FeatureReweightingModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        
        # 创建1D动态卷积层
        DynamicConv1d = dynamic_conv1d(is_first=True, partial=None)
        self.dynamic_conv = DynamicConv1d(
            in_channels=feature_dim,
            out_channels=feature_dim, 
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )
    
    def forward(self, query_features, support_weights):
        """
        特征重加权前向传播
        
        Args:
            query_features: (batch, seq_len, feature_dim) 查询集特征
            support_weights: (num_classes, feature_dim) 支持集权重
            
        Returns:
            reweighted_features: (batch*num_classes, feature_dim, seq_len) 重加权特征
        """
        batch_size, seq_len, feature_dim = query_features.shape
        num_classes = support_weights.size(0)
        
        # 转置查询特征以符合conv1d格式: (batch, feature_dim, seq_len)
        query_features = query_features.transpose(1, 2)
        
        # 扩展支持权重维度: (num_classes, feature_dim) -> (num_classes, feature_dim, kernel_size)
        support_weights = support_weights.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        
        # 执行动态卷积
        reweighted_features = self.dynamic_conv((query_features, support_weights))
        
        return reweighted_features
    
    def get_output_shape(self, batch_size, seq_len, num_classes):
        """
        获取输出形状信息
        
        Returns:
            output_shape: (batch*num_classes, feature_dim, seq_len)
        """
        return (batch_size * num_classes, self.feature_dim, seq_len)

