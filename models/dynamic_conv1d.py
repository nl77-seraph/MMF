

import torch
import torch.nn as nn
import torch.nn.functional as F


def dynamic_conv1d(is_first, partial=None):
    """
    Create a 1D dynamic convolution class.
    Follows the design idea from dynamic_conv.py.
    
    Args:
        is_first: Whether this is the first dynamic convolution layer.
        partial: Optional partial prediction parameter.
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
            
            # Create shared weights if partial parameters are used.
            if self.partial is not None:
                assert self.partial <= self.out_channels
                self.weight = nn.Parameter(torch.Tensor(self.partial, kernel_size))
                self._init_weights()
            else:
                self.register_parameter('weight', None)
            
            # Do not use bias, consistent with the original implementation.
            self.register_parameter('bias', None)
        
        def _init_weights(self):
            """Initialize weights."""
            if self.partial is not None:
                n = self.partial * self.kernel_size
                stdv = 1. / (n ** 0.5)
                self.weight.data.uniform_(-stdv, stdv)
        
        def forward(self, inputs):
            """
            1D dynamic convolution forward pass.
            
            Args:
                inputs: tuple (input_features, dynamic_weight)
                    input_features: (batch, channels, length)
                    dynamic_weight: (num_classes, channels, kernel_size)
            
            Returns:
                Convolution result: (batch*num_classes, out_channels, output_length)
            """
            assert self.is_first is not None, 'Please set the state of DynamicConv1d first.'
            
            input_features, dynamic_weight = inputs
            
            # Validate the kernel size of dynamic weights.
            assert dynamic_weight.size(-1) == self.kernel_size, \
                f"Dynamic weight kernel size {dynamic_weight.size(-1)} != {self.kernel_size}"
            
            # Validate channel matching.
            assert dynamic_weight.size(1) % input_features.size(1) == 0, \
                "Dynamic weight channels must be divisible by input channels"
            
            n_cls = dynamic_weight.size(0)  # Number of classes.
            
            # Handle partial weights if present.
            if self.partial is not None:
                # Repeat shared weights for all classes.
                shared_weight = self.weight.repeat(n_cls, 1, 1)
                # Concatenate shared weights and dynamic weights.
                dynamic_weight = torch.cat([shared_weight, dynamic_weight], dim=1)
            
            if self.is_first:
                # First layer: repeat inputs for all classes.
                batch_size = input_features.size(0)
                n_channels = input_features.size(1)
                # input tensor (N, C, L) -> (N, C*n_cls, L)
                input_features = input_features.repeat(1, n_cls, 1)
            else:
                # Later layers: inputs already include the class dimension.
                assert input_features.size(0) % n_cls == 0, \
                    "Input batch size does not match with n_cls"
                batch_size = input_features.size(0) // n_cls
                n_channels = input_features.size(1)
                input_length = input_features.size(-1)
                input_features = input_features.view(batch_size, n_cls * n_channels, input_length)
            
            # Compute the groups parameter.
            group_size = dynamic_weight.size(1) // n_channels
            groups = n_cls * n_channels // group_size
            
            # Reshape dynamic weights from (n_cls, channels, kernel_size) to (n_cls*channels, 1, kernel_size).
            dynamic_weight = dynamic_weight.view(-1, group_size, dynamic_weight.size(-1))
            
            # Run 1D grouped convolution.
            conv_result = F.conv1d(input_features, dynamic_weight, self.bias, 
                                 self.stride, self.padding, self.dilation, groups)
            
            # Reshape output.
            feat_length = conv_result.size(-1)
            conv_result = conv_result.view(-1, n_channels, feat_length)
            
            return conv_result
    
    # Set class attributes.
    DynamicConv1d.is_first = is_first
    DynamicConv1d.partial = partial
    return DynamicConv1d


class FeatureReweightingModule(nn.Module):
    """
    Feature reweighting module.
    Fuses query features with support-set weights.
    """
    
    def __init__(self, feature_dim=256, kernel_size=1):
        super(FeatureReweightingModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        
        # Create a 1D dynamic convolution layer.
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
        Feature reweighting forward pass.
        
        Args:
            query_features: (batch, seq_len, feature_dim) Query set features.
            support_weights: (num_classes, feature_dim) Support set weights.
            
        Returns:
            reweighted_features: (batch*num_classes, feature_dim, seq_len) Reweighted features.
        """
        batch_size, seq_len, feature_dim = query_features.shape
        num_classes = support_weights.size(0)
        
        # Transpose query features to match conv1d format: (batch, feature_dim, seq_len).
        query_features = query_features.transpose(1, 2)
        
        # Expand support weights: (num_classes, feature_dim) -> (num_classes, feature_dim, kernel_size).
        support_weights = support_weights.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        
        # Run dynamic convolution.
        reweighted_features = self.dynamic_conv((query_features, support_weights))
        
        return reweighted_features
    
    def get_output_shape(self, batch_size, seq_len, num_classes):
        """
        Get output shape information.
        
        Returns:
            output_shape: (batch*num_classes, feature_dim, seq_len)
        """
        return (batch_size * num_classes, self.feature_dim, seq_len)

