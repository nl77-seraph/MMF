"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from typing import Optional, List

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
# if float(torchvision.__version__[:3]) < 0.5:
#     import math
#     from torchvision.ops.misc import _NewEmptyTensorOp
#     def _check_size_scale_factor(dim, size, scale_factor):
#         # type: (int, Optional[List[int]], Optional[float]) -> None
#         if size is None and scale_factor is None:
#             raise ValueError("either size or scale_factor should be defined")
#         if size is not None and scale_factor is not None:
#             raise ValueError("only one of size or scale_factor should be defined")
#         if not (scale_factor is not None and len(scale_factor) != dim):
#             raise ValueError(
#                 "scale_factor shape must match input shape. "
#                 "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
#             )
#     def _output_size(dim, input, size, scale_factor):
#         # type: (int, Tensor, Optional[List[int]], Optional[float]) -> List[int]
#         assert dim == 2
#         _check_size_scale_factor(dim, size, scale_factor)
#         if size is not None:
#             return size
#         # if dim is not 2 or scale_factor is iterable use _ntuple instead of concat
#         assert scale_factor is not None and isinstance(scale_factor, (int, float))
#         scale_factors = [scale_factor, scale_factor]
#         # math.floor might return float in py2.7
#         return [
#             int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
#         ]
# elif float(torchvision.__version__[:3]) < 0.7:
#     from torchvision.ops import _new_empty_tensor
#     from torchvision.ops.misc import _output_size




class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    batch = list(zip(*batch))
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor(tensor, num_tabs=None, is_support=False):
    """
    将单个张量转换为嵌套张量，并根据是否为支持样本和标签数量进行填充或截断。
    
    Args:
        tensor: 输入张量
        num_tabs: 标签数量，用于确定查询样本目标长度
        is_support: 是否为支持样本
        
    Returns:
        NestedTensor: 包含处理后的张量和掩码的对象
    """
    # 如果已经是NestedTensor，直接返回
    if isinstance(tensor, NestedTensor):
        return tensor
    
    # 设置目标长度
    if is_support:
        # 支持样本固定长度为10000
        target_length = 10000
    elif num_tabs is not None:
        # 查询样本长度为 num_tabs * 10000
        target_length = num_tabs * 10000 
    else:
        # 不指定长度则不进行处理
        target_length = None
    
    # 确保2D格式 [C,L]
    if tensor.dim() == 1:  # [L]
        tensor = tensor.unsqueeze(0)  # 变为 [1,L]
        
    # 获取原始长度和通道数
    c, l = tensor.shape
    
    # 创建掩码，默认全为True（所有位置无效）
    mask = torch.ones(target_length, dtype=torch.bool, device=tensor.device)
    
    # 标记原始数据部分为False（有效数据）
    valid_length = min(l, target_length)
    mask[:valid_length] = False
    
    # 调整长度（如果需要）
    if target_length is not None and l != target_length:
        # 创建目标长度的新张量
        new_tensor = torch.zeros((c, target_length), 
                               dtype=tensor.dtype, 
                               device=tensor.device)
        # 填充或截断
        if l > target_length:
            # 截断
            new_tensor[:, :target_length] = tensor[:, :target_length]
        else:
            # 填充
            new_tensor[:, :l] = tensor[:, :l]
        
        tensor = new_tensor
    
    # 返回NestedTensor对象
    return NestedTensor(tensor, mask)

def nested_list_from_list_nested(list_nested):
    # 提取所有 tensors 和 masks
    tensors = [nt.tensors for nt in list_nested]
    masks = [nt.mask for nt in list_nested]

    # 检查所有 tensor 的通道数是否一致
    channels = [t.size(0) for t in tensors]
    if len(set(channels)) != 1:
        raise ValueError("所有张量的通道数必须相同。")
    C = channels[0]

    # 处理 tensor 的填充和堆叠
    # 将每个 tensor 转换为 (L, C) 以便使用 pad_sequence
    tensors_to_pad = [t.permute(1, 0) for t in tensors]  # (L, C)
    padded_tensors = torch.nn.utils.rnn.pad_sequence(
        tensors_to_pad, 
        batch_first=True, 
        padding_value=0
    )  # (B, L_max, C)
    batched_tensors = padded_tensors.permute(0, 2, 1)  # (B, C, L_max)

    # 处理 mask 的填充和堆叠
    masks_to_pad = []
    for i, mask in enumerate(masks):
        L_i = tensors[i].size(1)  # 原始长度 (C, L_i)
        if mask is None:
            # 生成全 True 的 mask
            mask = torch.ones(L_i, dtype=torch.bool, device=tensors[i].device)
        else:
            # 确保 mask 是布尔类型且长度匹配
            mask = mask.bool()
            assert mask.size(0) == L_i, f"第 {i} 个 mask 的长度与 tensor 不匹配"
        masks_to_pad.append(mask)
    
    # 填充 mask 到最大长度
    padded_masks = torch.nn.utils.rnn.pad_sequence(
        masks_to_pad, 
        batch_first=True, 
        padding_value=False
    )  # (B, L_max)

    return NestedTensor(batched_tensors, padded_masks)

        

def nested_tensor_from_tensor_list(tensor_list, num_tabs=None, is_support=False):
    """
    将张量列表转换为嵌套张量，并根据是否为支持样本和标签数量进行填充或截断。
    
    Args:
        tensor_list: 张量列表
        num_tabs: 标签数量，用于确定查询样本目标长度
        is_support: 是否为支持样本
    
    Returns:
        NestedTensor: 包含批处理张量和掩码的对象
    """
    # 如果已经是NestedTensor，直接返回
    if isinstance(tensor_list, NestedTensor):
        return tensor_list
    
    # 设置目标长度
    if is_support:
        # 支持样本固定长度为10000
        target_length = 10000
    elif num_tabs is not None:
        # 查询样本长度为 num_tabs * 10000
        target_length = num_tabs * 10000
    else:
        # 不指定长度则不进行处理
        target_length = None
    
    # 确保tensor_list是列表
    if isinstance(tensor_list, torch.Tensor):
        tensor_list = [tensor_list]
    
    # 标准化张量为 [C,L] 格式并处理长度
    normalized_tensors = []
    for tensor in tensor_list:
        # 确保2D格式 [C,L]
        if tensor.dim() == 1:  # [L]
            tensor = tensor.unsqueeze(0)  # 变为 [1,L]
        elif tensor.dim() == 2 and tensor.shape[0] > tensor.shape[1]:
            # 假设第一维是长度，需要转置
            tensor = tensor.transpose(0, 1)  # [L,C] -> [C,L]
        
        # 调整长度（如果需要）
        if target_length is not None:
            c, l = tensor.shape
            if l != target_length:
                # 创建目标长度的新张量
                new_tensor = torch.zeros((c, target_length), 
                                       dtype=tensor.dtype, 
                                       device=tensor.device)
                # 填充或截断
                if l > target_length:
                    # 截断
                    new_tensor[:, :target_length] = tensor[:, :target_length]
                else:
                    # 填充
                    new_tensor[:, :l] = tensor[:, :l]
                
                tensor = new_tensor
        
        normalized_tensors.append(tensor)
    
    # 创建批次
    if len(normalized_tensors) == 0:
        return NestedTensor(torch.zeros(0), torch.zeros(0))
    
    # 获取所有张量的最大尺寸
    max_size = _max_by_axis([list(tensor.shape) for tensor in normalized_tensors])
    batch_shape = [len(normalized_tensors)] + max_size
    b, c, l = batch_shape  # batch, channel, length
    
    # 创建批次张量和掩码
    dtype = normalized_tensors[0].dtype
    device = normalized_tensors[0].device
    batch_tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, l), dtype=torch.bool, device=device)  # 掩码初始化为True（填充位置）
    
    # 填充每个序列并更新掩码
    for i, tensor in enumerate(normalized_tensors):
        # 复制实际数据
        c_len, l_len = tensor.shape
        batch_tensor[i, :c_len, :l_len] = tensor
        # 更新掩码（实际数据位置为False）
        mask[i, :l_len] = False
    
    return NestedTensor(batch_tensor, mask)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29600')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate_img(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        if float(torchvision.__version__[:3]) < 0.5:
            return _NewEmptyTensorOp.apply(input, output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def interpolate_sequence(input, target_len=None, scale_factor=None, mode="linear"):
    """
    序列数据的插值函数
    
    参数:
        input: 输入序列 [batch_size, seq_len, feature_dim]
        target_len: 目标序列长度
        scale_factor: 缩放因子
        mode: 插值模式，可以是 'linear' 或 'nearest'
    """
    if input.numel() == 0:
        return input
        
    if target_len is None and scale_factor is None:
        raise ValueError("target_len 和 scale_factor 必须指定一个")
        
    if target_len is None:
        target_len = int(input.shape[1] * scale_factor)
        
    # 对序列长度维度进行插值
    return F.interpolate(
        input.transpose(1, 2),  # [batch_size, feature_dim, seq_len]
        size=target_len,
        mode=mode
    ).transpose(1, 2)  # [batch_size, seq_len, feature_dim]


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def inverse_sigmoid(x, eps=1e-6):  # TODO: original 1e-5
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)



def read_pkl_file(file_path):#Note
    return None