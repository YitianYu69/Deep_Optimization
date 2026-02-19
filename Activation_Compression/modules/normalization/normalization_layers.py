import torch 
from torch import nn
import torch.distributed as dist

from .normalization_ops import _DOBatchNorm, _DOSyncBatchNorm

       
class DOBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, group=0,
                 target_name=None, meta={}):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        self.target_name = target_name
        self.meta = meta

        # self.ema_grad_meta = {
        #     'momentum' : 0.9,
        #     'lr' : torch.tensor([1e-2 * 5], device='cuda'),
        #     'ema_smooth' : torch.torch.zeros_like(self.weight, device='cuda')
        # }
        self.ema_grad_meta = None

    def forward(self, x):
        if not self.training:
            return super().forward(x)

        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / self.num_batches_tracked.float()
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        return _DOBatchNorm.apply(
            x, self.weight, self.bias, 
            self.running_mean,
            self.running_var,
            exponential_average_factor, self.eps,
            self.target_name, self.meta, self.ema_grad_meta)



class DOSyncBatchNorm2d(nn.SyncBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, group=0,
                 target_name=None, meta={}):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, process_group)

        self.target_name = target_name
        self.meta = meta

        # self.ema_grad_meta = {
        #     'momentum' : 0.9,
        #     'lr' : torch.tensor([1e-2 * 5], device='cuda'),
        #     'ema_smooth' : torch.torch.zeros_like(self.weight, device='cuda')
        # }
        self.ema_grad_meta = None

    def forward(self, x):
        if not x.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')
        
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        need_sync = self.training
        if need_sync:
            process_group = dist.group.WORLD 
            if self.process_group:
                process_group = self.process_group
            world_size = dist.get_world_size(process_group)
            need_sync = world_size > 1

        if not need_sync:
            return _DOBatchNorm.apply(
                x, self.weight, self.bias, 
                self.running_mean,
                self.running_var,
                exponential_average_factor, self.eps,
                self.quantizer, self.target_name, self.graph_mode, self.meta, self.ema_grad_meta)
        else:
            if not self.ddp_gpu_size:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

            assert self.training
            return _DOSyncBatchNorm.apply(
                input, self.weight, self.bias, self.running_mean, self.running_var,
                self.eps, exponential_average_factor, process_group, world_size, 
                self.target_name, self.meta, self.ema_grad_meta)
