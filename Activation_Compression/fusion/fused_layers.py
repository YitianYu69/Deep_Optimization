import torch 
from torch import nn
import torch.nn.functional as F

from .fused_ops import _DOBatchNormReLU2d


class DOBatchNormReLU2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, relu=True, relu6=False, affine=True, track_running_stats=True, group=0,
                target_name=None, meta={}):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

        self.target_name = target_name
        self.meta = meta
        self.relu = relu
        self.relu6 = relu6

        self.ema_grad_meta = None
    
    def forward(self, x):
        if not self.training:
            return F.relu(super().forward(x))
        else:
            self._check_input_dim(x)

            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum

            if self.training and self.track_running_stats:
                if self.num_batches_tracked is None:
                    self.num_batches_tracked.add_(1)
                    if self.momentum is None:
                        exponential_average_factor = 1.0 / self.num_batches_tracked.float()
                    else:
                        exponential_average_factor = self.momentum

        return _DOBatchNormReLU2d.apply(
            x, self.weight, self.bias, self.relu, self.relu6, self.running_mean, self.running_var, exponential_average_factor,
            self.target_name, self.meta, self.ema_grad_meta
        )
