import torch
from torch import nn
import torch.nn.functional as F


from torch.nn import Parameter

from modules.ops import _DOLinear, _DOConv1d, _DOConv2d, _DOConv3d, GroupPACTClampFn


class DOLinear(nn.Linear):
    def __init__(self, in_features, out_features, clamp_alpha=3.0, target_name=None, meta={}):
        super().__init__(in_features, out_features)

        self.clamp_alpha = clamp_alpha
        self.target_name = target_name
        self.meta = meta

        # self.ema_grad_meta = {
        #     'momentum' : 0.9,
        #     'lr' : torch.tensor([1e-2 * 5], device='cuda'),
        #     'ema_smooth' : torch.torch.zeros_like(self.weight, device='cuda')
        # }
        self.ema_grad_meta = None


    def forward(self, x):
        if self.training:
            # x = GroupPACTClampFn.apply(x, self.clamp_alpha, self.meta[f"{self.target_name}"]['group_size'], self.meta, self.target_name)
            return _DOLinear.apply(x, self.weight, self.clamp_alpha, self.target_name, self.meta, self.ema_grad_meta)
        else:
            # x = torch.clamp(x, -self.clamp_alpha, self.clamp_alpha)
            return super().forward(x)
        

    
    
class DOConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros',
                 clamp_alpha=3.0, target_name=None, meta={}):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=False, padding_mode=padding_mode)
    
        self.clamp_alpha = clamp_alpha
        self.target_name = target_name
        self.meta = meta

        # self.ema_grad_meta = {
        #     'momentum' : 0.9,
        #     'lr' : torch.tensor([1e-2 * 5], device='cuda'),
        #     'ema_smooth' : torch.torch.zeros_like(self.weight, device='cuda')
        # }
        self.ema_grad_meta = None

        # test = self.meta.get(f'{target_name}', False)
        # if not test:
        #     self.pact_alpha = nn.Parameter(torch.ones((10,), dtype=torch.bfloat16) * 3.0)
        # else:   
        #     self.pact_alpha = nn.Parameter(torch.ones((self.meta[f'{self.target_name}']['NG'],), dtype=torch.bfloat16) * 3.0)

        # self.pact_alpha = nn.Parameter(torch.ones((1,), dtype=torch.bfloat16) * acc_var)

        self.learnable_scale =  nn.Parameter(torch.ones((1,), dtype=torch.bfloat16) * 3.0)

    def forward(self, x):
        if self.training:
            if self.padding_mode != 'zeros':
                x = GroupPACTClampFn.apply(x, self.pact_alpha, self.meta[f"{self.target_name}"]['group_size'], self.meta, self.target_name)
                return _DOConv1d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                   self.weight, self.learnable_scale,
                                   self.stride, self.padding, self.dilation, self.groups,
                                   self.clamp_alpha, self.target_name, self.meta, self.ema_grad_meta)
            x = GroupPACTClampFn.apply(x, self.pact_alpha, self.meta[f"{self.target_name}"]['group_size'], self.meta, self.target_name)
            return _DOConv1d.apply(x, self.weight, self.learnable_scale,
                                     self.stride, self.padding, self.dilation, self.groups,
                                     self.clamp_alpha, self.target_name, self.meta, self.ema_grad_meta)
        else:
            x = torch.clamp(x, -self.clamp_alpha, self.clamp_alpha)
            return super().forward(x)
   

        
class DOConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channesl, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros',
                 clamp_alpha=3.0, target_name=None, meta={}):
        super().__init__(in_channels, out_channesl, kernel_size,
                         stride, padding, dilation, groups, bias=False, padding_mode=padding_mode)

        self.clamp_alpha = clamp_alpha
        self.target_name = target_name
        self.meta = meta


        # self.ema_grad_meta = {
        #     'momentum' : 0.9,
        #     'lr' : torch.tensor([1e-2 * 5], device='cuda'),
        #     'ema_smooth' : torch.torch.zeros_like(self.weight, device='cuda')
        # }
        self.ema_grad_meta = None
        self.learnable_scale = None

    def forward(self, x):
        if self.training:
            if self.padding_mode != 'zeros':
                # x = GroupPACTClampFn.apply(x, self.clamp_alpha, self.meta[f"{self.target_name}"]['group_size'], self.meta, self.target_name)
                return _DOConv2d.apply(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                        self.weight, self.learnable_scale,
                                        self.stride, self.padding, self.dilation, self.groups,
                                        self.clamp_alpha, self.target_name, self.meta, self.ema_grad_meta)
            # x = GroupPACTClampFn.apply(x, self.clamp_alpha, self.meta[f"{self.target_name}"]['group_size'], self.meta, self.target_name)
            return _DOConv2d.apply(x, self.weight, self.learnable_scale,
                                    self.stride, self.padding, self.dilation, self.groups,
                                    self.clamp_alpha, self.target_name, self.meta, self.ema_grad_meta)
        else:
            # x = torch.clamp(x, -self.clamp_alpha, self.clamp_alpha)
            return super().forward(x)
        

        
class DOConv3d(nn.Conv3d):
    def __init__(self, in_chanels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros',
                 clamp_alpha=3.0, target_name=None, meta={}):
        super().__init__(in_chanels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias=False, padding_mode=padding_mode,)

        self.clamp_alpha = clamp_alpha
        self.target_name = target_name
        self.meta = meta

        # self.ema_grad_meta = {
        #     'momentum' : 0.9,
        #     'lr' : torch.tensor([1e-2 * 5], device='cuda'),
        #     'ema_smooth' : torch.torch.zeros_like(self.weight, device='cuda')
        # }
        self.ema_grad_meta = None

        # test = self.meta.get(f'{target_name}', False)
        # if not test:
        #     self.pact_alpha = nn.Parameter(torch.ones((10,), dtype=torch.bfloat16) * 3.0)
        # else:   
            # self.pact_alpha = nn.Parameter(torch.ones((self.meta[f'{self.target_name}']['NG'],), dtype=torch.bfloat16) * 3.0)
        # self.pact_alpha = nn.Parameter(torch.ones((1,), dtype=torch.bfloat16) * acc_var)

        self.learnable_scale =  nn.Parameter(torch.ones((1,), dtype=torch.bfloat16) * 3.0)

    def forward(self, x):
        if self.training:
            if self.padding_mode != 'zeros':
                x = GroupPACTClampFn.apply(x, self.pact_alpha, self.meta[f"{self.target_name}"]['group_size'], self.meta, self.target_name)
                return _DOConv3d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                  self.weight, self.learnable_scale,
                                  self.stride, self.padding, self.dilation, self.groups,
                                  self.clamp_alpha, self.target_name, self.meta, self.ema_grad_meta)
            x = GroupPACTClampFn.apply(x, self.pact_alpha, self.meta[f"{self.target_name}"]['group_size'], self.meta, self.target_name)
            return _DOConv3d(x, self.weight, self.learnable_scale,
                              self.stride, self.padding, self.dilation, self.groups,
                              self.clamp_alpha, self.target_name, self.meta, self.ema_grad_meta)
        else:
            x = torch.clamp(x, -self.clamp_alpha, self.clamp_alpha)
            return super().forward(x)
        


 





class DODepthPointConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros',
                 acc_var=5.0,target_name=None, meta={}):
        super().__init__()

        self.depth_quanconv2d = DOConv2d(in_channels, in_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=in_channels, padding_mode=padding_mode,
                              acc_var=acc_var, target_name=target_name, dilation=dilation, meta=meta)

        self.point_quanconv2d = DOConv2d(in_channels, out_channels, kernel_size=1,
                              stride=1, padding=0, groups=1, padding_mode='zeros',
                              acc_var=acc_var, target_name=target_name, meta=meta)
        
    def forward(self, x):
        return self.point_quanconv2d(self.depth_quanconv2d(x))
    


class LearnableLp(nn.Module):
    def __init__(self, init_p=1.5):
        super().__init__()
        # p is learnable
        self.p = nn.Parameter(torch.tensor([init_p], dtype=torch.float32))

    def forward(self, x, kernel_size):
        # keep p in stable range
        p = torch.clamp(self.p, 0.5, 6.0)
        return F.lp_pool2d(x, p, kernel_size=kernel_size)




class RMSNorm(nn.Module):
    def __init__(self, dims, eps=1e-8):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_normed = x / rms
        return x_normed * self.weight






class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        delta = torch.max(x) - torch.min(x)
        x = (x/delta + 0.5)
        return x.round() * 2 - 1
    @staticmethod
    def backward(ctx, g):
        return g 











