import torch
from torch import nn

from .act_ops import (_DOReLU, _DOSiLU, _DOGELU)


class DOReLU_Variance(nn.ReLU):
    def __init__(self, inplace=False, relu=True, relu6=False, target_name=None, meta={}):
        super().__init__(inplace=inplace)

        self.target_name = target_name
        self.meta = meta
        self.relu = relu
        self.relu6 = relu6


    def forward(self, x):
        if self.training:
            return _DOReLU.apply(x, self.relu, self.relu6, self.target_name, self.meta)
        else:
            return super().forward(x)
        

class DOSiLU(nn.SiLU):
    def __init__(self, inplace=False, target_name=None, meta={}):
        super().__init__(inplace=inplace)

        self.target_name = target_name
        self.meta = meta

    def forward(self, x):
        if self.training:
            return _DOSiLU.apply(x, self.target_name, self.meta)
        else:
            return super().forward(x)
        


class DOGELU(nn.SiLU):
    def __init__(self, inplace=False, target_name=None, meta={}):
        super().__init__(inplace=inplace)

        self.target_name = target_name
        self.meta = meta

    def forward(self, x):
        if self.training:
            return _DOGELU.apply(x, self.target_name, self.meta)
        else:
            return super().forward(x)