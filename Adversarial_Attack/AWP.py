import torch
from torch import nn

from ..Activation_Compression.modules import layers

from typing import Callable
import copy


def normalize(perturbated_weights, ori_weights):
    perturbated_weights.mul_(ori_weights.norm() / (perturbated_weights.norm() + 1e-8))

def normalize_grad_by_weights(perturbated_weights, ori_weights):
    for name, w in perturbated_weights:
        if name in ori_weights and w.ndim > 1:
            normalize(w.grad.data, ori_weights[name])
        else:
            w.grad.data.fill_(0)



class AWP():
    def __init__(self,
                 proxy_cls: nn.Module,
                 proxy_kwarys: dict,
                 cri: Callable,
                 proxy_opt: torch.optim.Optimizer,
                 opt_kwargs: dict,
                 device: str = 'cuda'):

        self.proxy_cls = proxy_cls # Object Class
        self.proxy_kwarys = proxy_kwarys

        self.cri = cri
        self.proxy_opt_cls = proxy_opt
        self.opt_kwargs = opt_kwargs
        self.diff = None

        self.device = device


    def compute_diff(self, adv_data, target, model, epoch, num_iters=1):

        proxy = self.proxy_cls(**self.proxy_kwarys).to(self.device)
        # if epoch <= 2:
        #     apply_parametrizations(proxy)
        
        if isinstance(model, nn.parallel.DistributedDataParallel):
            proxy.load_state_dict(model.module.state_dict())
        else:
            proxy.load_state_dict(model.state_dict())

        proxy.train()

        proxy_opt = self.proxy_opt_cls(proxy.parameters(), **self.opt_kwargs)

        ori_params = {
            name: p.detach().clone()
            for name, p in proxy.named_parameters()
        }

        for _ in range(num_iters):

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                proxy_opt.zero_grad(set_to_none=True)

                logits = proxy(adv_data)
                loss = - self.cri(logits, target)

                loss.backward()

                normalize_grad_by_weights(proxy.named_parameters(), ori_params)

                proxy_opt.step()
                proxy_opt.zero_grad(set_to_none=True)
            
        self.diff = {}
        for name, p in proxy.named_parameters():
            if name in ori_params:
                d = p.detach() - ori_params[name]
                self.diff[name] = d

        del proxy

    @torch.no_grad()
    def perturbate(self, model, gamma=0.01):
        if not self.diff:
            raise ValueError('Please run perbute() before restore!')

        for name, p in model.named_parameters():
            if name in self.diff and p.ndim > 1:
                d = self.diff[name]
                p.add_(d.to(dtype=p.dtype, device=p.device), alpha=gamma)

    @torch.no_grad()
    def restore(self, model, gamma=0.01):
        if not self.diff:
            raise ValueError('Please run perbute() before restore!')

        for name, p in model.named_parameters():
            if name in self.diff and p.ndim > 1:
                d = self.diff[name]
                p.add_(d.to(dtype=p.dtype, device=p.device), alpha=-gamma)





def apply_parametrizations(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, layers.DOLinear, layers.DOConv1d, layers.DOConv2d)):
            if not nn.utils.parametrize.is_parametrized(m, "weight"):
                nn.utils.parametrizations.spectral_norm(m)
