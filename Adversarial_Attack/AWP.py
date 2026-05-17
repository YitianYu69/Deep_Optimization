import torch
from torch import nn
import torch.nn.functional as F

from ..Activation_Compression.modules import layers

from typing import Callable
import copy


def normalize_and_cmpute(diff, ori_weights):
    return diff * (ori_weights.norm() / (diff.norm() + 1e-8))



class AWP():
    def __init__(self,
                 proxy_cls: nn.Module,
                 proxy_kwarys: dict,
                 cri: Callable,
                 proxy_opt: torch.optim.Optimizer,
                 opt_kwargs: dict,
                 gamma: float = 0.01,
                 device: str = 'cuda'):

        self.proxy_cls = proxy_cls # Object Class
        self.proxy_kwarys = proxy_kwarys

        self.cri = cri
        self.proxy_opt_cls = proxy_opt
        self.opt_kwargs = opt_kwargs
        self.diff = None

        self.gamma = gamma
        self.device = device


    def compute_diff(self, adv_data, target, model, epoch, num_iters=1, clean_data=None):

        proxy = self.proxy_cls(**self.proxy_kwarys).to(self.device)
        # if epoch <= 2:
        # apply_parametrizations(proxy)
        
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

        if clean_data is not None:
            with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                clean_logits = proxy(clean_data)
                clean_prob = F.softmax(clean_logits, dim=1).detach()

        for _ in range(num_iters):

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                proxy_opt.zero_grad(set_to_none=True)

                logits = proxy(adv_data)
                loss = -self.cri(logits, target)

                if clean_data is not None:
                    loss -= F.kl_div(
                        F.log_softmax(logits, dim=1),
                        clean_prob,
                        reduction='sum'
                    )

                loss.backward()
                proxy_opt.step()
            
        self.diff = {}
        for name, p in proxy.named_parameters():
            if name in ori_params:
                diff = p.detach() - ori_params[name]
                self.diff[name] = normalize_and_cmpute(diff, ori_params[name])

        del proxy

    @torch.no_grad()
    def perturbate(self, model):
        if not self.diff:
            raise ValueError('Please run perbute() before restore!')

        for name, p in model.named_parameters():
            if name in self.diff and p.ndim > 1:
                d = self.diff[name]
                p.add_(d.to(dtype=p.dtype, device=p.device), alpha=gamma)

    @torch.no_grad()
    def restore(self, model):
        if not self.diff:
            raise ValueError('Please run perbute() before restore!')

        for name, p in model.named_parameters():
            if name in self.diff and p.ndim > 1:
                d = self.diff[name]
                p.add_(d.to(dtype=p.dtype, device=p.device), alpha=-gamma)





class Symmetric(nn.Module):
    """
    Symmetrize the last two dimensions.

    Supports:
        2D: [N, N]
        4D: [C_out, C_in, K, K]

    For Conv2d weights, this makes each spatial kernel symmetric.
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim not in (2, 4):
            raise ValueError(
                f"Symmetric only supports 2D or 4D tensors, but got shape {tuple(X.shape)}"
            )

        if X.shape[-1] != X.shape[-2]:
            raise ValueError(
                f"Last two dims must be square, but got shape {tuple(X.shape)}"
            )

        upper = torch.triu(X, diagonal=0)
        upper_no_diag = torch.triu(X, diagonal=1)

        return upper + upper_no_diag.transpose(-1, -2)

    def right_inverse(self, S: torch.Tensor) -> torch.Tensor:
        """
        Needed for torch.nn.utils.parametrize.
        Stores only the upper-triangular part.
        """
        if S.ndim not in (2, 4):
            raise ValueError(
                f"Symmetric only supports 2D or 4D tensors, but got shape {tuple(S.shape)}"
            )

        if S.shape[-1] != S.shape[-2]:
            raise ValueError(
                f"Last two dims must be square, but got shape {tuple(S.shape)}"
            )

        return torch.triu(S, diagonal=0)


def apply_parametrizations(model):
    for m in model.modules():
        if isinstance(m, (
            nn.Linear, nn.Conv1d, nn.Conv2d,
            layers.DOLinear, layers.DOConv1d, layers.DOConv2d
        )):
            if not nn.utils.parametrize.is_parametrized(m, "weight"):
                # parametrizations.spectral_norm(m, name="weight")
                if m.weight.shape[-1] == m.weight.shape[-2]:
                    nn.utils.parametrize.register_parametrization(m, "weight", Symmetric())
