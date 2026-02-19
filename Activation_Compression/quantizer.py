import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Subset, DataLoader

import numpy as np
import cupy as cp




class Quantizer():
    def __init__(self, config, target_dict):
        self.unrelated_tensors = set()
        self.default_bit = config['default_bits']
        self.config = config

        self.bits = target_dict
        self.dims = {}

        self.graph_mode = False


    def filter_tensors(self, pair):
        for _, v in pair:
            self.unrelated_tensors.add(v.data_ptr())

    def check_quantize(self, input_tensor):
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False
        
        if input_tensor.numel() > 0 and input_tensor.dtype == torch.uint8:
            return False
        
        if input_tensor.dtype not in [torch.float16, torch.float32, torch.bfloat16]:
            return False
        
        if input_tensor.requires_grad is False:
            return False
        
        if (len(input_tensor.shape != 2) and (len(input_tensor.shape != 3) and (len(input_tensor.shape != 4)))):
            return False
        
        return True
    
    def clean_iterate(self):
        self.bits.clear()
        self.dims.clear()

    def iterate(self, model=None, criterion=None, dataloader=None):
        if self.config['auto_precision'] is None:
            for target, _ in self.bits.items():
                # self.bits[target] = torch.tensor(self.config['default_bits'],  dtype=torch.int32, device='cuda')
                self.bits[target] = self.config['default_bits']
        else:
            # self.model = model.eval()
            # self.layer_largest_hessian_eigenvalue = {}
            # self.layer_hessian_eigenvalue_spectral_density = {}

            # sub_data_size = torch.arange(0, 64) 
            # self.criterion = criterion
            # dataset = Subset(dataloader.dataset, sub_data_size)
            # dataloader = DataLoader(dataset, shuffle=True, batch_size=64, drop_last=True)

            # x, y = next(iter(dataloader))
            # self.x_cuda, self.y_cuda = x.to('cuda'), y.to('cuda')

            # for name, module in self.model.named_modules():
            #     if hasattr(module, 'weight') and module.weight is not None:
            #         eigenvalue = self.max_eigenvalue(module.weight, iters=20)
            #         # eigenvalue_spectral_density = self.hessian_spectral_density(module.weight)
            #         self.layer_largest_hessian_eigenvalue[name] = eigenvalue
            #         # self.layer_hessian_eigenvalue_spectral_density[name] = eigenvalue_spectral_density
            
            # for target, _ in self.bits.items():
            #     # bit = self.hessian2bit_lookup_table(self.layer_hessian_eigenvalue[target])
            #     # self.bits[target] = bit  
            #     self.bits[target] = self.config['default_bits'] 
            self.low_rank_activations, self.ratio, self.k = self.low_rank_approximation(model, criterion, dataloader, n_approx=1, rank_ratio=0.9)

    def normalize(self, vec):
        norm_sq = sum((v_i ** 2).sum() for v_i in vec)
        norm = torch.sqrt(norm_sq)

        return [v_i / (norm + 1e-12) for v_i in vec]

    def layer_hpv(self, layer_params, v):
        self.model.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with nn.attention.sdpa_kernel(nn.attention.SDPBackend.MATH):
                logits = self.model(self.x_cuda)
            
                loss = self.criterion(logits, self.y_cuda)
                grads = grad(loss, layer_params, create_graph=True)

                g_dot_v = sum((g * v_i).sum() for g, v_i in zip(grads, v))

                hpv = grad(g_dot_v, layer_params, allow_unused=True)
        return hpv
    
    def max_eigenvalue(self, layer_params, iters=20):
        v = [torch.randn_like(p) for p in layer_params]
        
        v_norm = self.normalize(v)
        eigenvalue = None
        for _ in range(iters):
            hv = self.layer_hpv(layer_params, v_norm)

            num = sum((v_norm_i * hv_i).sum() for v_norm_i, hv_i in zip(v_norm, hv))
            dem = sum((v_norm_i * v_norm_i).sum() for v_norm_i in v_norm)
            eigenvalue = (num / (dem + 1e-12)).item()

            v_norm = self.normalize(hv)
        return eigenvalue
    
    def hessian2bit_lookup_table(self, eigenvalue):
        if eigenvalue >= 1000:
            return 8
        elif eigenvalue >= 500:
            return 4
        elif eigenvalue >= 250:
            return 3
        elif eigenvalue >= 50:
            return 2
        else:
            return 1
        
    def lanczos(self, layer_params, m=30):
        q = [torch.randn_like(p) for p in layer_params]
        q = self.normalize(q)

        alphas = []
        betas = []

        q_prev = None

        for _ in range(m):
            Hq = self.layer_hpv(layer_params, q)

            alpha = sum((qi * hi).sum() for qi, hi in zip(q, Hq))
            alphas.append(alpha.item())

            if q_prev is not None:
                Hq = [hi - alpha * qi - beta * qpi
                    for hi, qi, qpi in zip(Hq, q, q_prev)]
            else:
                Hq = [hi - alpha * qi for hi, qi in zip(Hq, q)]

            beta = torch.sqrt(sum((hi**2).sum() for hi in Hq))
            betas.append(beta.item())

            q_prev = q
            q = [hi / (beta + 1e-12) for hi in Hq]

        return alphas, betas
    
    def tridiag_eigs(self, alphas, betas):
        T = np.diag(alphas) + np.diag(betas[:-1], 1) + np.diag(betas[:-1], -1)
        return np.linalg.eigvalsh(T)
    
    def hessian_spectral_density(self, layer_params, K=20, m=30):
        eigs = []
        for _ in range(K):
            alphas, betas = self.lanczos(layer_params, m)
            eigs.extend(self.tridiag_eigs(alphas, betas))
        return np.array(eigs)
        

    def select_division_layer_helper(self, model, criterion, dataloader):
        model.train()
        activation_cache = {}
        low_frequency_energy_ratio_grad = {}
        low_frequency_energy_ratio_act = {}
        lars_trust_ratio = {}
        signal_noise_ratio = {}
        activation_var = {}

        sub_data_size = torch.arange(0, 64)
        sub_dataset = Subset(dataloader.dataset, sub_data_size)
        sub_dataloader = DataLoader(sub_dataset, shuffle=True, batch_size=64, drop_last=True)
        x, y = next(iter(sub_dataloader))

        def forward_hook(name):
            def hook(module, input, output):
                activation_cache[name] = input[0]
            return hook

        @torch.no_grad()
        def compute_lfer_act(x, k=4):
            if x.ndim == 4:
                H, W = x.shape[-2:]
                k = min(k, H, W)
                x_lp = F.interpolate(F.avg_pool2d(x, kernel_size=k, stride=k, padding=0), size=x.shape[-2:], mode='nearest')
                return (x_lp.square().sum() / (x.square().sum() + 1e-7)).item()
            else:
                feature_dim = x.shape[-1]
                k = min(k, feature_dim)
                if x.ndim != 2:
                    pooled = F.avg_pool1d(x, kernel_size=k, stride=k, padding=0)
                    x_lp = F.interpolate(pooled, size=feature_dim, mode='nearest')
                else:
                    x_lp = x.mean(dim=-1, keepdim=True).expand_as(x)
                return (x_lp.square().sum() / (x.square().sum() + 1e-7)).item()

        @torch.no_grad()
        def compute_lfer_grad(p, k=4):
            dW = p.grad.detach()
        
            # if dW.ndim == 4:
            #     Kh, Kw = dW.shape[-2], dW.shape[-1]
            #     if Kh == 1 and Kw == 1:
            #         return 1  
            #     dc = dW.mean(dim=(-2, -1), keepdim=True) 
            # else:
            #     feature_dim = dW.shape[-1]
            #     if feature_dim == 1:
            #         return 1
            #     dc = dW.mean(dim=-1, keepdim=True)
            # dW_lf = dc.expand_as(dW)               

            # num = (dW_lf * dW_lf).sum()
            # den = (dW * dW).sum().add(1e-7)
            # return (num / den).item()
            if dW.ndim == 2:
                feature_dim = dW.shape[-1]
                if feature_dim == 1:
                    return 1
                grad_lf_pooled = dW.mean(dim=-1, keepdim=True)
                grad_lf = grad_lf_pooled.expand_as(dW)
                return (grad_lf.square().sum() / (dW.square().sum() + 1e-7)).item()
            elif dW.ndim == 3:
                feature_dim = dW.shape[-1]
                if feature_dim == 1:
                    return 1
                k = min(k, feature_dim)
                pooled = F.avg_pool1d(dW, kernel_size=k, stride=3, padding=0)
                grad_lf = F.interpolate(pooled, size=feature_dim, mode='nearest')
                return (grad_lf.square().sum() / (dW.square().sum() + 1e-7)).item()
            elif dW.ndim == 4:
                H, W = dW.shape[-2:]
                if H == 1 and W == 1:
                    return 1
                k = min(k, H, W)
                pooled = F.avg_pool2d(dW, kernel_size=k, stride=k, padding=0)
                grad_lf = F.interpolate(pooled, size=(H, W), mode='nearest')
                return (grad_lf.square().sum() / (dW.square().sum() + 1e-7)).item()
            else:
                feature_dim = dW.shape[0]
                if feature_dim == 1:
                    return 1
                dc = dW.mean(dim=-1, keepdim=True)
                dW_lf = dc.expand_as(dW)               

                num = (dW_lf * dW_lf).sum()
                den = (dW * dW).sum().add(1e-7)
                return (num / den).item()


        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d, nn.Linear)):
                module.register_forward_hook(forward_hook(name))

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            x, y = x.to('cuda'), y.to('cuda')
            raw_logits = model(x)

            if isinstance(raw_logits, tuple):
                logits = raw_logits[0]
            else:
                logits = raw_logits

            loss = criterion(logits, y)
            loss.backward()

        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d, nn.Linear)) or module.weight.grad is None:
                continue

            lfer_grad = compute_lfer_grad(module.weight)
            lfer_act = compute_lfer_act(activation_cache[name])
            low_frequency_energy_ratio_grad[name] = lfer_grad
            low_frequency_energy_ratio_act[name] = lfer_act
            lars_trust_ratio[name] = torch.norm(module.weight).item() / (torch.norm(module.weight.grad).item() + 1e-7) 
            signal_noise_ratio[name] = torch.abs(module.weight.grad.mean()).item() / (torch.sqrt(module.weight.grad.var()) + 1e-7).item()
            activation_var[name] = activation_cache[name].var().item()
        return low_frequency_energy_ratio_grad, low_frequency_energy_ratio_act, lars_trust_ratio, signal_noise_ratio, activation_var


    def low_rank_approximation(self, model, criterion, dataloader, n_approx=2, rank_ratio=0.5):
        model.train()
        activation_cache = {}
        oei_shape = {} 
        auto_correlation_matrix = {}
        low_rank_activations = {}

        sub_data_size = torch.arange(0, 64) 
        self.criterion = criterion
        dataset = Subset(dataloader.dataset, sub_data_size)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=64, drop_last=True)

        def forward_hook(name):
            def hook(module, input, output):
                activation_cache[name] = input[0]
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                module.register_forward_hook(forward_hook(name))

        for n in range(n_approx):
            for x, y in dataloader:
                x, y = x.to('cuda'), y.to('cuda')

                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = model(x)

                # Compute auto-correlation matrices for each activation and accumulate it
                for name, act in activation_cache.items():
                    if len(act.shape) != 4:
                        continue

                    N, C, H, W = act.shape
                    oei_shape[name] = (N, C, H, W)
                    act = act.float()
                    act_flat = act.permute(1, 0, 2, 3).reshape(C, -1).contiguous()
                    Nsample = act_flat.shape[1]
                    _auto_correlation_matrix = torch.mm(act_flat, act_flat.T) / Nsample

                    if name in auto_correlation_matrix:
                        auto_correlation_matrix[name] += _auto_correlation_matrix
                    else:
                        auto_correlation_matrix[name] = _auto_correlation_matrix
                    
        
        # Normalize the accumulated auto-correlation matrices
        for name, acm in auto_correlation_matrix.items():
            auto_correlation_matrix[name] = acm / (n_approx * len(dataloader))
            
        # Perform eigen-decomposition and low-rank approximation
        for name, acm in auto_correlation_matrix.items():
            eigenvalues, eigenvectors = torch.linalg.eigh(acm)
            
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            ratio = torch.cumsum(eigenvalues, dim=0) / torch.sum(eigenvalues)
            k = int(torch.searchsorted(ratio, rank_ratio).item()) + 1
            k = max(1, min(k, eigenvectors.shape[1]))

            _low_rank_eigenvectors = eigenvectors[:, :k]
            low_rank_activations[name] = _low_rank_eigenvectors
                    
        return low_rank_activations, ratio, k


    


    

