import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torch.linalg as LA

import math

from ..Train.log import get_logger

logger = get_logger()


class SGD_NS_Overshoot(Optimizer):
    def __init__(self,
                 params,
                 lr,
                 actual_bs,
                 noise_decay_steps,
                 momentum=0.9,
                 overshoot=0.0,
                 weight_decay=0.0,
                 dampening=0.0,
                 nesterov=True,
                 rms_beta=0.99,
                 eps=1e-8):
        if overshoot > 0 and momentum == 0.0:
            raise ValueError("Overshoot requires momentum to be non zero!")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            overshoot=overshoot,
            weight_decay=weight_decay,
            dampening=dampening,
            rms_beta=rms_beta,
            nesterov=nesterov,
            eps=eps
        )
        super().__init__(params, defaults)

        self._global_step = 0
        self._temp_step = 0
        self._noise_decay_steps = noise_decay_steps

        self._actual_bs = actual_bs
        self._effective_bs = 512
        self._noise_scale = (self._actual_bs / self._effective_bs) ** 0.5

    
    def _gradient_centralization(self, grad):
        if grad.ndim == 4: 
            # return grad - grad.mean(dim=tuple(range(1 - grad.ndim)), keepdim=True)
            Gf = torch.fft.rfft2(grad, norm='ortho')

            mean = Gf.mean(dim=tuple(range(1, Gf.ndim)), keepdim=True)
            # mean_norm = LA.vector_norm(mean.expand_as(Gf))
            # grad_norm = LA.vector_norm(Gf)
            # ratio = mean_norm / (grad_norm + 1e-8)
            # Gf = Gf - ratio * mean
            Gf = Gf - mean
            return torch.fft.irfft2(Gf, s=grad.shape[-2:], norm='ortho')

            # mean = grad.mean(dim=tuple(range(1 - grad.ndim)), keepdim=True)
            # mean_norm = LA.vector_norm(mean)
            # grad_norm = LA.vector_norm(grad)
            # ratio = mean_norm / (grad_norm + 1e-8)
            # ratio = ratio / (1.0 + ratio)
            # return grad - ratio * mean

        else:
            return grad


    def _compute_trust_ratio(self, grad, p, eps):
        if grad.numel() == 0 or p.ndim == 1:
            return grad.new_tensor(1.0)

        grad_norm = LA.vector_norm(grad)
        p_norm = LA.vector_norm(p)

        if grad_norm < eps or p_norm < eps:
            return grad.new_tensor(1.0)

        ratio = p_norm / (grad_norm + eps)

        if torch.isnan(ratio) or torch.isinf(ratio):
            return grad.new_tensor(1.0)
        
        return ratio * 1e-3
    
    def _update_param_ratio_status(self):
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    continue

                state = self.state[p]
                ratio = self._compute_trust_ratio(grad, p, 1e-8)

                state['ratio'] = ratio.clone()

    def _generate_heavy_tailed_noise(self, x, alpha=3.0, complex=False, clamp_value=10.0, eps=1e-8):
        u = torch.rand_like(x)
        base = u.pow(-1.0 / alpha) - 1.0

        if complex:
            real = torch.randn_like(x.real)
            imag = torch.randn_like(x.real)
            direction = torch.complex(real, imag)
            direction = direction / (direction.abs() + eps)
        else:
            direction = torch.sign(torch.randn_like(x))

        noise = direction * base

        if complex:
            mag = noise.abs()
            scale = torch.clamp(mag, max=clamp_value) / (mag + eps)
            noise *= scale
        else:
            noise = torch.clamp(noise, -clamp_value, clamp_value)
        
        return noise
    

    def _newtonschulz5(self, x, steps=5, eps=1e-8):
        assert x.ndim == 2
        a, b, c = (3.4445, -4.7750, 2.0315)

        X = x.bfloat16()
        X /= (LA.vector_norm(X) + eps)
        if X.shape[0] > X.shape[1]:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if X.shape[0] > X.shape[1]:
            X = X.T
        return X


    def _compute_cosine_similarity(self, grad, m, eps=1e-8):
        grad = grad.reshape(-1)
        m = m.reshape(-1)
        return F.cosine_similarity(grad, m, dim=0)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        if self._global_step <= 1:
            self._update_param_ratio_status()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            overshoot = group['overshoot']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            rms_beta = group['rms_beta']
            nesterov = group['nesterov']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if grad.is_sparse:
                    raise TypeError('Current Optimizer does not support sparse gradient!')
                
            
                # ----------------------------------------
                # 1.0 Apply weight gradient ratio to grad
                # ----------------------------------------
                ratio = state.get('ratio', torch.tensor(1.0, dtype=grad.dtype, device=grad.device))
                grad *= ratio

            
                # ----------------------------------------
                # 2.0 Apply weight decay
                # ----------------------------------------
                def should_decay(p):
                    if p.ndim <= 1:
                        return False  # skip bias and norm weights

                    # if p.ndim == 2 and min(p.shape) <= 64:
                    #     return False  # skip small Linear / 2D matrices

                    return True
                    
                if weight_decay != 0.0 and should_decay(p):
                    p.add_(p, alpha=-lr * weight_decay)

                # ----------------------------------------
                # 3.0 Apply gradient geometry precondition
                # ----------------------------------------
                if grad.ndim == 4:
                    out_c, in_c, k1, k2 = grad.shape
                    _grad = grad.view(out_c, -1)
                    _grad = self._newtonschulz5(_grad)
                    grad = _grad.reshape(out_c, in_c, k1, k2)
                elif grad.ndim == 2 and min(grad.shape) >= 64:
                    grad = self._newtonschulz5(grad)
                grad = grad.float()


                grad = self._gradient_centralization(grad)

                # ----------------------------------------
                # 4.0 Apply RMS precondition
                # ----------------------------------------               
                if "exp_avg_sq" not in state:
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg_sq = state['exp_avg_sq']
                exp_avg_sq.mul_(rms_beta).addcmul_(grad, grad, value=1.0 - rms_beta)

                bias_correction = 1.0 - (rms_beta ** self._global_step)
                denom = (exp_avg_sq / bias_correction).sqrt().add_(eps)
                grad /= denom


                # ----------------------------------------
                # 5.0 Apply heavy tailed noise
                # ----------------------------------------   
                noise_decay = max((self._noise_decay_steps - self._global_step) / self._noise_decay_steps, 0.1)
                if grad.ndim == 4:

                    Gf = torch.fft.rfft2(grad, norm='ortho')

                    noise = self._generate_heavy_tailed_noise(Gf, complex=True)
                    noise = noise / (noise.abs().mean() + eps)
                    Gf_noisy = Gf + noise * self._noise_scale * noise_decay * ratio

                    grad = torch.fft.irfft2(Gf_noisy, s=grad.shape[-2:], norm='ortho')

                elif grad.ndim == 2:
                    noise = self._generate_heavy_tailed_noise(grad, complex=False)
                    noise = noise / (noise.abs().mean() + eps)

                    grad = grad + noise * self._noise_scale * noise_decay * ratio


                # ----------------------------------------
                # 6.0 Apply momentum and nesterov
                # ----------------------------------------  
                if momentum != 0.0: 
                    if "momentum" not in state:
                        buf = state['momentum'] = grad.clone()
                        m = buf
                    else:
                        buf = state['momentum']
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)

                        if nesterov:
                            m = grad.add_(buf, alpha=momentum)
                        else:
                            m = buf
                else:
                    m = grad


                # -------------------------------------------
                # 7.0 Apply directional conditioned overshoot
                # -------------------------------------------
                if overshoot != 0.0 and momentum != 0.0:
                    alignment_gate = self._compute_cosine_similarity(grad, m)
                    alignment_gate = alignment_gate / (1.0 + alignment_gate)
                    effective_overshoot = overshoot * alignment_gate

                    gc = -lr * effective_overshoot / momentum
                    mc = -lr * (effective_overshoot - (effective_overshoot / momentum) + 1)

                    p.add_(grad, alpha=gc)
                    p.add_(m, alpha=mc)
                else:
                    p.add_(m, alpha=-lr)

        return loss


    def tiny_max_step(self, p, momentum, dampening, rms_beta, nesterov, eps):
            if p.grad is None:
                return None

            grad = p.grad
            state = self.state[p]
            if grad.is_sparse:
                raise TypeError('Current Optimizer does not support sparse gradient!')
            
        
            # ----------------------------------------
            # 1.0 Apply weight gradient ratio to grad
            # ----------------------------------------
            ratio = state.get('ratio', torch.tensor(1.0, dtype=grad.dtype, device=grad.device))
            grad *= ratio

            # ----------------------------------------
            # 3.0 Apply gradient geometry precondition
            # ----------------------------------------
            if grad.ndim == 4:
                out_c, in_c, k1, k2 = grad.shape
                _grad = grad.view(out_c, -1)
                _grad = self._newtonschulz5(_grad)
                grad = _grad.reshape(out_c, in_c, k1, k2)
            elif grad.ndim == 2 and min(grad.shape) >= 64:
                grad = self._newtonschulz5(grad)
            grad = grad.float()


            grad = self._gradient_centralization(grad)

            # ----------------------------------------
            # 4.0 Apply RMS precondition
            # ----------------------------------------               
            exp_avg_sq = state['exp_avg_sq']
            exp_avg_sq.mul(rms_beta).addcmul(grad, grad, value=1.0 - rms_beta)

            bias_correction = 1.0 - (rms_beta ** self._global_step)
            denom = (exp_avg_sq / bias_correction).sqrt().add(eps)
            grad /= denom


            # ----------------------------------------
            # 5.0 Apply heavy tailed noise
            # ----------------------------------------   
            noise_decay = max((self._noise_decay_steps - self._global_step) / self._noise_decay_steps, 0.1)
            if grad.ndim == 4:

                Gf = torch.fft.rfft2(grad, norm='ortho')

                noise = self._generate_heavy_tailed_noise(Gf, complex=True)
                noise = noise / (noise.abs().mean() + eps)
                Gf_noisy = Gf + noise * self._noise_scale * noise_decay * ratio

                grad = torch.fft.irfft2(Gf_noisy, s=grad.shape[-2:], norm='ortho')

            elif grad.ndim == 2:
                noise = self._generate_heavy_tailed_noise(grad, complex=False)
                noise = noise / (noise.abs().mean() + eps)

                grad = grad + noise * self._noise_scale * noise_decay * ratio


            # ----------------------------------------
            # 6.0 Apply momentum and nesterov
            # ----------------------------------------  
            if momentum != 0.0: 
                buf = state['momentum']
                buf.mul(momentum).add(grad, alpha=1.0 - dampening)

                if nesterov:
                    m = grad.add(buf, alpha=momentum)
                else:
                    m = buf
            else:
                m = grad


            return m

    



class SGD_NS_Overshoot_Noise(Optimizer):
    def __init__(self,
                 params,
                 lr,
                 actual_bs,
                 noise_decay_steps,
                 momentum=0.9,
                 overshoot=0.0,
                 weight_decay=0.0,
                 dampening=0.0,
                 nesterov=True,
                 rms_beta=0.99,
                 eps=1e-8,

                 # ---- layerwise noise-adaptive LR ----
                 layer_noise_beta=0.99,
                 layer_noise_alpha=1.0,
                 min_lr_scale=0.15,
                 max_lr_scale=10.0,
                 use_layerwise_noise=True):
        if overshoot > 0 and momentum == 0.0:
            raise ValueError("Overshoot requires momentum to be non zero!")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            overshoot=overshoot,
            weight_decay=weight_decay,
            dampening=dampening,
            rms_beta=rms_beta,
            nesterov=nesterov,
            eps=eps,

            layer_noise_beta=layer_noise_beta,
            layer_noise_alpha=layer_noise_alpha,
            min_lr_scale=min_lr_scale,
            max_lr_scale=max_lr_scale,
            use_layerwise_noise=use_layerwise_noise,
        )
        super().__init__(params, defaults)

        self._global_step = 0
        self._temp_step = 0
        self._noise_decay_steps = noise_decay_steps

        self._actual_bs = actual_bs
        self._effective_bs = 512
        self._noise_scale = (self._actual_bs / self._effective_bs) ** 0.5

        self._base_weights = False

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _gradient_centralization(self, grad):
        if grad.ndim == 4:
            Gf = torch.fft.rfft2(grad, norm='ortho')
            mean = Gf.mean(dim=tuple(range(1, Gf.ndim)), keepdim=True)
            Gf = Gf - mean
            return torch.fft.irfft2(Gf, s=grad.shape[-2:], norm='ortho')
        else:
            return grad

    def _compute_trust_ratio(self, grad, p, eps):
        if grad.numel() == 0 or p.ndim == 1:
            return grad.new_tensor(1.0)

        grad_norm = LA.vector_norm(grad)
        p_norm = LA.vector_norm(p)

        if grad_norm < eps or p_norm < eps:
            return grad.new_tensor(1.0)

        ratio = p_norm / (grad_norm + eps)

        if torch.isnan(ratio) or torch.isinf(ratio):
            return grad.new_tensor(1.0)

        return ratio * 1e-3

    def _update_param_ratio_status(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    continue

                state = self.state[p]
                ratio = self._compute_trust_ratio(grad, p, 1e-8)
                state['ratio'] = ratio.detach().clone()

    def _generate_heavy_tailed_noise(self, x, alpha=3.0, complex=False, clamp_value=10.0, eps=1e-8):
        u = torch.rand_like(x)
        base = u.pow(-1.0 / alpha) - 1.0

        if complex:
            real = torch.randn_like(x.real)
            imag = torch.randn_like(x.real)
            direction = torch.complex(real, imag)
            direction = direction / (direction.abs() + eps)
        else:
            direction = torch.sign(torch.randn_like(x))

        noise = direction * base

        if complex:
            mag = noise.abs()
            scale = torch.clamp(mag, max=clamp_value) / (mag + eps)
            noise *= scale
        else:
            noise = torch.clamp(noise, -clamp_value, clamp_value)

        return noise

    def _newtonschulz5(self, x, steps=5, eps=1e-8):
        assert x.ndim == 2
        # a, b, c = (3.4445, -4.7750, 2.0315)

        COEFF_MAP_CANONICAL = {
            (2048, 2048): (3.4916, -4.8224, 2.1095),
            (2048, 16384): (3.1943, -3.8221, 1.5322),
            (2048, 3072): (3.4509, -4.5790, 1.9325),
            (2048, 8192): (2.9794, -3.6674, 1.6207)
        }
        
        rows, cols = x.size(0), x.size(1)
        shape_key = (min(rows, cols), max(rows, cols))
        
        try:
            a, b, c = COEFF_MAP_CANONICAL[shape_key]
        except KeyError:
            # Fallback to default Newton-Schulz coefficients for unknown shapes
            a, b, c = (3.4445, -4.7750, 2.0315)

        X = x.bfloat16()
        X /= (LA.vector_norm(X) + eps)
        transposed = False

        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

        if transposed:
            X = X.T
        return X

    def _compute_cosine_similarity(self, grad, m, eps=1e-8):
        grad = grad.reshape(-1)
        m = m.reshape(-1)
        return F.cosine_similarity(grad, m, dim=0)

    def _group_key(self, p):
        """
        Group tensors with similar geometry together for layerwise normalization.
        """
        if p.ndim == 4:
            return "conv4d"
        elif p.ndim == 2 and min(p.shape) >= 64:
            return "mat2d_large"
        elif p.ndim == 2:
            return "mat2d_small"
        elif p.ndim == 1:
            return "vec1d"
        else:
            return f"other_{p.ndim}d"

    # def _noise_metric(self, g, eps=1e-8):
    #     """
    #     First practical version:
    #     use squared Frobenius / l2 diff.
    #     This is the safest starting point.
    #     """
    #     return g.float().pow(2).sum()

    # def _matrix_nuclear_proxy(self, X, eps=1e-8):
    #     X = X.float()
    #     fro = torch.linalg.norm(X, ord='fro')
    #     spec = torch.linalg.matrix_norm(X, ord=2)
    #     stable_rank = (fro / (spec + eps)) ** 2
    #     return fro * torch.sqrt(stable_rank.clamp_min(1.0))

    # def _matrix_nuclear_proxy(self, X, eps=1e-8):
    #     X = X.float()
    #     fro = torch.linalg.norm(X, ord='fro')
    #     spec = torch.linalg.matrix_norm(X, ord=2)
    #     rank_proxy = (fro / (spec + eps)) ** 2
    #     nuc_proxy = torch.sqrt(rank_proxy.clamp_min(1.0)) * spec
    #     return nuc_proxy
    
    # def _noise_metric(self, diff, eps=1e-8):
    #     if diff.ndim == 4:
    #         X = diff.view(diff.shape[0], -1)
    #         return self._matrix_nuclear_proxy(X, eps=eps) ** 2
    #     elif diff.ndim == 2 and min(diff.shape) >= 64:
    #         return self._matrix_nuclear_proxy(diff, eps=eps) ** 2
    #     elif diff.ndim == 2:
    #         return diff.float().pow(2).mean()
    #     elif diff.ndim == 1:
    #         return diff.float().pow(2).mean()
    #     else:
    #         return diff.float().pow(2).mean()


    def _noise_metric(self, diff, eps=1e-8):
        if diff.ndim == 4:
            X = diff.view(diff.shape[0], -1).float()
            d_out, d_in = X.shape
            nuc = torch.linalg.norm(X, ord='nuc')
            dual = math.sqrt(d_out / max(d_in, 1)) * nuc
            return dual ** 2
        elif diff.ndim == 2 and min(diff.shape) >= 64:
            X = diff.float()
            d_out, d_in = X.shape
            nuc = torch.linalg.norm(X, ord='nuc')
            dual = math.sqrt(d_out / max(d_in, 1)) * nuc
            return dual ** 2
        elif diff.ndim == 2:
            return diff.float().pow(2).mean()
        elif diff.ndim == 1:
            return diff.float().pow(2).mean()
        else:
            return diff.float().pow(2).mean()

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        if self._base_weights:
            raise Exception("Calling `step` without calling `move_to_overshoot` first.")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        if self._global_step <= 1:
            self._update_param_ratio_status()

        # ==============================================================
        # PASS 1:
        # Build geometry-processed gradient and update noise EMA
        # ==============================================================
        group_alpha_max = {}

        for group in self.param_groups:
            eps = group['eps']
            layer_noise_beta = group['layer_noise_beta']
            layer_noise_alpha = group['layer_noise_alpha']
            use_layerwise_noise = group['use_layerwise_noise']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise TypeError('Current Optimizer does not support sparse gradient!')

                state = self.state[p]

                ratio = state.get('ratio', torch.tensor(1.0, dtype=grad.dtype, device=grad.device))
                g = grad.detach().clone()
                g = g * ratio

                # # Clean up
                # epsilon = torch.quantile(g.abs().float(), 0.9) + eps
                # o = g.sign() * F.relu(g.abs() - epsilon)
                # g = g - o


                # Geometry processing only
                if g.ndim == 4:
                    out_c, in_c, k1, k2 = g.shape
                    _g = g.view(out_c, -1)
                    _g = self._newtonschulz5(_g)
                    g = _g.reshape(out_c, in_c, k1, k2)
                elif g.ndim == 2 and min(g.shape) >= 64:
                    g = self._newtonschulz5(g)


                g = g.float()
                g = self._gradient_centralization(g)

                state['geom_grad'] = g

                if use_layerwise_noise:
                    if 'prev_geom_grad' not in state:
                        state['prev_geom_grad'] = torch.zeros_like(g)
                    if 'layer_noise_ema' not in state:
                        state['layer_noise_ema'] = torch.zeros(
                            (), device=g.device, dtype=torch.float32
                        )

                    diff = g - state['prev_geom_grad']
                    noise_sq = self._noise_metric(diff, eps=eps)

                    state['layer_noise_ema'].mul_(layer_noise_beta).add_(
                        noise_sq, alpha=(1.0 - layer_noise_beta)
                    )

                    alpha_layer = layer_noise_alpha / torch.sqrt(
                        layer_noise_alpha * layer_noise_alpha + state['layer_noise_ema']
                    )
                    state['alpha_layer'] = alpha_layer
                else:
                    state['alpha_layer'] = torch.tensor(
                        1.0, device=g.device, dtype=torch.float32
                    )

                key = self._group_key(p)
                alpha_val = float(state['alpha_layer'].item())

                if key not in group_alpha_max:
                    group_alpha_max[key] = alpha_val
                else:
                    group_alpha_max[key] = max(group_alpha_max[key], alpha_val)

        # ==============================================================
        # PASS 2:
        # Apply actual update using layer-specific lr
        # ==============================================================
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            overshoot = group['overshoot']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            rms_beta = group['rms_beta']
            nesterov = group['nesterov']
            eps = group['eps']

            min_lr_scale = group['min_lr_scale']
            max_lr_scale = group['max_lr_scale']
            use_layerwise_noise = group['use_layerwise_noise']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if grad.is_sparse:
                    raise TypeError('Current Optimizer does not support sparse gradient!')

                # ----------------------------------------
                # 0.0 Compute layer-specific lr
                # ----------------------------------------
                if use_layerwise_noise:
                    key = self._group_key(p)
                    alpha_layer = state['alpha_layer']
                    alpha_max = max(group_alpha_max.get(key, 1.0), eps)

                    lr_scale = torch.sqrt(alpha_layer / alpha_max).item()
                    lr_scale = max(min_lr_scale, min(max_lr_scale, lr_scale))
                    lr_layer = lr * lr_scale
                else:
                    lr_layer = lr
                    lr_scale = 1.0

                state['lr_scale'] = lr_scale
                state['lr_layer'] = lr_layer

                # ----------------------------------------
                # 1.0 Start from cached geometry grad
                # ----------------------------------------
                grad = state['geom_grad']

                # ----------------------------------------
                # 2.0 Apply weight decay
                # ----------------------------------------
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr_layer * weight_decay)

                # ----------------------------------------
                # 3.0 Apply RMS precondition
                # ----------------------------------------
                if "exp_avg_sq" not in state:
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)

                exp_avg_sq = state['exp_avg_sq']
                grad_fp32 = grad.float()

                exp_avg_sq.mul_(rms_beta).addcmul_(grad_fp32, grad_fp32, value=1.0 - rms_beta)

                bias_correction = 1.0 - (rms_beta ** self._global_step)
                denom = (exp_avg_sq / bias_correction).sqrt().add_(eps)
                grad = grad_fp32 / denom
                

                # ----------------------------------------
                # 4.0 Apply heavy tailed noise
                # ----------------------------------------
                noise_decay = max((self._noise_decay_steps - self._global_step) / self._noise_decay_steps, 0.1)
                ratio = state.get('ratio', torch.tensor(1.0, dtype=grad.dtype, device=grad.device))

                if grad.ndim == 4:
                    Gf = torch.fft.rfft2(grad, norm='ortho')

                    noise = self._generate_heavy_tailed_noise(Gf, complex=True)
                    noise = noise / (noise.abs().mean() + eps)
                    Gf_noisy = Gf + noise * self._noise_scale * noise_decay * ratio

                    grad = torch.fft.irfft2(Gf_noisy, s=grad.shape[-2:], norm='ortho')

                elif grad.ndim == 2:
                    noise = self._generate_heavy_tailed_noise(grad, complex=False)
                    noise = noise / (noise.abs().mean() + eps)
                    grad = grad + noise * self._noise_scale * noise_decay * ratio

                # ----------------------------------------
                # 5.0 Apply momentum and nesterov
                # ----------------------------------------
                if momentum != 0.0:
                    if "momentum" not in state:
                        buf = state['momentum'] = grad.clone()
                        m = buf
                    else:
                        buf = state['momentum']
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)

                        if nesterov:
                            m = grad + momentum * buf
                        else:
                            m = buf
                else:
                    m = grad

                # ----------------------------------------
                # 6.0 Apply directional conditioned overshoot
                # ----------------------------------------
                if overshoot != 0.0 and momentum != 0.0:
                    alignment_gate = self._compute_cosine_similarity(grad, m)
                    alignment_gate = alignment_gate / (alignment_gate + 1)
                    effective_overshoot = overshoot * alignment_gate

                    gc = -lr_layer * effective_overshoot / momentum
                    mc = -lr_layer * (effective_overshoot - (effective_overshoot / momentum) + 1)

                    p.add_(grad, alpha=gc)
                    p.add_(m, alpha=mc)
                else:
                    p.add_(m, alpha=-lr_layer)
                # cos = self._compute_cosine_similarity(grad, m)

                # if cos > 0:
                #     # aligned → overshoot system
                #     effective_overshoot = overshoot * cos

                #     gc = -lr_layer * effective_overshoot / momentum
                #     mc = -lr_layer * (effective_overshoot - effective_overshoot / momentum + 1)

                #     p.add_(grad, alpha=gc)
                #     p.add_(m, alpha=mc)
                # elif cos > -0.3:
                #     # weak conflict → fallback to momentum
                #     p.add_(m, alpha=-lr_layer)
                # else:
                #     # strong conflict → trust gradient
                #     p.add_(grad, alpha=-lr_layer)

                # ----------------------------------------
                # 7.0 Save previous geometry grad
                # ----------------------------------------
                if 'prev_geom_grad' in state:
                    state['prev_geom_grad'].copy_(state['geom_grad'])

        return loss
    

    @torch.no_grad()
    def move_to_base(self):
        if len(self.state) == 0:
            return
        if self._base_weights:
            raise Exception("Calling `move_to_base` without calling `move_to_overshoot` first.")
        self._base_weights = True
        for group in self.param_groups:
            for param in group["params"]:
                if "momentum" in self.state[param]:
                    m = self.state[param]["momentum"]

                    gate = 1.0
                    if param.grad is not None:
                        grad = param.grad

                        gate = self._compute_cosine_similarity(grad, m)
                        gate = gate / (gate + 1.0)

                    self.state[param]['gate'] = gate
                    param.add_(m, alpha=self.state[param]["lr_layer"] * group["overshoot"] * gate)
                
    @torch.no_grad()
    def move_to_overshoot(self):
        if len(self.state) == 0:
            return
        if not self._base_weights:
            raise Exception("Calling `move_to_overshoot` without calling `move_to_base` first.")
        self._base_weights = False
        for group in self.param_groups:
            for param in group["params"]:
                if "momentum" in self.state[param]:
                    m = self.state[param]["momentum"]

                    gate = 1.0
                    if 'gate' in self.state[param]:
                        gate = self.state[param]['gate']
                    
                    param.add_(m, alpha=-self.state[param]["lr_layer"] * group["overshoot"] * gate)
    

    def tiny_max_step(self, p, momentum, dampening, rms_beta, nesterov, eps):
            if p.grad is None:
                return None

            grad = p.grad
            state = self.state[p]
            if grad.is_sparse:
                raise TypeError('Current Optimizer does not support sparse gradient!')
            
        
            # ----------------------------------------
            # 1.0 Apply weight gradient ratio to grad
            # ----------------------------------------
            ratio = state.get('ratio', torch.tensor(1.0, dtype=grad.dtype, device=grad.device))
            grad *= ratio

            # ----------------------------------------
            # 3.0 Apply gradient geometry precondition
            # ----------------------------------------
            if grad.ndim == 4:
                out_c, in_c, k1, k2 = grad.shape
                _grad = grad.view(out_c, -1)
                _grad = self._newtonschulz5(_grad)
                grad = _grad.reshape(out_c, in_c, k1, k2)
            elif grad.ndim == 2 and min(grad.shape) >= 64:
                grad = self._newtonschulz5(grad)
            grad = grad.float()


            grad = self._gradient_centralization(grad)

            # ----------------------------------------
            # 4.0 Apply RMS precondition
            # ----------------------------------------               
            exp_avg_sq = state['exp_avg_sq']
            exp_avg_sq.mul(rms_beta).addcmul(grad, grad, value=1.0 - rms_beta)

            bias_correction = 1.0 - (rms_beta ** self._global_step)
            denom = (exp_avg_sq / bias_correction).sqrt().add(eps)
            grad /= denom


            # ----------------------------------------
            # 5.0 Apply heavy tailed noise
            # ----------------------------------------   
            noise_decay = max((self._noise_decay_steps - self._global_step) / self._noise_decay_steps, 0.1)
            if grad.ndim == 4:

                Gf = torch.fft.rfft2(grad, norm='ortho')

                noise = self._generate_heavy_tailed_noise(Gf, complex=True)
                noise = noise / (noise.abs().mean() + eps)
                Gf_noisy = Gf + noise * self._noise_scale * noise_decay * ratio

                grad = torch.fft.irfft2(Gf_noisy, s=grad.shape[-2:], norm='ortho')

            elif grad.ndim == 2:
                noise = self._generate_heavy_tailed_noise(grad, complex=False)
                noise = noise / (noise.abs().mean() + eps)

                grad = grad + noise * self._noise_scale * noise_decay * ratio


            # ----------------------------------------
            # 6.0 Apply momentum and nesterov
            # ----------------------------------------  
            if momentum != 0.0: 
                buf = state['momentum']
                buf.mul(momentum).add(grad, alpha=1.0 - dampening)

                if nesterov:
                    m = grad.add(buf, alpha=momentum)
                else:
                    m = buf
            else:
                m = grad

            return m







import math
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch import linalg as LA


class SGD_NS_Overshoot_LANTON(Optimizer):
    def __init__(
        self,
        params,
        lr,
        actual_bs,
        noise_decay_steps,
        momentum=0.9,
        overshoot=0.0,
        weight_decay=0.0,
        dampening=0.0,
        nesterov=True,
        rms_beta=0.99,
        eps=1e-8,

        # -----------------------------
        # LANTON controls
        # -----------------------------
        lanton=True,
        lanton_alpha=1.0,
        lanton_beta=0.90,
        lanton_update_interval=10,
        lanton_min_scale=0.25,
        lanton_max_scale=1.0,
        lanton_noise_norm="fro",   # "fro" is cheap; "nuclear" is closer to paper but slower
        lanton_nuclear_rank=64,
    ):
        if overshoot > 0 and momentum == 0.0:
            raise ValueError("Overshoot requires momentum to be non-zero!")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            overshoot=overshoot,
            weight_decay=weight_decay,
            dampening=dampening,
            rms_beta=rms_beta,
            nesterov=nesterov,
            eps=eps,

            lanton=lanton,
            lanton_alpha=lanton_alpha,
            lanton_beta=lanton_beta,
            lanton_update_interval=lanton_update_interval,
            lanton_min_scale=lanton_min_scale,
            lanton_max_scale=lanton_max_scale,
            lanton_noise_norm=lanton_noise_norm,
            lanton_nuclear_rank=lanton_nuclear_rank,
        )

        super().__init__(params, defaults)

        self._global_step = 0
        self._noise_decay_steps = noise_decay_steps

        self._actual_bs = actual_bs
        self._effective_bs = 512
        self._noise_scale = (self._actual_bs / self._effective_bs) ** 0.5

    # ============================================================
    # Your original helper functions
    # ============================================================

    def _gradient_centralization(self, grad):
        if grad.ndim == 4:
            Gf = torch.fft.rfft2(grad, norm="ortho")
            mean = Gf.mean(dim=tuple(range(1, Gf.ndim)), keepdim=True)
            Gf = Gf - mean
            return torch.fft.irfft2(Gf, s=grad.shape[-2:], norm="ortho")
        else:
            return grad

    def _compute_trust_ratio(self, grad, p, eps):
        if grad.numel() == 0 or p.ndim == 1:
            return grad.new_tensor(1.0)

        grad_norm = LA.vector_norm(grad)
        p_norm = LA.vector_norm(p)

        if grad_norm < eps or p_norm < eps:
            return grad.new_tensor(1.0)

        ratio = p_norm / (grad_norm + eps)

        if torch.isnan(ratio) or torch.isinf(ratio):
            return grad.new_tensor(1.0)

        return ratio * 1e-3

    def _update_param_ratio_status(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                if grad.is_sparse:
                    continue

                state = self.state[p]
                ratio = self._compute_trust_ratio(grad, p, 1e-8)
                state["ratio"] = ratio.clone()

    def _generate_heavy_tailed_noise(
        self,
        x,
        alpha=3.0,
        complex=False,
        clamp_value=10.0,
        eps=1e-8,
    ):
        u = torch.rand_like(x)
        base = u.pow(-1.0 / alpha) - 1.0

        if complex:
            real = torch.randn_like(x.real)
            imag = torch.randn_like(x.real)
            direction = torch.complex(real, imag)
            direction = direction / (direction.abs() + eps)
        else:
            direction = torch.sign(torch.randn_like(x))

        noise = direction * base

        if complex:
            mag = noise.abs()
            scale = torch.clamp(mag, max=clamp_value) / (mag + eps)
            noise = noise * scale
        else:
            noise = torch.clamp(noise, -clamp_value, clamp_value)

        return noise

    def _newtonschulz5(self, x, steps=5, eps=1e-8):
        """
        Fixed version of your NS function.

        Your old version checked X.shape again after transposing,
        so tall matrices might not transpose back correctly.
        """
        assert x.ndim == 2

        a, b, c = 3.4445, -4.7750, 2.0315

        X = x.bfloat16()
        X = X / (LA.vector_norm(X) + eps)

        transposed = False
        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True

        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X

        if transposed:
            X = X.T

        return X

    def _compute_cosine_similarity(self, grad, m, eps=1e-8):
        grad = grad.reshape(-1)
        m = m.reshape(-1)
        return F.cosine_similarity(grad, m, dim=0)

    # ============================================================
    # LANTON helper functions
    # ============================================================

    def _lanton_group_key(self, p, group):
        """
        Group parameters for alpha_l / alpha_max normalization.

        You can override this by adding:
            {"params": ..., "lanton_group": "conv"}
        to a param group.
        """
        if "lanton_group" in group:
            return group["lanton_group"]

        if p.ndim <= 1:
            return "vector"

        if p.ndim == 4:
            return "conv_matrix"

        if p.ndim == 2:
            if min(p.shape) >= 64:
                return "large_matrix"
            else:
                return "small_matrix"

        return "other"

    @torch.no_grad()
    def _approx_nuclear_norm(self, A, rank=64, niter=1, eps=1e-8):
        """
        Randomized nuclear norm approximation.
        Closer to the paper, but more expensive than Frobenius.
        """
        A = A.float()
        m, n = A.shape
        q = min(rank, m, n)

        if q <= 0:
            return A.new_tensor(0.0)

        if min(m, n) <= 64:
            return torch.linalg.svdvals(A).sum().clamp_min(eps)

        Omega = torch.randn(n, q, device=A.device, dtype=A.dtype)
        Y = A @ Omega

        for _ in range(niter):
            Y = A @ (A.T @ Y)

        Q, _ = torch.linalg.qr(Y, mode="reduced")
        B = Q.T @ A
        s = torch.linalg.svdvals(B)

        return s.sum().clamp_min(eps)

    @torch.no_grad()
    def _lanton_dual_noise_norm(self, delta, p, mode="fro", rank=64, eps=1e-8):
        """
        Estimate layerwise stochastic gradient noise.

        Paper-like:
            matrix layers: nuclear norm proxy
            vector layers: sqrt(d) * L2 norm

        Practical:
            use Frobenius by default because nuclear norm is costly.
        """
        delta = delta.float()

        if delta.ndim <= 1:
            d = delta.numel()
            return math.sqrt(max(d, 1)) * LA.vector_norm(delta)

        D = delta.reshape(delta.shape[0], -1)
        d_out, d_in = D.shape

        if mode == "nuclear":
            nuc = self._approx_nuclear_norm(D, rank=rank, eps=eps)
            return math.sqrt(d_out / max(d_in, 1)) * nuc

        elif mode == "fro":
            return LA.vector_norm(D)

        elif mode == "l1":
            return D.abs().sum()

        else:
            raise ValueError(f"Unknown lanton_noise_norm: {mode}")

    @torch.no_grad()
    def _update_lanton_statistics(self):
        """
        First pass:
        compute H_l, alpha_l, and groupwise alpha_max.

        Important:
        use the raw gradient BEFORE NS, RMS, artificial heavy-tailed noise.
        Otherwise LANTON would measure your injected optimizer noise instead
        of the minibatch gradient noise.
        """
        group_max_alpha = {}

        for group in self.param_groups:
            if not group["lanton"]:
                continue

            beta = group["lanton_beta"]
            alpha_base = group["lanton_alpha"]
            interval = group["lanton_update_interval"]
            mode = group["lanton_noise_norm"]
            rank = group["lanton_nuclear_rank"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                raw_grad = p.grad.detach()

                if raw_grad.is_sparse:
                    raise TypeError("LANTON does not support sparse gradients.")

                state = self.state[p]

                if "lanton_H" not in state:
                    state["lanton_H"] = torch.zeros(
                        (),
                        device=p.device,
                        dtype=torch.float32,
                    )
                    state["lanton_prev_grad"] = None
                    state["lanton_alpha_l"] = torch.ones(
                        (),
                        device=p.device,
                        dtype=torch.float32,
                    )
                    state["lanton_lr_scale"] = 1.0

                prev_grad = state["lanton_prev_grad"]

                if prev_grad is not None and self._global_step % interval == 0:
                    delta = raw_grad.float() - prev_grad.float()
                    noise_norm = self._lanton_dual_noise_norm(
                        delta,
                        p,
                        mode=mode,
                        rank=rank,
                        eps=eps,
                    )

                    state["lanton_H"].mul_(beta).add_(
                        noise_norm.pow(2),
                        alpha=1.0 - beta,
                    )

                state["lanton_prev_grad"] = raw_grad.clone(
                    memory_format=torch.preserve_format
                )

                H = state["lanton_H"]

                alpha_l = alpha_base / torch.sqrt(
                    raw_grad.new_tensor(alpha_base * alpha_base).float() + H
                )

                state["lanton_alpha_l"] = alpha_l

                key = self._lanton_group_key(p, group)
                alpha_float = float(alpha_l.detach().cpu())

                if key not in group_max_alpha:
                    group_max_alpha[key] = alpha_float
                else:
                    group_max_alpha[key] = max(group_max_alpha[key], alpha_float)

        return group_max_alpha

    @torch.no_grad()
    def _get_lanton_lr_scale(self, p, group, group_max_alpha):
        if not group["lanton"]:
            return 1.0

        state = self.state[p]

        if "lanton_alpha_l" not in state:
            return 1.0

        key = self._lanton_group_key(p, group)

        alpha_l = float(state["lanton_alpha_l"].detach().cpu())
        alpha_max = group_max_alpha.get(key, alpha_l)
        alpha_max = max(alpha_max, 1e-12)

        scale = math.sqrt(alpha_l / alpha_max)

        scale = max(scale, group["lanton_min_scale"])
        scale = min(scale, group["lanton_max_scale"])

        state["lanton_lr_scale"] = scale

        return scale

    # ============================================================
    # Main optimizer step
    # ============================================================

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        if self._global_step <= 1:
            self._update_param_ratio_status()

        # ------------------------------------------------------------
        # 0. LANTON first pass
        # ------------------------------------------------------------
        group_max_alpha = self._update_lanton_statistics()

        # ------------------------------------------------------------
        # 1. Normal optimizer update
        # ------------------------------------------------------------
        for group in self.param_groups:
            base_lr = group["lr"]
            momentum = group["momentum"]
            overshoot = group["overshoot"]
            weight_decay = group["weight_decay"]
            dampening = group["dampening"]
            rms_beta = group["rms_beta"]
            nesterov = group["nesterov"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                raw_grad = p.grad.detach()

                if raw_grad.is_sparse:
                    raise TypeError("Current optimizer does not support sparse gradients!")

                state = self.state[p]

                # LANTON layerwise LR multiplier
                lanton_scale = self._get_lanton_lr_scale(p, group, group_max_alpha)
                lr = base_lr * lanton_scale

                state["effective_lr"] = lr

                # ----------------------------------------
                # 1.0 Apply weight gradient ratio to grad
                # ----------------------------------------
                ratio = state.get("ratio", raw_grad.new_tensor(1.0))

                # Important:
                # do not do grad *= ratio directly on p.grad
                grad = raw_grad.float().clone()
                grad = grad * ratio

                # ----------------------------------------
                # 2.0 Decoupled weight decay
                # ----------------------------------------
                if weight_decay != 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                # ----------------------------------------
                # 3.0 Geometry precondition: Newton-Schulz
                # ----------------------------------------
                if grad.ndim == 4:
                    out_c, in_c, k1, k2 = grad.shape
                    grad_2d = grad.reshape(out_c, -1)
                    grad_2d = self._newtonschulz5(grad_2d, eps=eps)
                    grad = grad_2d.reshape(out_c, in_c, k1, k2)

                elif grad.ndim == 2 and min(grad.shape) >= 64:
                    grad = self._newtonschulz5(grad, eps=eps)

                grad = grad.float()

                # ----------------------------------------
                # 3.5 Fourier gradient centralization
                # ----------------------------------------
                grad = self._gradient_centralization(grad)

                # ----------------------------------------
                # 4.0 RMS precondition
                # ----------------------------------------
                if "exp_avg_sq" not in state:
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(rms_beta).addcmul_(
                    grad,
                    grad,
                    value=1.0 - rms_beta,
                )

                bias_correction = 1.0 - (rms_beta ** self._global_step)
                denom = (exp_avg_sq / bias_correction).sqrt().add_(eps)

                grad = grad / denom

                # ----------------------------------------
                # 5.0 Heavy-tailed optimizer noise
                # ----------------------------------------
                noise_decay = max(
                    (self._noise_decay_steps - self._global_step)
                    / self._noise_decay_steps,
                    0.1,
                )

                if grad.ndim == 4:
                    Gf = torch.fft.rfft2(grad, norm="ortho")

                    noise = self._generate_heavy_tailed_noise(
                        Gf,
                        complex=True,
                        eps=eps,
                    )

                    noise = noise / (noise.abs().mean() + eps)

                    Gf_noisy = Gf + noise * self._noise_scale * noise_decay * ratio

                    grad = torch.fft.irfft2(
                        Gf_noisy,
                        s=grad.shape[-2:],
                        norm="ortho",
                    )

                elif grad.ndim == 2:
                    noise = self._generate_heavy_tailed_noise(
                        grad,
                        complex=False,
                        eps=eps,
                    )

                    noise = noise / (noise.abs().mean() + eps)

                    grad = grad + noise * self._noise_scale * noise_decay * ratio

                # ----------------------------------------
                # 6.0 Momentum and Nesterov
                # ----------------------------------------
                if momentum != 0.0:
                    if "momentum" not in state:
                        buf = state["momentum"] = grad.clone()
                        m = buf
                    else:
                        buf = state["momentum"]
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)

                        if nesterov:
                            m = grad + momentum * buf
                        else:
                            m = buf
                else:
                    m = grad

                # ----------------------------------------
                # 7.0 Directional conditioned overshoot
                # ----------------------------------------
                if overshoot != 0.0 and momentum != 0.0:
                    cos = self._compute_cosine_similarity(grad, m)

                    # Safer gate than cos / (1 + cos).
                    # This maps cos in [-1, 1] to [0, 1].
                    alignment_gate = ((cos + 1.0) * 0.5).clamp(0.0, 1.0)

                    effective_overshoot = overshoot * alignment_gate

                    gc = -lr * effective_overshoot / momentum
                    mc = -lr * (
                        effective_overshoot
                        - effective_overshoot / momentum
                        + 1.0
                    )

                    p.add_(grad, alpha=gc)
                    p.add_(m, alpha=mc)
                else:
                    p.add_(m, alpha=-lr)

        return loss