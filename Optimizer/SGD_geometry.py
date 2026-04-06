import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torch.linalg as LA


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
        self._noise_decay_steps = noise_decay_steps

        self._actual_bs = actual_bs
        self._effective_bs = 512
        self._noise_scale = (self._actual_bs / self._effective_bs) ** 0.5


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
                if weight_decay != 0.0:
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
                            m = grad.add_(buf, alpha=momentum)
                        else:
                            m = buf
                else:
                    m = grad


                # -------------------------------------------
                # 5.0 Apply directional conditioned overshoot
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




# class SGD_NS_Overshoot(Optimizer):
#     """
#     SGDO + RMS preconditioning + optional Nesterov

#     Combines:
#     - Overshoot (future gradient usage)
#     - RMS (Adam-like conditioning)
#     - Momentum / Nesterov acceleration

#     This version is designed to:
#     - NOT break movement (important for 2-bit)
#     - Preserve overshoot behavior
#     - Keep RMS stable
#     """

#     def __init__(
#         self,
#         params,
#         actual_bs,
#         noise_decay_steps,
#         lr=1e-3,
#         momentum=0.9,
#         overshoot=0.0,
#         weight_decay=0.0,
#         dampening=0.0,
#         nesterov=False,
#         rms_beta=0.999,
#         eps=1e-8,
#     ):
#         if overshoot > 0 and momentum <= 0:
#             raise ValueError("Overshoot requires momentum > 0")

#         defaults = dict(
#             lr=lr,
#             momentum=momentum,
#             overshoot=overshoot,
#             weight_decay=weight_decay,
#             dampening=dampening,
#             nesterov=nesterov,
#             rms_beta=rms_beta,
#             eps=eps,
#         )
#         super().__init__(params, defaults)

#         self._global_step = 0
#         self._noise_decay_steps = noise_decay_steps

#         self._effective_bs = 512
#         self._actual_bs = actual_bs
#         self._noise_scale = (self._actual_bs / self._effective_bs) ** 0.5


#     @torch.no_grad()
#     def _compute_tensor_gsnr(self, grad, p, eps):
#         if grad.numel() == 0 or p.ndim == 1:
#             return grad.new_tensor(1.0)

#         # norm = grad.norm()

#         # sigma = grad.std()
#         # sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)

#         # gsnr = norm / (sigma + eps)

#         # if not torch.isfinite(gsnr):
#         #     gsnr = grad.new_tensor(1.0)
#         g_norm = torch.linalg.vector_norm(grad)
#         w_norm = torch.linalg.vector_norm(p)

#         if g_norm < eps or w_norm < eps:
#             return grad.new_tensor(1.0)
        
#         gsnr = w_norm / (g_norm + eps)

#         if torch.isnan(gsnr) or torch.isinf(gsnr):
#             gsnr = grad.new_tensor(1.0)

#         return gsnr * 1e-3

#     @torch.no_grad()
#     def _update_gsnr_stats(self):
#         eps = 1e-8
#         for group in self.param_groups:

#             for p in group["params"]:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad
#                 if grad.is_sparse:
#                     continue

#                 state = self.state[p]
#                 raw_gsnr = self._compute_tensor_gsnr(grad, p, eps)

#                 state["gsnr"] = raw_gsnr.clone()

#     @torch.no_grad()
#     def heavy_tail_noise_like(self, x, alpha=3.0, complex=False, eps=1e-8, clip=10.0):
#         # avoid u → 0 explosion
#         u = torch.rand_like(x)

#         base = (u.pow(-1.0 / alpha) - 1.0)

#         if complex:
#             # proper complex direction (unit magnitude)
#             real = torch.randn_like(x.real)
#             imag = torch.randn_like(x.real)
#             direction = torch.complex(real, imag)
#             direction = direction / (direction.abs() + eps)
#         else:
#             direction = torch.sign(torch.randn_like(x))

#         noise = direction * base

#         # CLIP heavy tail (VERY IMPORTANT)
#         if complex:
#             mag = noise.abs()
#             scale = torch.clamp(mag, max=clip) / (mag + eps)
#             noise = noise * scale
#         else:
#             noise = torch.clamp(noise, -clip, clip)

#         return noise
    
#     def newtonschulz5(self, G, steps=5, eps=1e-7):
#         assert G.ndim == 2
#         a, b, c = (3.4445, -4.7750, 2.0315)
#         X = G.bfloat16()
#         X /= (X.norm() + eps)
#         if G.size(0) > G.size(1):
#             X = X.T
#         for _ in range(steps):
#             A = X @ X.T
#             B = b * A + c * A @ A
#             X = a * X + B @ X
#         if G.size(0) > G.size(1):
#             X = X.T
#         return X


#     @torch.no_grad()
#     def cosine_sim(self, a, b, eps=1e-8):
#         a_flat = a.reshape(-1)
#         b_flat = b.reshape(-1)
#         return torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + eps)

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         self._global_step += 1

#         if self._global_step <= 1:
#             self._update_gsnr_stats()

#         for group in self.param_groups:
#             lr = group["lr"]
#             momentum = group["momentum"]
#             overshoot = group["overshoot"]
#             weight_decay = group["weight_decay"]
#             dampening = group["dampening"]
#             nesterov = group["nesterov"]
#             rms_beta = group["rms_beta"]
#             eps = group["eps"]

#             for p in group["params"]:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad
#                 state = self.state[p]
#                 gsnr = state.get("gsnr", torch.tensor(1.0))
#                 grad = grad * gsnr

            
#                 # ---------------------------
#                 # 1. Weight decay (decoupled style)
#                 # ---------------------------
#                 if weight_decay != 0:
#                     p.add_(p, alpha=-lr * weight_decay)



#                 # geometry preconditioning
#                 if grad.ndim == 2 and min(grad.shape) >= 64:
#                     grad = self.newtonschulz5(grad)
#                 elif grad.ndim == 4:
#                     g_original_shape = grad.shape
#                     out_c = grad.shape[0]
#                     grad_new = grad.view(out_c, -1)
#                     grad_new = self.newtonschulz5(grad_new)
#                     grad = grad_new.view(g_original_shape)
#                 grad = grad.float()


#                 # ---------------------------
#                 # 2. RMS Preconditioning
#                 # ---------------------------
#                 if "exp_avg_sq" not in state:
#                         state["exp_avg_sq"] = torch.zeros_like(p)

#                 exp_avg_sq = state["exp_avg_sq"]
#                 exp_avg_sq.mul_(rms_beta).addcmul_(grad, grad, value=1.0 - rms_beta)

#                 bias_correction2 = 1.0 - (rms_beta ** self._global_step)
#                 denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
#                 g = grad / denom



#                 # noise injectin
#                 noise_decay = max((self._noise_decay_steps - self._global_step) / self._noise_decay_steps, 0.1)
#                 if p.ndim == 4:
#                     Gf = torch.fft.rfft2(g, norm='ortho')

#                     noise = self.heavy_tail_noise_like(Gf, complex=True)
#                     noise = noise / (noise.abs().mean() + 1e-8)

#                     Gf_noisy = Gf + noise * self._noise_scale * gsnr * noise_decay
#                     g = torch.fft.irfft2(Gf_noisy, s=g.shape[-2:], norm='ortho')
#                 else:
#                     noise = self.heavy_tail_noise_like(g)
#                     noise = noise / (noise.abs().mean() + 1e-8)

#                     g = g + noise * self._noise_scale * gsnr * noise_decay


#                 # ---------------------------
#                 # 3. Momentum buffer
#                 # ---------------------------
#                 if momentum != 0:
#                     if "momentum_buffer" not in state:
#                         buf = state["momentum_buffer"] = g.clone()
#                     else:
#                         buf = state["momentum_buffer"]
#                         buf.mul_(momentum).add_(g, alpha=1 - dampening)

#                     # ---------------------------
#                     # 4. Nesterov
#                     # ---------------------------
#                     if nesterov:
#                         m = g + momentum * buf
#                     else:
#                         m = buf
#                 else:
#                     m = g



#                 # Measure alignmnet
#                 align = self.cosine_sim(g, m).clamp(-1, 1)
#                 align_gate = align / (1.0 + align)

#                 # ---------------------------
#                 # 5. Overshoot update
#                 # ---------------------------
#                 if overshoot != 0 and momentum != 0:
#                     effective_overshoot = overshoot * align_gate
#                     # same structure as paper, but using m (preconditioned)
#                     gc = -lr * (effective_overshoot) / momentum
#                     mc = -lr * (effective_overshoot - (effective_overshoot / momentum) + 1)

#                     p.add_(g, alpha=gc)
#                     p.add_(m, alpha=mc)
#                 else:
#                     p.add_(m, alpha=-lr)
                
#         return loss



