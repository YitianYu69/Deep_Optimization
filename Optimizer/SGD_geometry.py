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