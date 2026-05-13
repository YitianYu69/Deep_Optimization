import torch
import torch.nn as nn
import torch.cuda as cuda
from torch.utils.data import DataLoader


class Graph:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        opt: torch.optim.Optimizer,
        data_loader: DataLoader,
        stream: torch.cuda.Stream,
        mode: str,
        num_of_graph: int,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.opt = opt
        self.data_loader = data_loader
        self.mode = mode
        self.num_of_graph = num_of_graph
        self.device = device


        # Streams & pools
        if stream is None:
            self.compute_stream = cuda.Stream()
        else:
            self.compute_stream = stream
        # self.cuda_graph_pool_handle = [None] * num_of_graph
        self.cuda_graph_pool_handle = cuda.graph_pool_handle()
        self.graphs = [None] * num_of_graph  # FIXED

        # Create static input placeholders
        x, y = next(iter(data_loader))
        self.static_x = torch.empty_like(x, device=device)
        self.static_y = torch.empty_like(y, device=device)

        prepare_theta(self.model)


    def init_graph(self):
        for i in range(self.num_of_graph):
            self.graphs[i] = cuda.CUDAGraph()
            # self.cuda_graph_pool_handle[i] = cuda.graph_pool_handle()




    def warmup(self, empty=False):

        self.compute_stream.wait_stream(cuda.current_stream())
        cuda.synchronize()
        x, y = next(iter(self.data_loader))
        # Warmup 2 (compute stream)
        with cuda.stream(self.compute_stream):
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                self.static_x.copy_(x.to('cuda'))
                self.static_y.copy_(y.to('cuda'))
                
                inject_noise(self.model)
                logits = self.model(self.static_x)
                loss = self.criterion(logits, self.static_y)
                loss.backward()
                denoise(self.model)
                self.opt.step()
                self.opt.zero_grad(set_to_none=False)



        torch.cuda.current_stream().wait_stream(self.compute_stream)
        torch.cuda.synchronize()

        if empty:
            torch.cuda.empty_cache()



    def capture_cuda_graph_qdrop(self):
        self.init_graph()
        self.warmup(empty=True)
        self.warmup()


        for i in range(self.num_of_graph):

            g = self.graphs[i]


            torch.cuda.current_stream().wait_stream(self.compute_stream)
            torch.cuda.synchronize()
            self.opt.zero_grad(set_to_none=False)
            

            with cuda.stream(self.compute_stream):
                with cuda.graph(g):
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):

                        inject_noise(self.model)
                        logits = self.model(self.static_x)
                        loss = self.criterion(logits, self.static_y)
                        loss.backward()
                        denoise(self.model)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.opt.step()

            cuda.synchronize()

        return (
            self.graphs,
            self.compute_stream,
            self.static_x,
            self.static_y,
            # self.static_logits,
            logits,
            loss
        )
    

    
def prepare_theta(model):
    for m in model.modules():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            
            if not isinstance(m, nn.BatchNorm2d):
                theta = torch.empty_like(p)
                p._theta = theta

@torch.no_grad
def inject_noise(model, alpha=0.0375):
    for p in model.parameters():
        if hasattr(p, "_theta"):
            p._theta.uniform_(-alpha, alpha)

            l2_p = p.pow(2).sum().sqrt()
            l2_theta = p._theta.pow(2).sum().sqrt()

            p._noise = alpha * (l2_p / l2_theta) * p._theta
            p.add_(p._noise)
            p._denoise = True

# @torch.no_grad
# def inject_noise(model, target_spectrum, alpha=0.0375):
#     for p in model.parameters():
#         if hasattr(p, "_theta"):

#             p._theta.copy_(
#                 noise_from_radial_spectrum_like(p, target_spectrum)
#             )

#             l2_p = p.norm()
#             l2_theta = p._theta.norm() + 1e-12

#             p._noise = alpha * (l2_p / l2_theta) * p._theta
#             p.add_(p._noise)
#             p._denoise = True

@torch.no_grad
def denoise(model):
    for p in model.parameters():
        if hasattr(p, '_denoise'):
            p.sub_(p._noise)

            del p._denoise
            del p._noise

def noise_from_radial_spectrum_like(
    p: torch.Tensor,
    target_spectrum: torch.Tensor,   # [num_bins], normalized
    eps=1e-8
):
    """
    Generate noise matching a pre-computed radial spectrum.
    No frequency computation from p or noise.
    """

    if p.ndim < 2:
        return torch.randn_like(p)

    device = p.device
    *rest, H, W = p.shape
    num_bins = target_spectrum.numel()

    # random complex spectrum (random phase)
    real = torch.randn(*rest, H, W, device=device)
    imag = torch.randn(*rest, H, W, device=device)
    F = torch.complex(real, imag)

    F = torch.fft.fftshift(F, dim=(-2, -1))

    # radial bin grid (STATIC, no spectrum computation)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = (r / r.max() * (num_bins - 1)).long()
    r = torch.clamp(r, 0, num_bins - 1)

    # apply target spectrum as magnitude
    mag = torch.sqrt(target_spectrum[r] + eps)
    F = F * mag

    # inverse FFT
    noise = torch.fft.ifft2(
        torch.fft.ifftshift(F, dim=(-2, -1)),
        dim=(-2, -1)
    ).real

    return noise
        

# def add_smoothout_noise(model, gamma=0.01):
#     for p in model.parameters():
#         if not p.requires_grad:
#             continue
#         # scale noise by parameter norm (AdaSmoothOut-ish)
#         rms = p.detach().pow(2).mean().sqrt()
#         sigma = gamma * rms
#         eps = torch.empty_like(p).uniform_(-sigma, sigma)
#         p.add_(eps)
#         p._smoothout_eps = eps  # stash


# def remove_smoothout_noise(model):
#     for p in model.parameters():
#         if hasattr(p, "_smoothout_eps"):
#             p.sub_(p._smoothout_eps)
#             del p._smoothout_eps

