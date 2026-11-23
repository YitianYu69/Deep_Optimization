"""Executable factory contract: a small classifier with optional pinned input transfer.

Replace build() with your own model/loader/loss, keeping the return contract.
No pretrained weights, datasets, downloads or additional dependencies are used.
"""

import os


DEFAULT_CIFAR10_ROOT = os.environ.get("DEEPOPT_CIFAR10_ROOT", "data")


def build_attention_block(device, batch_size=512, sequence_length=197, width=768, heads=12,
                          source_layout="interleaved"):
    """Complete trainable projection/attention/projection boundary, no downloads."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    if source_layout not in ("interleaved", "bshd", "bhsd") or width % heads:
        raise ValueError("Valid source_layout and width divisible by heads are required")

    class AttentionBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(width, 3 * width)
            self.proj = nn.Linear(width, width)

        def forward(self, x):
            b, t, _ = x.shape
            values = self.qkv(x).reshape(b, t, 3, heads, width // heads).unbind(2)
            if source_layout == "bshd":
                values = tuple(v.contiguous() for v in values)
            values = tuple(v.transpose(1, 2) for v in values)
            if source_layout == "bhsd":
                values = tuple(v.contiguous() for v in values)
            attended = F.scaled_dot_product_attention(*values)
            return self.proj(attended.transpose(1, 2).reshape(b, t, width))

    model = AttentionBlock().to(device)
    x = torch.randn(batch_size, sequence_length, width, device=device, requires_grad=True)
    return {"model": model, "inputs": (x,), "kwargs": {},
            "benchmark_metadata": {"consumer": "qkv_linear_attention_out_linear", "source_layout": source_layout,
                                   "weights": None, "dataset_epoch_measured": False}}


def build(device, batch_size=2, sequence_length=16, width=64, heads=4, layers=2,
          classes=10, input_mode="resident", batches=8):
    import torch
    from torch import nn
    from torch.nn import functional as F

    if input_mode not in ("resident", "pinned"):
        raise ValueError("input_mode must be resident or pinned")

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(width)
            self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

        def forward(self, x):
            h = self.norm1(x)
            x = x + self.attention(h, h, h, need_weights=False)[0]
            return x + self.mlp(self.norm2(x))

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(width, width)
            self.blocks = nn.Sequential(*(Block() for _ in range(layers)))
            self.head = nn.Linear(width, classes)

        def forward(self, x):
            return self.head(self.blocks(self.embedding(x)).mean(1))

    model = Classifier().to(device)
    samples = [(torch.randn(batch_size, sequence_length, width, dtype=torch.bfloat16),
                torch.randint(classes, (batch_size,))) for _ in range(batches)]
    if input_mode == "pinned" and device.type == "cuda":
        samples = [(x.pin_memory(), y.pin_memory()) for x, y in samples]
    state = {"index": 0, "target": samples[0][1].to(device)}
    resident = samples[0][0].to(device)

    def reset():
        state["index"] = 0
        state["target"] = samples[0][1].to(device)

    def fetch():
        x, y = samples[state["index"] % len(samples)]
        state["index"] += 1
        state["target"] = y.to(device, non_blocking=True)
        return (x.to(device, non_blocking=True),), {}

    result = {"model": model, "inputs": (resident,), "kwargs": {},
              "loss_fn": lambda output: F.cross_entropy(output.float(), state["target"]),
              "fingerprint_data": samples}
    if input_mode == "pinned":
        result.update(fetch=fetch, reset=reset)
    return result


def build_timm_vit(device, batch_size=512, image_size=224, classes=10,
                   model_name="vit_base_patch16_224", ema_decay=0.95):
    """Resident-input ViT factory matching the reported H100 workload; no downloads."""
    import torch
    from torch.nn import functional as F
    try:
        import timm
    except ImportError as error:
        from .workloads import UnsupportedWorkload
        raise UnsupportedWorkload("The ViT factory requires timm in the GPU environment") from error

    # timm CNN constructors do not accept the ViT-specific img_size keyword.
    model_options = {} if model_name in ("resnet18", "efficientnet_b0") else {"img_size": image_size}
    model = timm.create_model(model_name, pretrained=False, num_classes=classes, **model_options).to(device)
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    targets = torch.randint(classes, (batch_size,), device=device)
    result = {"model": model, "inputs": (images,), "kwargs": {},
              "loss_fn": lambda output: F.cross_entropy(output.float(), targets), "fingerprint_data": (targets,),
              "benchmark_metadata": {"input_source": "resident_synthetic", "model": model_name,
                  "library": "timm", "pretrained": False, "batch_size": batch_size,
                  "image_size": image_size, "dataset_epoch_measured": False}}
    if ema_decay is not None:
        from deepopt.engine.training.ema import ExponentialMovingAverage

        ema = ExponentialMovingAverage(model, ema_decay)
        initial = {**ema.state_dict(), "shadow": {name: value.cpu().clone() for name, value in ema.shadow.items()}}

        def reset():
            ema.load_state_dict(initial, completed_updates=0)

        result.update(after_step=lambda: ema.update(model), reset=reset)
    return result


def build_variable_mlp(device, batch_sizes=(16, 32, 31, 33), sequence_length=197, width=768):
    """Repeatable varying-shape eager allocator workload; no data downloads."""
    import torch
    from torch import nn

    if not batch_sizes or any(type(n) is not int or n < 1 for n in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    model = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)).to(device)
    samples = [torch.randn(n, sequence_length, width, device=device) for n in batch_sizes]
    state = {"index": 0}

    def fetch():
        value = samples[state["index"] % len(samples)]
        state["index"] += 1
        return (value,), {}

    def reset():
        state["index"] = 0

    return {"model": model, "inputs": (samples[0],), "kwargs": {}, "fetch": fetch, "reset": reset,
            "fingerprint_data": samples}


def build_norm_boundary(device, batch_size=512, sequence_length=197, width=768):
    """FP32 residual input; prepared LayerNorm→Linear gives a real FP8 consumer edge."""
    import torch
    from torch import nn

    model = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width)).to(device)
    source = torch.randn(batch_size, sequence_length, width, device=device, requires_grad=True)
    return {"model": model, "inputs": (source,), "kwargs": {}}


def build_torchvision_cnn(device, model_name="resnet18", batch_size=128, image_size=224,
                          classes=10, ema_decay=0.95, memory_format="contiguous", cifar_stem=False):
    """Randomly initialized torchvision CNNs; never request or download weights."""
    import torch
    from torch.nn import functional as F

    if model_name not in ("resnet18", "efficientnet_b0"):
        raise ValueError("model_name must be resnet18 or efficientnet_b0")
    if any(type(n) is not int or n < 1 for n in (batch_size, image_size, classes)):
        raise ValueError("batch_size, image_size and classes must be positive integers")
    if memory_format not in ("contiguous","channels_last"):
        raise ValueError("memory_format must be contiguous or channels_last")
    if cifar_stem and model_name!="resnet18":
        raise ValueError("cifar_stem is supported only for ResNet18")
    try:
        from torchvision import models
    except ImportError as error:
        from .workloads import UnsupportedWorkload
        raise UnsupportedWorkload("CNN benchmarks require torchvision; no pretrained weights are used") from error

    model = getattr(models, model_name)(weights=None, num_classes=classes)
    if cifar_stem:
        model.conv1=torch.nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
        torch.nn.init.kaiming_normal_(model.conv1.weight,mode="fan_out",nonlinearity="relu")
        model.maxpool=torch.nn.Identity()
    fmt=torch.channels_last if memory_format=="channels_last" else torch.contiguous_format
    model=model.to(device=device,memory_format=fmt)
    images = torch.randn(batch_size, 3, image_size, image_size, device=device).to(memory_format=fmt)
    targets = torch.randint(classes, (batch_size,), device=device)
    result = {"model": model, "inputs": (images,), "kwargs": {},
              "loss_fn": lambda output: F.cross_entropy(output.float(), targets),
              "fingerprint_data": (targets,), "benchmark_metadata":{"input_source":"resident_synthetic",
                "model":model_name,"weights":None,"batch_size":batch_size,"image_size":image_size,
                "classes":classes,"memory_format":memory_format,"cifar_stem":cifar_stem,
                "dataset_epoch_measured":False}}
    if ema_decay is not None:
        from deepopt.engine.training.ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model, ema_decay)
        initial = {**ema.state_dict(), "shadow": {name: value.cpu().clone() for name, value in ema.shadow.items()}}
        result.update(after_step=lambda: ema.update(model),
                      reset=lambda: ema.load_state_dict(initial, completed_updates=0))
    return result


def build_local_cifar10(device, dataset_root=DEFAULT_CIFAR10_ROOT, model_name="efficientnet_b0", batch_size=512,
                        image_size=224, workers=4, drop_last=False, ema_decay=0.95,
                        memory_format="contiguous", cifar_stem=False, seed=42):
    """Real, local CIFAR10 training data; download=False, explicit preprocessing.

    For a full eager epoch use benchmark_cifar10_epoch.py. Matrix timings over
    this factory remain update timings, with loader/H2D work included.
    """
    import torch
    from torchvision import datasets,transforms
    from torch.utils.data import DataLoader
    from torch.nn import functional as F

    package=(build_timm_vit(device,batch_size,image_size,10,ema_decay=ema_decay) if model_name=="vit_base_patch16_224" else
             build_torchvision_cnn(device,model_name,batch_size,image_size,10,ema_decay,
                                   memory_format=memory_format,cifar_stem=cifar_stem))
    dataset=datasets.CIFAR10(dataset_root,train=True,download=False,
        transform=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),
                                     transforms.Normalize((.4914,.4822,.4465),(.247,.243,.261))]))
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=workers,
        pin_memory=device.type=="cuda",drop_last=drop_last,generator=generator,persistent_workers=workers>0)
    state={"iterator":None,"target":None,"examples":0,"updates":0}
    reset_model=package.get("reset")
    fmt=torch.channels_last if memory_format=="channels_last" else torch.contiguous_format
    def reset():
        generator.manual_seed(seed)
        state.update(iterator=None,target=None,examples=0,updates=0)
        if reset_model:reset_model()
    def fetch():
        if state["iterator"] is None:state["iterator"]=iter(loader)
        try:images,targets=next(state["iterator"])
        except StopIteration:
            state["iterator"]=iter(loader);images,targets=next(state["iterator"])
        state["target"]=targets.to(device,non_blocking=True)
        state["examples"]+=len(images);state["updates"]+=1
        return (images.to(device=device,memory_format=fmt,non_blocking=True),),{}
    package.update(fetch=fetch,reset=reset,loss_fn=lambda output:F.cross_entropy(output.float(),state["target"]),
        epoch_updates=len(loader),epoch_state=state,
        benchmark_metadata={"input_source":"local_CIFAR10","download":False,"split":"train","dataset_examples":len(dataset),
            "batch_size":batch_size,"image_size":image_size,"drop_last":drop_last,"workers":workers,
            "epoch_updates":len(loader),"preprocessing":"resize, tensor, CIFAR10 mean/std; no random augmentation",
            "memory_format":memory_format,"cifar_stem":cifar_stem})
    return package
