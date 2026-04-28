import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import Deep_Optimization.Activation_Compression.modules.layers as layers

@torch.enable_grad()
def FGSM_attack(model, criterion, images, labels, eps=(8/255,),
                both=False, random_start=False,
                random_eps=(8/255,), alpha=(10/255,), mu=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                target_top2=False,
                device='cuda'):
    
    was_training = model.training
    model.eval()

    mu = torch.tensor(mu, device=device).view(3, 1, 1)
    std = torch.tensor(std, device=device).view(3, 1, 1)

    clamp_min = (0.0 - mu) / std
    clamp_max = (1.0 - mu) / std

    if len(eps) == 2:
        eps = tuple(sorted(eps))

        eps1 = eps[0] / std
        eps2 = eps[1] / std
        eps = (eps1, eps2)
    else:
        eps = eps[0] / std

    if len(alpha) == 2:
        idx = sorted(range(len(alpha)), key=lambda i : alpha[i])

        alpha = tuple(alpha[i] for i in idx)
        alpha1 = alpha[0] / std
        alpha2 = alpha[1] / std
        alpha = (alpha1, alpha2)

        if len(random_eps) == 2:
            random_eps = (random_eps[0] / std, random_eps[1] / std)
        elif len(random_eps) == 1:
            random_eps = random_eps[0] / std
        else:
            raise ValueError("random_eps's length cannot be greater than 2")
    else:
        assert len(random_eps) == 1, "random_eps's length must be 1"

        alpha = alpha[0] / std
        random_eps = random_eps[0] / std



    if both:
        clean_images = images.detach().to(device).requires_grad_(True)
        labels = labels.to(device)

        delta = torch.zeros_like(clean_images)
        for j in range(len(random_eps)):
            delta[:, j, :, :].uniform_(-random_eps[j][0][0].item(), random_eps[j][0][0].item())
        delta = torch.clamp(delta, clamp_min - clean_images, clamp_max - clean_images)
        delta = delta.detach().requires_grad_(True)

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(clean_images)

            if target_top2:
                top2 = logits.topk(2, dim=1).indices
                target = top2[:, 1]
                loss = -criterion(logits, target.detach())
            else:
                loss = criterion(logits, labels)

            clean_images_grad = torch.autograd.grad(
                loss, clean_images,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False
            )[0]

            model.zero_grad()

            rs_logits = model(clean_images + delta)

            if target_top2:
                top2 = rs_logits.topk(2, dim=1).indices
                target = top2[:, 1]
                rs_loss = -criterion(rs_logits, target.detach())
            else:
                rs_loss = criterion(rs_logits, labels)

            delta_grad = torch.autograd.grad(
                rs_loss, delta,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False
            )[0]

            model.zero_grad()

        if isinstance(eps, tuple):
            fgsm_small_images = clean_images + eps[0] * clean_images_grad.sign()
            fgsm_small_images = fgsm_small_images.clamp(clamp_min, clamp_max).detach()

            fgsm_large_images = clean_images + eps[1] * clean_images_grad.sign()
            fgsm_large_images = fgsm_large_images.clamp(clamp_min, clamp_max).detach()

            fgsm_images = torch.cat([fgsm_small_images, fgsm_large_images], dim=0)
        else:
            fgsm_images = clean_images + eps * clean_images_grad.sign()
            fgsm_images = fgsm_images.clamp(clamp_min, clamp_max).detach()


        if isinstance(alpha, tuple):
            delta_small = delta.detach() + alpha[0] * delta_grad.sign()
            delta_small = torch.max(torch.min(delta_small, random_eps), -random_eps)
            delta_small = torch.clamp(delta_small, clamp_min - clean_images, clamp_max - clean_images)
            fgsm_rs_small_images = (clean_images + delta_small).detach()

            delta_large = delta.detach() + alpha[1] * delta_grad.sign()
            delta_large = torch.max(torch.min(delta_large, random_eps), -random_eps)
            delta_large = torch.clamp(delta_large, clamp_min - clean_images, clamp_max - clean_images)
            fgsm_rs_large_images = (clean_images + delta_large).detach()

            fgsm_rs_images = torch.cat([fgsm_rs_small_images, fgsm_rs_large_images], dim=0)
        else:
            delta = delta.detach() + alpha * delta_grad.sign()
            delta = torch.max(torch.min(delta, random_eps), -random_eps)
            delta = torch.clamp(delta, clamp_min - clean_images, clamp_max - clean_images)
            fgsm_rs_images = (clean_images + delta).detach()

        if was_training:
            model.train()

        return [(fgsm_images * std + mu), (fgsm_rs_images * std + mu)]
    
    if not random_start:
        clean_images = images.detach().to(device).requires_grad_(True)
        labels = labels.to(device)

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(clean_images)

            target = labels
            if target_top2:
                top2 = logits.topk(2, dim=1).indices
                target = top2[:, 1]
                loss = -criterion(logits, target.detach())
            else:
                loss = criterion(logits, target.detach())

            clean_images_grad = torch.autograd.grad(
                loss,
                clean_images,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False
            )[0]

        model.zero_grad()

        if isinstance(eps, tuple):
            fgsm_small_images = clean_images + eps[0] * clean_images_grad.sign()
            fgsm_small_images = fgsm_small_images.clamp(clamp_min, clamp_max).detach()

            fgsm_large_images = clean_images + eps[1] * clean_images_grad.sign()
            fgsm_large_images = fgsm_large_images.clamp(clamp_min, clamp_max).detach()

            fgsm_images = torch.cat([fgsm_small_images, fgsm_large_images], dim=0)
        else:
            fgsm_images = clean_images + eps * clean_images_grad.sign()
            fgsm_images = fgsm_images.clamp(clamp_min, clamp_max).detach()

        if was_training:
            model.train()
        return (fgsm_images * std + mu)
    
    else:
        clean_images = images.detach().to(device)
        labels = labels.to(device)

        delta = torch.zeros_like(clean_images)
        for j in range(len(random_eps)):
            delta[:, j, :, :].uniform_(-random_eps[j][0][0].item(), random_eps[j][0][0].item())

        delta = torch.clamp(delta, clamp_min - clean_images, clamp_max - clean_images)
        delta.requires_grad_(True)

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=False):
            logits = model(clean_images + delta)

            target = labels
            if target_top2:
                top2 = logits.topk(2, dim=1).indices
                target = top2[:, 1]
                loss = -criterion(logits, target.detach())
            else:
                loss = criterion(logits, target.detach())

            delta_grad = torch.autograd.grad(
                loss,
                delta,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False
            )[0]
        model.zero_grad()

        if isinstance(alpha, tuple):
            delta_small = delta.detach() + alpha[0] * delta_grad.sign()
            delta_small = torch.max(torch.min(delta_small, random_eps), -random_eps)
            delta_small = torch.clamp(delta_small, clamp_min - clean_images, clamp_max - clean_images)
            fgsm_rs_small_images = (clean_images + delta_small).detach()

            delta_large = delta.detach() + alpha[1] * delta_grad.sign()
            delta_large = torch.max(torch.min(delta_large, random_eps), -random_eps)
            delta_large = torch.clamp(delta_large, clamp_min - clean_images, clamp_max - clean_images)
            fgsm_rs_large_images = (clean_images + delta_large).detach()

            fgsm_rs_images = torch.cat([fgsm_rs_small_images, fgsm_rs_large_images], dim=0)
        else:
            delta = delta.detach() + alpha * delta_grad.sign()
            delta = torch.max(torch.min(delta, random_eps), -random_eps)
            delta = torch.clamp(delta, clamp_min - clean_images, clamp_max - clean_images)
            fgsm_rs_images = (clean_images + delta).detach()


        if was_training:
            model.train()

        return (fgsm_rs_images * std + mu)



@torch.enable_grad()
def PGD_attack(model, cri, data, labels, num_iters=7, random_eps=8/255, alpha=2/255,
               mu=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
               target_top2=False,
               device='cuda'):
    was_training = model.training
    model.eval()

    mu  = torch.tensor(mu, device=device).view(3,1,1)
    std = torch.tensor(std, device=device).view(3,1,1)

    random_eps = random_eps / std
    alpha = alpha / std
    clamp_min = (0.0 - mu) / std
    clamp_max = (1.0 - mu) / std


    data = data.to(device)
    labels = labels.to(device)

    ori_data = data.detach()

    delta = torch.zeros_like(data)
    for j in range(len(random_eps)):
        delta[:, j, :, :].uniform_(-random_eps[j][0][0].item(), random_eps[j][0][0].item())

    data = ori_data + delta
    data = torch.clamp(data, clamp_min, clamp_max)

    if target_top2:
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=False):
                logits = model(data)
        top2 = logits.topk(2, dim=1).indices
        target = top2[:, 1]
    else:
        target = labels


    for _ in range(num_iters):
        data = data.detach().requires_grad_(True)

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=False):
            logits = model(data)

            if target_top2:
                loss = -cri(logits, target.detach())
            else:
                loss = cri(logits, target.detach())

            grad = torch.autograd.grad(
                loss,
                data,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False
                )[0]

        adv_data = data + alpha * grad.sign()

        # projection
        eta = torch.clamp(adv_data - ori_data, -random_eps, random_eps)
        data = torch.clamp(ori_data + eta, clamp_min, clamp_max)

    if was_training:
        model.train()

    return (data * std + mu).detach()