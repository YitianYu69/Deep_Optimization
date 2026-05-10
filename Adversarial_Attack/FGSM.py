import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

import Deep_Optimization.Activation_Compression.modules.layers as layers

@torch.enable_grad()
def FGSM_attack(model, criterion, images, labels, eps=(8/255,),
                both=False, random_start=False,
                random_eps=(8/255,), alpha=(10/255,), mu=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                target_top2=False,
                LI=True, num_class=10, min_mask_ratio=0.1, max_mask_ratio=0.5,
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


    if LI:
        li_images = LIET_attack(model, criterion, images, labels,
                           eps=eps, num_class=num_class,
                           clamp_min=clamp_min, clamp_max=clamp_max,
                           min_mask_ratio=min_mask_ratio, max_mask_ratio=max_mask_ratio)



    if both:
        clean_images = images.detach().to(device)
        if LI:
            attack_images = li_images.detach().requires_grad_(True)
        else:
            attack_images = images.detach().to(device).requires_grad_(True)

        labels = labels.to(device)

        delta = torch.zeros_like(clean_images)
        for j in range(len(random_eps)):
            delta[:, j, :, :].uniform_(-random_eps[j][0][0].item(), random_eps[j][0][0].item())
        delta = torch.clamp(delta, clamp_min - clean_images, clamp_max - clean_images)
        delta = delta.detach().requires_grad_(True)

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(attack_images)

            if target_top2:
                top2 = logits.topk(2, dim=1).indices
                target = top2[:, 1]
                loss = -criterion(logits, target.detach())
            else:
                loss = criterion(logits, labels)

            clean_images_grad = torch.autograd.grad(
                loss, attack_images,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False
            )[0]

            model.zero_grad()

            rs_logits = model(attack_images + delta)

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
            fgsm_small_images = attack_images + eps[0] * clean_images_grad.sign()
            delta_small = torch.max(torch.min(fgsm_small_images - clean_images, eps[0]), -eps[0])
            fgsm_small_images = torch.clamp(clean_images + delta_small, clamp_min, clamp_max).detach()

            fgsm_large_images = attack_images + eps[1] * clean_images_grad.sign()
            delta_large = torch.max(torch.min(fgsm_large_images - clean_images, eps[1]), -eps[1])
            fgsm_large_images = torch.clamp(clean_images + delta_large, clamp_min, clamp_max).detach()

            fgsm_images = torch.cat([fgsm_small_images, fgsm_large_images], dim=0)
        else:
            fgsm_images = attack_images + eps * clean_images_grad.sign()
            delta = torch.max(torch.min(fgsm_images - clean_images, eps), -eps)
            fgsm_images = torch.clamp(clean_images + delta, clamp_min, clamp_max).detach()


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
        clean_images = images.detach().to(device)
        if LI:
            attack_images = li_images.detach().requires_grad_(True)
        else:
            attack_images = images.detach().to(device).requires_grad_(True)

        labels = labels.to(device)

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(attack_images)

            target = labels
            if target_top2:
                top2 = logits.topk(2, dim=1).indices
                target = top2[:, 1]
                loss = -criterion(logits, target.detach())
            else:
                loss = criterion(logits, target.detach())

            clean_images_grad = torch.autograd.grad(
                loss,
                attack_images,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False
            )[0]

        model.zero_grad()

        if isinstance(eps, tuple):
            fgsm_small_images = attack_images + eps[0] * clean_images_grad.sign()
            delta_small = torch.max(torch.min(fgsm_small_images - clean_images, eps[0]), -eps[0])
            fgsm_small_images = torch.clamp(clean_images + delta_small, clamp_min, clamp_max).detach()

            fgsm_large_images = attack_images + eps[1] * clean_images_grad.sign()
            delta_large = torch.max(torch.min(fgsm_large_images - clean_images, eps[1]), -eps[1])
            fgsm_large_images = torch.clamp(clean_images + delta_large, clamp_min, clamp_max).detach()

            fgsm_images = torch.cat([fgsm_small_images, fgsm_large_images], dim=0)
        else:
            fgsm_images = attack_images + eps * clean_images_grad.sign()
            delta = torch.max(torch.min(fgsm_images - clean_images, eps), -eps)
            fgsm_images = torch.clamp(clean_images + delta, clamp_min, clamp_max).detach()

        if was_training:
            model.train()
        return (fgsm_images * std + mu)
    
    else:
        clean_images = images.detach().to(device)
        if LI:
            attack_images = li_images.detach().requires_grad_(True)
        else:
            attack_images = images.detach().to(device).requires_grad_(True)

        labels = labels.to(device)

        delta = torch.zeros_like(clean_images)
        for j in range(len(random_eps)):
            delta[:, j, :, :].uniform_(-random_eps[j][0][0].item(), random_eps[j][0][0].item())

        delta = torch.clamp(delta, clamp_min - clean_images, clamp_max - clean_images)
        delta.requires_grad_(True)

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(attack_images + delta)

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
def LIET_attack(model, cri, data, labels, eps,
                clamp_min, clamp_max, num_class=10,
                min_mask_ratio=0.1, max_mask_ratio=0.5,
                mu=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                device='cuda'):
    
    was_training = model.training
    model.eval()

    mu = torch.tensor(mu, device=device).view(3, 1, 1)
    std = torch.tensor(std, device=device).view(3, 1, 1)

    image_size = data.shape[3]

    gray = torch.full(
        (num_class, 3, image_size, image_size),
        0.5,
        device=device
    )
    gray = (gray - mu) / std
    gray.requires_grad_(True)
    valid_labels = torch.arange(num_class, device=device)

    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(gray)
        loss = cri(logits, valid_labels)

        grad = torch.autograd.grad(
            loss,
            gray,
            grad_outputs=None,
            retain_graph=False,
            create_graph=False
        )[0]
    
    LI = eps * grad.sign().detach()
    model.zero_grad()

    delta = LI[labels].clone()

    uniform_noise = torch.empty_like(data).uniform_(-1.0, 1.0) * eps

    ratio_mask = torch.rand(1).item() * (max_mask_ratio - min_mask_ratio) + min_mask_ratio
    mask = torch.rand_like(data) < ratio_mask

    delta[mask] = uniform_noise[mask]

    sign = torch.where(
        torch.rand((data.shape[0], 1, 1, 1), device=device) < 0.5,
        1.0,
        -1.0
    )

    data = data + sign * delta
    data = torch.clamp(data, clamp_min, clamp_max).detach()

    if was_training:
        model.train()
    return data




@torch.enable_grad()
def PGD_attack(model, cri, data, labels, num_iters=(7,), random_eps=8/255, alpha=2/255,
               mu=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
               target_top2=False,
               device='cuda'):
    was_training = model.training
    model.eval()

    mu  = torch.tensor(mu, device=device).view(3,1,1)
    std = torch.tensor(std, device=device).view(3,1,1)

    random_eps = random_eps / std
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
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(data)
        top2 = logits.topk(2, dim=1).indices
        target = top2[:, 1]
    else:
        target = labels


    for _ in range(num_iters):
        data = data.detach().requires_grad_(True)

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
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




def squared_l2_norm(x):
    flattened = x.reshape(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


@torch.enable_grad()
def TRADES_attack(model, data, labels=None,
                  num_iters=10,
                  epsilon=8/255,
                  random_eps=0.003,
                  alpha=2/255,
                  mu=(0.5, 0.5, 0.5),
                  std=(0.5, 0.5, 0.5),
                  distance='l_inf',
                  device='cuda',
                  use_amp=True):
    """
    TRADES adversarial sample generation only.

    Important:
        data is already normalized.
        model only receives normalized inputs.
        return value is also normalized.

    For l_inf:
        epsilon, alpha, random_eps are interpreted in pixel space,
        then converted to normalized space channel-wise.

    For l_2:
        epsilon is interpreted as normalized-space L2 radius,
        following the official TRADES optimizer_delta style.
    """

    was_training = model.training
    model.eval()

    device = torch.device(device)
    data = data.to(device)

    batch_size = data.shape[0]

    mu = torch.tensor(mu, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    clamp_min = (0.0 - mu) / std
    clamp_max = (1.0 - mu) / std

    ori_data = data.detach()

    criterion_kl = nn.KLDivLoss(reduction='sum')

    # --------------------------------------------------
    # Clean prediction target: model(x_clean)
    # --------------------------------------------------
    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp and device.type == 'cuda'
        ):
            clean_logits = model(ori_data)

        clean_prob = F.softmax(clean_logits.float(), dim=1).detach()

    # ==================================================
    # L_inf TRADES attack
    # ==================================================
    if distance == 'l_inf':
        epsilon_norm = epsilon / std
        alpha_norm = alpha / std
        random_eps_norm = random_eps / std

        # Official TRADES starts from small Gaussian noise
        data_adv = ori_data + random_eps_norm * torch.randn_like(ori_data)
        data_adv = torch.max(torch.min(data_adv, clamp_max), clamp_min)

        # Project random start into L_inf epsilon ball
        eta = torch.max(
            torch.min(data_adv - ori_data, epsilon_norm),
            -epsilon_norm
        )
        data_adv = torch.max(torch.min(ori_data + eta, clamp_max), clamp_min).detach()

        for _ in range(num_iters):
            data_adv = data_adv.detach().requires_grad_(True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp and device.type == 'cuda'
            ):
                adv_logits = model(data_adv)

                loss_kl = criterion_kl(
                    F.log_softmax(adv_logits.float(), dim=1),
                    clean_prob
                )

                # loss_kl += F.cross_entropy(adv_logits, labels)

            grad = torch.autograd.grad(
                loss_kl,
                [data_adv],
                grad_outputs=None,
                retain_graph=False,
                create_graph=False
            )[0]

            # Official TRADES: gradient ascent on KL
            data_adv = data_adv.detach() + alpha_norm * torch.sign(grad.detach())

            # L_inf projection
            eta = torch.max(
                torch.min(data_adv - ori_data, epsilon_norm),
                -epsilon_norm
            )

            data_adv = torch.max(torch.min(ori_data + eta, clamp_max), clamp_min).detach()

    # ==================================================
    # L2 TRADES attack
    # ==================================================
    elif distance == 'l_2':
        # Official TRADES-style delta
        delta = random_eps * torch.randn_like(ori_data).detach()
        delta = torch.autograd.Variable(delta.data, requires_grad=True)

        # Official TRADES-style optimizer on delta
        optimizer_delta = optim.SGD(
            [delta],
            lr=epsilon / num_iters * 2
        )

        for _ in range(num_iters):
            adv = ori_data + delta
            adv = torch.max(torch.min(adv, clamp_max), clamp_min)

            optimizer_delta.zero_grad()

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp and device.type == 'cuda'
            ):
                adv_logits = model(adv)

                # Official TRADES minimizes -KL for delta,
                # which is equivalent to maximizing KL.
                loss = (-1.0) * criterion_kl(
                    F.log_softmax(adv_logits.float(), dim=1),
                    clean_prob
                )

            loss.backward()

            # --------------------------------------------------
            # Official TRADES: renorm delta gradient
            # --------------------------------------------------
            grad_norms = delta.grad.reshape(batch_size, -1).norm(p=2, dim=1)

            zero_mask = grad_norms == 0
            if zero_mask.any():
                delta.grad[zero_mask] = torch.randn_like(delta.grad[zero_mask])
                grad_norms = delta.grad.reshape(batch_size, -1).norm(p=2, dim=1)

            delta.grad.div_(grad_norms.view(-1, 1, 1, 1).clamp_min(1e-12))

            optimizer_delta.step()

            # --------------------------------------------------
            # Official TRADES projection style, but clamp in
            # normalized image space instead of [0, 1].
            # --------------------------------------------------
            delta.data.add_(ori_data)

            delta.data.copy_(
                torch.max(torch.min(delta.data, clamp_max), clamp_min)
            )

            delta.data.sub_(ori_data)

            # L2 projection around clean input
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)

        data_adv = ori_data + delta.detach()
        data_adv = torch.max(torch.min(data_adv, clamp_max), clamp_min).detach()

    else:
        raise ValueError(f"Unsupported distance: {distance}. Use 'l_inf' or 'l_2'.")

    if was_training:
        model.train()

    return (data_adv.detach() * std) + mu