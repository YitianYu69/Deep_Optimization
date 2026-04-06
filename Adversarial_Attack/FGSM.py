import torch

def FGSM_attack(model, criterion, images, labels, eps=0.007, device='cpu'):
    images = images.detach().to(device).requires_grad_(True)
    labels = labels.to(device)

    p = torch.rand((1,), dtype=torch.float32, device=device)

    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(images)
        loss = criterion(logits, labels)

        images_grad = torch.autograd.grad(
            loss,
            images,
            grad_outputs=None,
            retain_graph=False,
            create_graph=False
        )[0]

        attack_images = images + eps * images_grad.sign()
        attack_images = attack_images.clamp(0, 1).detach()
        return attack_images, labels

