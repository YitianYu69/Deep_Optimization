import torch

def FGSM_attack(model, criterion, images, labels, eps=0.007, attack_p=0.5, device='cpu'):
    p = torch.rand((1,), dtype=torch.float32, device=model.device)

    if p > attack_p:

        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, labels)

            images_grad = torch.autograd(
                loss,
                images,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False
            )

            attack_images = images + eps * images_grad.sign()
            attack_images = attack_images.clamp(0, 1)
            return attack_images, labels
    else:
        return images, labels

