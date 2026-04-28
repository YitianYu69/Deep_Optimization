import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"
)

import sys
sys.path.append('/home/hice1/yyu496/kaggle/CW')

import torch
from torch.utils.data import Subset, random_split
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.optim as optim
import torch.multiprocessing as mp

import timm

import torchvision.models as models
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder


from Deep_Optimization.Train.Trainer import Trainer
from Deep_Optimization.Train.data import get_dataloader
from Deep_Optimization.Train.utils_train import build_metrics, EMA
from Deep_Optimization.Train.utils_ddp import setup_ddp, get_ddp_meta, rank0, clean
from Deep_Optimization.Optimizer.SGD_geometry import SGD_NS_Overshoot, SGD_NS_Overshoot_Noise

import Deep_Optimization.Activation_Compression.modules.layers as layers
from Deep_Optimization.Activation_Compression.modules.normalization import normalization_layers as norm_layers
from Deep_Optimization.Activation_Compression.modules.normalization.norm_layer_utils import check_sync_bn
from Deep_Optimization.Initialization.Init import init

from Deep_Optimization.Train.log import get_logger

from robustbench.utils import load_model, clean_accuracy



logger = get_logger()

@torch.compile(fullgraph=True)
def focal_loss(logits, target, gamma=5.0, weight=None):
    ce = F.cross_entropy(
        logits,
        target,
        weight=weight,
        reduction="none"
    )
    pt = torch.exp(-ce)
    focal = (1.0 - pt).pow(gamma) * ce
    return focal.mean()


def main():
    local_rank, global_rank, world_size, device = setup_ddp()

    batch_size = 512
    warmup_epochs = 25
    num_epochs = 512

    g = torch.Generator()
    g.manual_seed(42)

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]


    def dataset_sample_for_split(dataset, train_fraction=0.8, generator=None):
        total_samples = len(dataset)
        
        train_samples = int(total_samples * train_fraction)
        val_samples = total_samples - train_samples
        return random_split(range(total_samples), [train_samples, val_samples], generator=generator)


    transform_train = v2.Compose([
        v2.ToImage(),                    
        v2.ToDtype(torch.float32, scale=True),

        v2.Resize((224, 224), antialias=True),

        v2.RandomHorizontalFlip(p=0.5),

        v2.RandomApply([
            v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )
        ], p=0.5),

        v2.RandAugment(num_ops=3, magnitude=9),

        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

        v2.RandomErasing(p=0.5, scale=(0.03, 0.33),
                        ratio=(0.3, 3.3), value=0),
    ])



    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),

        v2.Resize((224, 224), antialias=True),

        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


    checker_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),

        v2.Resize((224, 224), antialias=True),

        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    cifar10 = ImageFolder(
        root='/home/hice1/yyu496/scratch/data/cifar10_resized/train',
        transform=transform_train
    )
    test_cifar10 = ImageFolder(
        root='/home/hice1/yyu496/scratch/data/cifar10_resized/test',
        transform=checker_transform
    )
    clean_cifar10 = ImageFolder(
        root='/home/hice1/yyu496/scratch/data/cifar10_resized/train',
        transform=checker_transform
    )

    train_cifar10_indices, valid_cifar10_indices = dataset_sample_for_split(cifar10, train_fraction=0.8, generator=g)
    train_dataset_cifar10 = Subset(cifar10, train_cifar10_indices.indices)
    valid_dataset_cifar10 = Subset(clean_cifar10, valid_cifar10_indices.indices)
    test_dataset_cifar10 = test_cifar10


    train_dataloader_cifar10, valid_dataloader_cifar10, test_dataloader_cifar10 = get_dataloader(batch_size=batch_size, num_workers=15, drop_last=True,
                                                                                                train_dataset=train_dataset_cifar10,
                                                                                                valid_dataset=valid_dataset_cifar10,
                                                                                                test_dataset=test_dataset_cifar10,
                                                                                                ddp=True,
                                                                                                global_rank=global_rank,
                                                                                                world_size=world_size,
                                                                                                pin_memory_device=device)






    num_classes = 10
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model = timm.create_model('wide_resnet101_2', pretrained=False, num_classes=num_classes)
     
    # model = load_model('Wong2020Fast', norm='Linf')
    ema_model = EMA(model, decay=0.99, tau=0, device='cuda', kahan_compensation=True)


    # criterion = nn.CrossEntropyLoss()
    criterion = {"Train" : focal_loss,
                 'Valid' : nn.CrossEntropyLoss()}



    # optimizer = optim.AdamW
    # optimizer_kwargs = dict(
    #     lr=1e-4 * 8,
    #     fused=True,
    #     capturable=True
    # )
    optimizer = SGD_NS_Overshoot_Noise
    optimizer_kwargs = dict(
        lr = 1e-4 * 8,
        actual_bs=batch_size * 6,
        noise_decay_steps=20 * len(train_dataloader_cifar10),
        overshoot=5.0,
        layer_noise_beta=0.99,
        layer_noise_alpha=1.0,
        min_lr_scale=0.05,
        max_lr_scale=10.0,
        use_layerwise_noise=True,
    )

    # optimizer = SGD_NS_Overshoot
    # optimizer_kwargs = dict(
    #     lr = 1e-4 * 8,
    #     actual_bs=batch_size * 6,
    #     noise_decay_steps=20 * len(train_dataloader_cifar10),
    #     overshoot=5.0
    # )

    Adversarial_Attack_config = {
        'Attack_Type' : {'FGSM' : {'eps' : (2/255, 8/255)}, 
                         'FGSM_RS' : {'alpah' : (2/255, 10/255)}, 
                         'PGD' : {'steps' : 7, 'alpha' : 2/255, 'kl_weight' : 10.0}},
        'KL_temperature' : 1.0,
        'mu' : IMAGENET_MEAN,
        'std' : IMAGENET_STD
    }

    Trainer_config = {
        'L1_Sparse_Loss' : {'trust_ratio' : 0.007},
        'EMA_Proximal_Loss' : {"Start_Epoch" : 6, 'rho' : 5e-4},
        'EMA_Teacher' : {'Start_Epoch' : 6, 'full_logits' : False, 'clean_logits' : False},
        'Multi_View' : True,
        'SAM' : True,
    }

    metric_list = ['Accuracy', 'AUROC']
    metrics = build_metrics(metric_lists=metric_list, 
                            task='multiclass', 
                            num_classes=num_classes, 
                            average_type='micro', 
                            sync=True, 
                            device=device)

    DIVISION = None
    ACT_config = {
        "default_bits": 2,
        'analyze' : False,
        'auto_precision': None,
        'DIVISION' : DIVISION,
        'AVG_ALAM' : False,
        "AVG_ALAM_BTS" : 4,
        "group_size": 256,
        'batch_size' : batch_size,
        'fp8' : False,
        'depth_point_conv' : False,
        'rms_norm' : False,
        'SyncBatchNorm' : False
    }

    DDP_config = {
        'device_ids' : [local_rank],
        'static_graph' : True,
        'gradient_as_bucket_view' : True,
        'output_device' : local_rank,
        'broadcast_buffers' : False
    }






    trainer = Trainer(model=model,
                    DDP_config=DDP_config,
                    ACT_config=ACT_config,
                    Adversarial_Attack=Adversarial_Attack_config,
                    Trainer_config=Trainer_config,
                    dataloader=train_dataloader_cifar10, 
                    metrics=metrics, criterion=criterion, 
                    ema=ema_model,
                    optimizer_type=optimizer, optimizer_kwargs=optimizer_kwargs,
                    grad_norm_clip=False,
                    device=device)

    opt = trainer.get_Optimizer()
    warmup_scheduler = LinearLR(opt, start_factor=0.5, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(opt, T_max=num_epochs - warmup_epochs, eta_min=1e-5)
    scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])


    res = check_sync_bn(trainer.get_Engine())
    if rank0():
        if res:
            logger.info("DOSyncBN convert succeed!")
        else:
            logger.info("DOSyncBN convert failed!")

    mu = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

    init(trainer.get_Engine(),
        dataloader=train_dataloader_cifar10,

        alpha=0.05,
        gamma=0.05,
        phase_noise=0.05,

        orthogonal=False,
        zero_init=True,
        ID_init=False,

        attack=True,
        cri=criterion,
        mu=mu,
        std=std,

        device='cuda')



    def print_metrics(metrics, Train=True, Attack=False, rs=False, target_top2=False, PGD=False, steps=100, auto=False):
        pre_fx = "Train "
        if not Train:
            pre_fx = 'Valid '
        if not Train and Attack:
            pre_fx = 'Valid Attack '
        if not Train and Attack and rs:
            pre_fx = 'Valid Attack RS '
        if not Train and Attack and target_top2:
            pre_fx = 'Valid Target top2 Attack '
        if not Train and Attack and rs and target_top2:
            pre_fx = 'Valid Target top2 Attack RS '
        if not Train and Attack and PGD:
            pre_fx = f'Valid Attack PGD({steps} steps) '
        if not Train and Attack and PGD and target_top2:
            pre_fx = f'Valid Target top2 Attack PGD({steps} steps) '
        if not Train and Attack and auto:
            pre_fx = 'Valid Auto Attack '
        

        for k, v in metrics.items():
            if rank0():
                logger.info(f"{pre_fx}{k}: {v}")



    max_clean_acc = 0
    max_pgd_acc = 0

    for epoch in range(1, num_epochs + 1):
        if rank0():
            logger.info(f"Epoch {epoch}: ")
            logger.info("Train:")

        threshold = 10
        turned_on = False if epoch < 6 else True
        # turned_on = False

        train_metrics = trainer.train(epoch, turned_on=turned_on, epoch=epoch)
        print_metrics(train_metrics, Train=True)
        if rank0():
            logger.info("Valid:")
        clean_valid_metrics = trainer.valid(valid_dataloader_cifar10)
        print_metrics(clean_valid_metrics, Train=False)

        valid_metrics = trainer.valid(valid_dataloader_cifar10, attack=True, eps=8/255)
        print_metrics(valid_metrics, Train=False, Attack=True)
        valid_metrics = trainer.valid(valid_dataloader_cifar10, attack=True, rs=True, alpha=10/255)
        print_metrics(valid_metrics, Train=False, Attack=True, rs=True)

        # valid_metrics = trainer.valid(valid_dataloader_cifar10, attack=True, target_top2=True)
        # print_metrics(valid_metrics, Train=False, Attack=True, target_top2=True)
        # valid_metrics = trainer.valid(valid_dataloader_cifar10, attack=True, rs=True, target_top2=True)
        # print_metrics(valid_metrics, Train=False, Attack=True, rs=True, target_top2=True)

        # valid_metrics = trainer.valid(valid_dataloader_cifar10, attack=True, PGD=True, target_top2=True, num_iters=50, random_eps=8/255, alpha=2/255)
        # print_metrics(valid_metrics, Train=False, Attack=True, target_top2=True, PGD=True, steps=50)
        pgd_valid_metrics = trainer.valid(valid_dataloader_cifar10, attack=True, PGD=True, target_top2=False, num_iters=50, random_eps=8/255, alpha=2/255, last_valid=not (epoch % threshold == 0))
        print_metrics(pgd_valid_metrics, Train=False, Attack=True, target_top2=False, PGD=True, steps=50)


        if epoch % threshold == 0:
            logger.info("PGD real robust test (100 steps): ")
            # valid_metrics = trainer.valid(valid_dataloader_cifar10, attack=True, target_top2=True, PGD=True, num_iters=100, random_eps=8/255, alpha=2/255)
            # print_metrics(valid_metrics, Train=False, Attack=True, target_top2=True, PGD=True, steps=100)
            valid_metrics = trainer.valid(valid_dataloader_cifar10, attack=True, target_top2=False, PGD=True, num_iters=100, random_eps=8/255,  alpha=2/255, last_valid=(epoch % threshold == 0))
            print_metrics(valid_metrics, Train=False, Attack=True, target_top2=False, PGD=True, steps=100)


        # if epoch % 10 == 0:
        #     valid_metrics = trainer.valid(valid_dataloader_cifar10, attack=True, target_top2=False, PGD=True, num_iters=100, alpha=2/255)
        #     print_metrics(valid_metrics, Train=False, Attack=True, target_top2=False, PGD=True, steps=100)
            # valid_metrics = trainer.valid(valid_dataloader_cifar10, attack=True, use_auto=True, last_valid=epoch % 5 == 0)
            # print_metrics(valid_metrics, Train=False, Attack=True, auto=True,)

        # if epoch == 10:
        if clean_valid_metrics['Accuracy'] > max_clean_acc and pgd_valid_metrics['Accuracy'] > max_pgd_acc:
            cache = '/home/hice1/yyu496/kaggle/CW/Deep_Optimization/Model/PGD.pt'
            torch.save(trainer.get_final_engine().state_dict(), cache)

            max_clean_acc = clean_valid_metrics['Accuracy'] 
            max_pgd_acc = pgd_valid_metrics['Accuracy']

        #     break


        scheduler.step()
        if rank0():
            logger.info("\n\n")




    if rank0():
        logger.info("Test unattack")
        test_metrics = trainer.valid(test_dataloader_cifar10, attack=False)
        print_metrics(test_metrics)


    if rank0():
        logger.info("Test attack")
        test_metrics = trainer.valid(test_dataloader_cifar10, attack=True)
        print_metrics(test_metrics)

    clean()




if __name__ == "__main__":
    main()