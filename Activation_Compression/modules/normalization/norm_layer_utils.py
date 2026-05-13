import torch
import torch.nn as nn
import torch.distributed as dist

from ..normalization import normalization_layers as norm_layers



def convert_do_sync_batchnorm(model):
    process_group = dist.group.WORLD

    for name, child in model.named_children():
        if isinstance(child, norm_layers.DOBatchNorm2d):
            new_bn = norm_layers.DOSyncBatchNorm2d(
                child.num_features,
                child.eps,
                child.momentum,
                child.affine,
                child.track_running_stats,
                process_group=process_group,
                target_name=f"sync_{child.target_name}",
                meta=child.meta,
            ).cuda()
            setattr(model, name, new_bn)
        else:
            convert_do_sync_batchnorm(child)
    return model



def check_sync_bn(model):
    res = False
    for m in model.modules():
        if isinstance(m, norm_layers.DOSyncBatchNorm2d):
            res = True
        elif isinstance(m, nn.BatchNorm2d):
            res = False
    return res
