import torch
from torch import nn
import torch.fx as FX
from torch.fx.passes.shape_prop import ShapeProp
from torch._subclasses.fake_tensor import FakeTensorMode


from quantizer import Quantizer
from layers import DOLinear, DOConv1d, DOConv2d, DOConv3d, DODepthPointConv2d, RMSNorm
from modules.activations.act_layers import DOReLU_Variance, DOSiLU, DOGELU
from modules.normalization.normalization_layers import DOBatchNorm2d, DOSyncBatchNorm2d
from fusion.fusion_utils import fuse_bn_act

from torchvision.transforms import v2



class Controller():
    def __init__(self, model, config, train_loader=None, criterion=None, test=True):
        self.model = model.to('cuda')
        self.config = config
        self.train_loader = train_loader
        self.criterion = criterion

        x, y = next(iter(self.train_loader))
        self.input_shape = x.shape[1:]
        self.config['batch_size'] = config['batch_size']

        self.quantizer = Quantizer(config, {})
        self.quantizer.filter_tensors(self.model.named_parameters())
        self.meta = {'group_size' : self.config.get('group_size', 256),
                     'fp8' : self.config.get('fp8', False),
                     'analyze' : self.config.get('analyze', False)}

        if not test:
            self.select_division_layer()
            self.warp_model()
            self.quantizer.bits = self.target_dict.copy()
        

    def iterate(self, criterion=None, trainloader=None):
        if trainloader is not None:
            self.quantizer.iterate(self.model, criterion, trainloader)
        else:
            self.quantizer.iterate(self.model, self.criterion, self.train_loader)

    def select_division_layer(self):
        self.low_frequency_energy_ratio_grad, self.low_frequency_energy_ratio_act, self.lars_trust_ratio, self.signal_noise_ratio, self.activation_var = self.quantizer.select_division_layer_helper(self.model, self.criterion, self.train_loader)
        vals_act = torch.tensor(list(self.low_frequency_energy_ratio_act.values()), dtype=torch.float32)
        vals_grad = torch.tensor(list(d for d in self.low_frequency_energy_ratio_grad.values() if d != 1), dtype=torch.float32)

        vals_lars_trust_ratio = torch.tensor(list(self.lars_trust_ratio.values()), dtype=torch.float32)
        vals_signal_noise_ratio = torch.tensor(list(self.signal_noise_ratio.values()), dtype=torch.float32)

        # 0.65
        self.base_division_threshold_act = torch.quantile(vals_act, 0.75).item()
        self.base_division_threshold_grad = torch.quantile(vals_grad, 0.75).item()
        self.base_threshold_signal_noise_ratio = torch.quantile(vals_signal_noise_ratio, 0.50).item()
        self.base_threshold_lars_trust_ratio = torch.quantile(vals_lars_trust_ratio, 0.5).item()

        self.layer_clamp = 3.0

    def warp_model(self, graph_mode=False, quantizer=False):
        traced_model = FX.symbolic_trace(self.model)
        named_mods = dict(traced_model.named_modules())

        with FakeTensorMode(allow_non_fake_inputs=True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fake_inputs = torch.empty(
                    self.config['batch_size'],
                    *self.input_shape,
                    dtype=torch.bfloat16,
                    device='cuda'
                )

                ShapeProp(traced_model).propagate(fake_inputs)

        replacement_pair = dict()
        for node in traced_model.graph.nodes:

            if node.op != "call_module":
                continue

            target = node.target
            mod = named_mods[target]

            if not isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm2d, nn.SyncBatchNorm, nn.ReLU, nn.SiLU, nn.GELU)):
                continue

            act_size = compute_activation_size_in_bytes(node)
            if act_size <= 10:
                continue
            
            
            tensor_meta = assign_tensor_meta(node)
            replacement_pair[target] = {
                'mod' : mod,
                'tensor_meta' : tensor_meta
            }

        
        for idx, (target, mod_dict) in enumerate(replacement_pair.items()):
            mod = mod_dict['mod']
            tm = mod_dict['tensor_meta']
            
            if quantizer:
                self.iterate()

                self.meta.update({f'{target}':{'tensor_meta' : tm,
                                               'pack_only' : False,
                                               'bits' : self.config['default_bits']},
                    'batch_size' : self.config['batch_size']})
                
                if isinstance(mod, (nn.ReLU, nn.ReLU6)):
                    self.meta[f'{target}']['pack_only'] = True
                    self.meta[f'{target}']['bits'] = 1
                else:
                    self.meta[f'{target}']['pack_only'] = False

                # self.low_rank_activations = self.quantizer.low_rank_activations
                # if target in self.low_rank_activations:
                #     low_rank_act = self.low_rank_activations[target]
                #     self.meta[f'{target}']['low_rank_activations'] = low_rank_act
                # else:
                #     self.meta[f'{target}']['low_rank_activations'] = None

                if len(tm.shape) == 4:
                    if isinstance(mod, nn.BatchNorm2d):
                        total_elt_in_group = tm.shape[0] * tm.shape[2] * tm.shape[3]
                        N = tm.shape[1]

                        special = True if isinstance(mod, nn.BatchNorm2d) else False
                        group_size, act_padding = find_group_size(total_elt_in_group, self.quantizer.bits[f'{target}'], special=special)
                        G = (total_elt_in_group + group_size - 1) // group_size
                    else:
                        # if tm.shape[1] % 2 == 0 and not self.meta[f'{target}']['pack_only']:
                        #     total_elt_in_group = (tm.shape[1] // 2) * tm.shape[2] * tm.shape[3]
                        # else:
                        total_elt_in_group = tm.shape[1] * tm.shape[2] * tm.shape[3]
                        N = tm.shape[0]

                        special = True if isinstance(mod, nn.BatchNorm2d) else False
                        group_size, act_padding = find_group_size(total_elt_in_group, self.quantizer.bits[f'{target}'], special=special)
                        G = (total_elt_in_group + group_size - 1) // group_size
                elif len(tm.shape) == 3:
                    if isinstance(mod, nn.BatchNorm2d):
                        total_elt_in_group = tm.shape[0] * tm.shape[2]
                        N = tm.shape[1]

                        special = True if isinstance(mod, nn.BatchNorm2d) else False
                        group_size, act_padding = find_group_size(total_elt_in_group, self.quantizer.bits[f'{target}'], special=special)
                        G = (total_elt_in_group + group_size - 1) // group_size
                    else:
                        # if tm.shape[1] % 2 == 0 and not self.meta[f'{target}']['pack_only']:
                        #     total_elt_in_group = (tm.shape[1] // 2) * tm.shape[2]
                        # else:
                        total_elt_in_group = tm.shape[1] * tm.shape[2]
                        N = tm.shape[0]

                        special = True if isinstance(mod, nn.BatchNorm2d) else False
                        group_size, act_padding = find_group_size(total_elt_in_group, self.quantizer.bits[f'{target}'], special=special)
                        G = (total_elt_in_group + group_size - 1) // group_size
                else:
                    # if tm.shape[1] % 2 == 0 and not self.meta[f'{target}']['pack_only']:
                    #     total_elt_in_group = tm.shape[1] // 2
                    # else:
                    total_elt_in_group = tm.shape[1]
                    N = tm.shape[0]

                    special = True if isinstance(mod, nn.BatchNorm2d) else False
                    group_size, act_padding = find_group_size(total_elt_in_group, self.quantizer.bits[f'{target}'], special=special)
                    G = (total_elt_in_group + group_size - 1) // group_size

                self.meta[f'{target}']['group_size'] = group_size
                self.meta[f'{target}']['act_padding'] = act_padding
                self.meta[f'{target}']['N'] = N
                self.meta[f'{target}']['NG'] = N*G

                if self.config.get('AVG_ALAM', None) is not None:
                    self.meta[f'{target}'].update({'AVG_ALAM' : self.config['AVG_ALAM']})
                    self.meta[f'{target}'].update({'ALAM_BITS' : self.config['AVG_ALAM_BTS']})
                else:
                    self.meta[f'{target}'].update({'AVG_ALAM' : False})
                    self.meta[f'{target}'].update({'ALAM_BITS' : 0})

                if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, DOLinear, DOConv1d, DOConv2d, DOConv3d)):
                    self.meta[f'{target}'].update({'AVG_ALAM' : False})
                    # self.meta[f'{target}'].update({'bits' : 2})
                # if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm, DOBatchNorm2d, DOSyncBatchNorm2d)):
                #     self.meta[f'{target}'].update({'bits' : 1})
                # if isinstance(mod, (nn.BatchNorm2d, nn.SyncBatchNorm, DOBatchNorm2d, DOSyncBatchNorm2d)):
                #     self.meta[f'{target}']['AVG_ALAM'] = True
                #     self.meta[f'{target}']['ALAM_BITS'] = 2
                
                
                if (self.config.get('DIVISION', None) is not None
                    and target in self.low_frequency_energy_ratio_grad
                    and self.low_frequency_energy_ratio_grad[target] > self.base_division_threshold_grad
                    and self.low_frequency_energy_ratio_grad[target] != 1):
                    self.meta[f'{target}'].update({'DIVISION' : self.config['DIVISION']})
                else:
                    self.meta[f'{target}'].update({'DIVISION' : None})
            
                if isinstance(mod, nn.Linear):
                    new_mod = DOLinear(
                        mod.in_features,
                        mod.out_features,
                        clamp_alpha=self.layer_clamp,
                        target_name=target,
                        meta=self.meta
                    )
                elif isinstance(mod, nn.Conv1d):
                    new_mod = DOConv1d(
                        mod.in_channels, mod.out_channels, mod.kernel_size,
                        mod.stride, mod.padding, mod.dilation, mod.groups, mod.padding_mode,
                        clamp_alpha=self.layer_clamp, target_name=target,
                        meta=self.meta
                    )
                elif isinstance(mod, nn.Conv2d):
                    if self.config['depth_point_conv']:
                        new_mod = DODepthPointConv2d(mod.in_channels, mod.out_channels, mod.kernel_size,
                                                    mod.stride, mod.padding, mod.dilation, mod.groups, mod.padding_mode,
                                                    acc_var=self.layer_clamp, target_name=target,
                                                    meta=self.meta)
                    else:
                        new_mod = DOConv2d(
                            mod.in_channels, mod.out_channels, mod.kernel_size,
                            mod.stride, mod.padding, mod.dilation, mod.groups, mod.padding_mode,
                            clamp_alpha=self.layer_clamp, target_name=target,
                            meta=self.meta
                        )
                elif isinstance(mod, nn.Conv3d):
                    new_mod = DOConv3d(
                        mod.in_channels, mod.out_channels, mod.kernel_size,
                        mod.stride, mod.padding, mod.dilation, mod.groups, mod.padding_mode,
                        clamp_alpha=self.layer_clamp, target_name=target,
                        meta=self.meta
                    )
                elif isinstance(mod, nn.BatchNorm2d):
                    new_mod = DOBatchNorm2d(
                        mod.num_features, mod.eps, mod.momentum,
                        mod.affine, mod.track_running_stats,
                        target_name=target,
                        meta=self.meta
                    )
                elif isinstance(mod, nn.SyncBatchNorm):
                    new_mod = DOSyncBatchNorm2d(
                        mod.num_features, mod.eps, mod.momentum,
                        mod.affine, mod.track_running_stats,
                        mod.process_group,
                        target_name=target,
                        meta=self.meta
                    )
                elif isinstance(mod, nn.ReLU):
                    new_mod = DOReLU_Variance(
                        inplace=False,
                        relu=True,
                        relu6=False,
                        target_name=target,
                        meta=self.meta
                    )
                elif isinstance(mod, nn.ReLU6):
                    new_mod = DOReLU_Variance(
                        inplace=False,
                        relu=False,
                        relu6=True,
                        target_name=target,
                        meta=self.meta
                    )
                elif isinstance(mod, nn.SiLU):
                    new_mod = DOSiLU(
                        inplace=False,
                        target_name=target,
                        meta=self.meta
                    )
                elif isinstance(mod, nn.GELU):
                    new_mod = DOGELU(
                        inplace=False,
                        target_name=target,
                        meta=self.meta
                    )
                
                if self.config['rms_norm']:
                    if isinstance(mod, nn.LayerNorm):
                        new_mod = RMSNorm(
                            dims=mod.normalized_shape[-1]
                        )

                if not isinstance(new_mod, DODepthPointConv2d) and hasattr(mod, 'weight'):
                    with torch.no_grad():
                        new_mod.weight.copy_(mod.weight)
                
                parent_path, _, child_path = target.rpartition(".")
                parent_module = traced_model.get_submodule(parent_path)
                setattr(parent_module, child_path, new_mod)

            traced_model.graph.eliminate_dead_code()
            traced_model.graph.lint()
            traced_model.recompile()

            if graph_mode:
                torch.backends.cuda.matmul.allow_tf32=True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
                for m in self.traced_model.modules():
                    if hasattr(m, 'graph_mode'):
                        m.graph_mode = True

            # if quantizer:
            #     traced_model = fuse_bn_act(traced_model)

            traced_model = traced_model.to('cuda').train()

        self.traced_model = traced_model
        self.target_dict = replacement_pair

    
    def enable_graph_mode(self, stream):
        for m in self.traced_model.modules():
            if hasattr(m, 'graph_mode'):
                m.graph_mode = True
        
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.traced_model.compile(fullgraph=True, mode='reduce-overhead')
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    
    def warmup(self, data_loader, criterion, stream):
        self.iterate(criterion)
        self.warp_model(graph_mode=True, quantizer=True)
        self.enable_graph_mode(stream)

        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                x, y = next(iter(data_loader))
                x, y = x.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)

                logits = self.traced_model(x)
                loss = criterion(logits, y)
                loss.backward()

        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()



def assign_tensor_meta(node):
    for arg in node.args:
        if isinstance(arg, torch.fx.Node) and arg.op != "get_attr":
            tm = arg.meta.get("tensor_meta", None)
            if tm is not None:
                return tm
        return None
    

def find_group_size(S, bits, max_group=512, special=False):
    groups = [512,256,128, 64]
    if special:
        groups = [1024, 512,256,128, 64]

    for gs in groups:
        if bits == 1 and bits % 1 == 0 and S % gs == 0:
            return gs, False
        elif bits == 2 and bits % 2 == 0 and S % gs == 0:
            return gs, False
        elif bits == 4 and bits % 4 == 0 and S % gs == 0:
            return gs, False
        elif bits == 8 and bits % 8 == 0 and S % gs == 0:
            return gs, False
    return 256, True


def compute_activation_size_in_bytes(node):
    tm = node.meta.get('tensor_meta', None)
    if tm is None:
        return 0
    
    numel = 1
    for s in tm.shape:
        numel *= s
    act_size = numel * torch.tensor([], dtype=tm.dtype).element_size()
    return act_size / (1024 ** 2)