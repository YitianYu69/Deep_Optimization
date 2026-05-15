import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchmetrics import Metric
from torchvision import transforms

import deepspeed
import torchattacks

from .log import get_logger
from .utils_train import warmup, build_CUDA_Graph, wrap_model_prepare_qat, Setup_Criterion, EMA
from .utils_ddp import rank0, setup_ddp


from ..Activation_Compression.controller import Controller
from ..Activation_Compression.modules import layers
from ..Activation_Compression.modules.normalization.norm_layer_utils import convert_do_sync_batchnorm

from ..Adversarial_Attack.FGSM import FGSM_attack, PGD_attack, TRADES_attack
from ..Adversarial_Attack.AWP import AWP
from ..Optimizer.SGD_geometry import SGD_NS_Overshoot, SGD_NS_Overshoot_Noise

import torchattacks
import torch_dct as dct

import queue
import time
import copy
from typing import Union, Callable, Dict, Optional

logger = get_logger()


class Trainer():
    def __init__(self, 
                 *,
                 model: nn.Module,
                 teacher_model: nn.Module = None,
                 compile_type: str = None,
                 DS_config: Dict = None,
                 DDP_config: Dict = None,
                 ACT_config: Dict = None,
                 CUDA_Graph: bool = False,
                 Trainer_config: Dict = None,
                 QAT: bool = False,
                 Adversarial_Attack: Dict = {},
                 amp_enable: bool = True,
                 dataloader: DataLoader = None,
                 sub_data_portion: float = 1.0,
                 criterion: dict = {},
                 optimizer_type: Optional[torch.optim.Optimizer] = None,
                 optimizer_kwargs: Optional[dict] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
                 scaler: torch.amp.GradScaler = None,
                 metrics: Dict[str, Metric] = None,
                 ema: Callable = None,
                 ema_kwargs: dict = {},
                 num_epochs: int = 200,
                 grad_acc_step: int = 1,
                 grad_norm_clip: bool = False,
                 image_size: int = 256,
                 device: Union[str, torch.device] = 'cpu'):
        
        self.DS_config = DS_config
        self.DDP_config = DDP_config
        self.ACT_config = ACT_config
        self.CUDA_Graph = CUDA_Graph
        self.Trainer_config = Trainer_config
        self.amp_enable = amp_enable
        self.QAT = QAT
        self.Adversarial_Attack = Adversarial_Attack
        self.train_dataloader = dataloader
        self.cri = criterion
        self.opt_type = optimizer_type
        self.opt_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scaler = scaler
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.grad_acc_step = grad_acc_step
        self.grad_norm_clip = grad_norm_clip
        self.device = device.type if isinstance(device, torch.device) else device
        self.teacher_model = teacher_model

        self.awp = None

        self.cuda_timer_start = torch.cuda.Event(enable_timing=True)
        self.cuda_timer_end = torch.cuda.Event(enable_timing=True)

        self.is_training = False

        if ema is not None:
            self._ema = ema
            self.ema_kwargs = ema_kwargs
        else:
            self._ema is None

        
        self.attack_types = self.Adversarial_Attack.get('Attack_Type', {})
        if self.Trainer_config.get('Multi_View', False):
            self.view_types, self.num_chunks = self._compute_views_and_counts()


        # Check confliction
        assert (self.QAT != self.amp_enable) or (not self.QAT and not self.amp_enable), "Please choose either QAT=True, or amp_enable=True!"
        assert not (self.DS_config is not None and self.DDP_config is not None) , "Please choose either Deep Speed, or DDP!"
        assert not (self.DS_config is not None and self.ACT_config is not None), "Please choose either Deep Speed, or Activation Compression!"
        assert not (self.ACT_config is not None and self.train_dataloader is None), "Please also pass the train_dataloader when ACT is enabled!"

        if self.Adversarial_Attack:
            assert 'Attack_Type' in self.Adversarial_Attack, "Please provide Attack_Type inside of the Adversarial_Attack config"
            assert 'mu' in self.Adversarial_Attack, "Please provide mu inside of the Adversarial_Attack config"
            assert 'std' in self.Adversarial_Attack, "Please provide mu inside of the Adversarial_Attack config"

        if self.Trainer_config.get("Multi_View", False) and self.Adversarial_Attack is None:
            raise ValueError("Current Multi_View only support adversaria attack!")


        if self.QAT:
            model = wrap_model_prepare_qat(model, image_size)

        # If using CUDA, move model to device first!
        if self.device != 'cpu':
            model.to(device)
            if teacher_model is not None:
                teacher_model.to(device).eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
                self.teacher_model = teacher_model
        
        # Check for compile
        if compile_type is not None:
            assert ACT_config is not None, "Please turn off compile for when ACT is enabled, they are not compatible at the moment!"

            fullgraph = False if self.DS_config is not None else True
            model.compile(fullgraph=fullgraph, mode=compile_type)
            if (self.DS_config is not None or self.DDP_config is not None) and compile_type != 'reduce-overhead' and rank0():
                logger.info("""For the max speed optimization, consider enabling reduce-overhead compile mode
                            to avoid rebuilding autograd graph!""")
            if teacher_model is not None:
                teacher_model.compile(fullgraph=fullgraph, mode=compile_type)
                self.teacher_model = teacher_model


        # ---------------------------------------------------
        # If PT2 AMP enabled, auto check the best cast dtype
        # ---------------------------------------------------
        if self.DS_config is None and amp_enable and self.device.startswith('cuda'):
            major, minor = torch.cuda.get_device_capability(torch.device(device))
            if major == 7 and minor == 5:
                self.cast_dtype = torch.bfloat16
            else:
                self.cast_dtype = torch.bfloat16 if major >= 8 and torch.cuda.is_available() else torch.float16

            if self.cast_dtype == torch.float16 and scaler is None:
                raise ValueError(f"AMP float16 is enabled, then the scaler cannot be None!")
        elif self.DS_config is None and amp_enable and self.device.startswith('cpu'):
            self.cast_dtype = torch.bfloat16
        else:
            self.cast_dtype = None
            

        # ---------------------------------------------
        # Wrap model to enable different optimization
        # ---------------------------------------------
        if self.CUDA_Graph and self.DS_config is None:
            if rank0():
                logger.info("CUDA Graph Enabled!")
            side = torch.cuda.Stream(device=device)
            warmup_stream = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.Stream(side):
                self.engine = self._wrap_model_to_engine(model)
            torch.cuda.current_stream(device).wait_stream(side)
            torch.cuda.synchronize(device)
            (self.graph_sync, self.graph_no_sync, self.static_x, self.static_y, self.static_logits, self.static_loss, self.compute_stream) = build_CUDA_Graph(self.engine, 
                                                                                                                                            self.cri['Train'], self.opt, self.train_dataloader, 
                                                                                                                                            self.amp_enable, self.cast_dtype, 
                                                                                                                                            self.device, self.scaler, self.grad_acc_step)
            self.copy_stream = torch.cuda.Stream(device=device)
            self.copy_event = torch.cuda.Event()
        else:
            self.engine = self._wrap_model_to_engine(model)

        if self.Trainer_config.get("L1_Sparse_Loss", False):
            self._get_target_activation()

        if self.opt_type is not None and self.opt_kwargs is not None:
            self.opt = self.opt_type(self.engine.parameters(), **self.opt_kwargs)
        else:
            raise ValueError("Please provide a optimizerr class")
        


    def train(self, epoch_idx, turned_on=False, epoch=0):
        assert not (self.train_dataloader is None), "Please pass the train_dataloader into the Trainer when you declare it first."
        assert not (self.metrics is None), "Please pass metrics into the Trainer when declare it as a dict."
        self.is_training = True
        return self._training(epoch_idx, turned_on=turned_on, epoch=epoch)
    
    def valid(self, dataloader, attack = False, rs=False, target_top2=False, PGD=False, num_iters=7, eps=8/255, random_eps=8/255, alpha=10/255, LI=True, num_class=10, use_auto=False, last_valid=False):
        self.is_training = False
        return self._validation(dataloader, attack=attack, rs=rs, target_top2=target_top2, PGD=PGD, num_iters=num_iters, eps=eps, random_eps=random_eps, alpha=alpha, LI=LI, num_class=num_class, use_auto=use_auto, last_valid=last_valid)
    
    def get_Engine(self):
        return self.engine
    
    def get_Optimizer(self):
        return self.opt

    def _is_deepspeed(self):
        return isinstance(self.engine, deepspeed.DeepSpeedEngine)

    def _guard_all_reduce_SUM(self, t):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t
    
    def _get_target_activation(self):
        self.l1_act = None
        def forward_hook():
            def hook(module, input, output):
                self.l1_act = output
            return hook
        
        modules = []
        for m in self.engine.modules():
            if isinstance(m, (nn.Conv2d, layers.DOConv2d)):
                if m.kernel_size[0] == m.kernel_size[1] and m.kernel_size[1] > 1:
                    modules.append(m)
        
        if modules:
            modules[-1].register_forward_hook(forward_hook())
        else:
            if rank0():
                logger.info('Apply fwd hook to extract fwd activatin failed')

    
    def _wrap_model_to_engine(self, model, wrap_type='raw'):
        model = model.to(self.device)

        if self.DS_config is not None:
            engine, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config=self.DS_config
            )
            if rank0():
                logger.info("Model Wrap Type: DeepSpeed!")
        elif self.DDP_config is not None:
            if self.ACT_config is not None:
                self.act_controller = Controller(model, self.ACT_config, self.train_dataloader, self.cri['Valid'], test=False)
                self.act_controller.iterate(criterion=self.cri['Valid'])
                self.act_controller.warp_model(graph_mode=True, quantizer=True)

                model = self.act_controller.traced_model

                if self.ACT_config.get('SyncBatchNorm', False):
                    model = convert_do_sync_batchnorm(model)
                if rank0():
                    logger.info("Model Inner Wrap Type: Activation Compression")
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


            # apply_parametrizations(model)


            if self.Adversarial_Attack.get('Attack_Type', {}) and self.Adversarial_Attack.get('Attack_Type', {}).get('AWP', {}):
                awp_config = self.Adversarial_Attack.get('Attack_Type', {}).get('AWP', {})
                self.awp = AWP(**awp_config)


            if self._ema is not None:
                self.ema = self._ema(model, **self.ema_kwargs)

            if rank0() and self.DDP_config.get('broadcast_buffers', True):
                logger.info('Please turn off the broadcast_buffers if you used the torch.nn.SyncBatchNorm.convert_sync_batchnorm().')
            if rank0() and self.DDP_config.get('gradient_as_bucket_view', False):
                logger.info('Please set set_to_none=True for optimizer.zero_grad(); otherwise DDP grad buckets may be zeroed out.')

            ddp_kwargs = dict(
                static_graph=self.DDP_config.get('static_graph', True),
                broadcast_buffers=self.DDP_config.get('broadcast_buffers', False),
                bucket_cap_mb=self.DDP_config.get('bucket_cap_mb', 25),
                find_unused_parameters=self.DDP_config.get('find_unused_parameters', False),
                gradient_as_bucket_view=self.DDP_config.get('gradient_as_bucket_view', False),
            )

            engine = DDP(
                model,
                device_ids=self.DDP_config.get('device_ids'),
                **ddp_kwargs)
            
            # replace the logger with a no-op
            class _NoopDDPLogger:
                def set_runtime_stats_and_log(self, *a, **k): pass
                def set_and_log_parameter(self, *a, **k): pass
                def _log_stats(self, *a, **k): pass
            engine.logger = _NoopDDPLogger()
            if rank0():
                logger.info("Model Wrap Type: DDP")


        elif self.ACT_config is not None:
                self.act_controller = Controller(model, self.ACT_config, self.train_dataloader, self.cri['Valid'], test=False)
                self.act_controller.iterate(criterion=self.cri['Valid'])
                self.act_controller.warp_model(graph_mode=True, quantizer=True)

                engine = self.act_controller.traced_model
                logger.info("Model Wrap Type: Activation Compression")

                if self._ema is not None:
                    self.ema = self._ema(engine, **self.ema_kwargs)

        else:
            engine = model
            if rank0():
                logger.info("Model Wrap Type: Raw")

            if self._ema is not None:
                self.ema = self._ema(engine, **self.ema_kwargs)

        return engine

    def _training_step(self, data, target, grad_step, epoch=0, step=0):
        backup = None

        if self.teacher_model is not None:
            with torch.inference_mode():
                teacher_logits = self.teacher_model(data)

        if self._is_deepspeed():
            logits = self.engine(data)
            ori_loss = self.cri['Train'](logits, labels=target) if self.teacher_model is None else self.cri['Train'](logits, labels=target, teacher_logits=teacher_logits)
            self.engine.backward(ori_loss)
            self.engine.step()
        else:
            # Branch using CUDA Graph or not
            # Not use
            if not self.CUDA_Graph:
                device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
                with torch.autocast(device_type=device_type, 
                                    dtype=(self.cast_dtype if device_type in ['cuda', 'cpu'] else None),
                                    enabled=self.amp_enable and device_type in ['cuda', 'cpu']):
                    self.opt.zero_grad(set_to_none=True)

                    EMA_teacher_controller = self.Trainer_config.get("EMA_Teacher", {})
                    ema_logits = None
                    if self.ema is not None and isinstance(self.ema, EMA) and len(EMA_teacher_controller) != 0:
                        if epoch >= EMA_teacher_controller.get("Start_Epoch", 6) and (EMA_teacher_controller.get("full_logits", False) or EMA_teacher_controller.get("clean_logits", False) or EMA_teacher_controller.get("MOC", False)):
                            with torch.no_grad():
                                with self.ema.average_parameters(self.engine):
                                    ema_logits = self.engine(data)

                        if self.Trainer_config.get("SAM", {}) and self.Trainer_config.get("SAM", {}).get('turn_on', False):
                            backup = self._sam(step, data, target)

                        
                        self.compute_AWP_diff_and_perturbate(data, target, epoch=epoch)
                        logits = self.engine(data)


                        if ema_logits is not None and EMA_teacher_controller.get("full_logits", False):
                            log_p = F.log_softmax(logits, dim=1)
                            q = F.softmax(ema_logits.detach(), dim=1)
                            dl_loss = F.kl_div(log_p, q, reduction='batchmean')
                    else:
                        if self.Trainer_config.get("SAM", {}) and self.Trainer_config.get("SAM", {}).get('turn_on', False):
                            backup = self._sam(step, data, target)


                        self.compute_AWP_diff_and_perturbate(data, target, epoch=epoch)
                        logits = self.engine(data)

                    if isinstance(self.cri, Setup_Criterion):
                        ori_loss = self.cri['Train'](logits, labels=target) if self.teacher_model is None else self.cri['Train'](logits, labels=target, teacher_logits=teacher_logits)
                    # elif isinstance(self.cri, nn.CrossEntropyLoss):
                    else:
                        # log_logit_stats(logits, labels=target, logger=logger)
                        ori_loss = self.cri['Train'](logits, target)


                        if self.Trainer_config.get("Multi_View", False):                                
                            assert logits.size(0) == target.size(0)
                            assert logits.size(0) % self.num_chunks == 0, (
                                f"Bad multiview chunking: logits={logits.size(0)}, "
                                f"target={target.size(0)}, num_chunks={self.num_chunks}, views={self.view_types}"
                            )

                            logits_chunks = logits.chunk(self.num_chunks, dim=0)
                            target_chunks = target.chunk(self.num_chunks, dim=0)

                            logits_view_map = dict(zip(self.view_types, logits_chunks))
                            target_view_map = dict(zip(self.view_types, target_chunks))

                            if ema_logits is not None and EMA_teacher_controller.get("clean_logits", False):
                                target_ema_logits = ema_logits.chunk(self.num_chunks, dim=0)
                                kl_clean_logits = target_ema_logits[0]
                            else:
                                kl_clean_logits = logits_view_map['Clean']
                            
                            T = float(self.Adversarial_Attack.get("KL_temperature", 1.0))
                            assert T > 0, f"KL_temperature must be > 0, got {T}"
                            if "FGSM" in self.attack_types:
                                if 'FGSM_Small' in self.view_types:
                                    attack_key = 'FGSM_Small'
                                else:
                                    attack_key = 'FGSM'
                                # kl1 = F.kl_div(
                                #     F.log_softmax(logits_view_map[attack_key] / T, dim=1),
                                #     F.softmax(kl_clean_logits.detach() / T, dim=1),
                                #     reduction='batchmean'
                                # )
                                # kl1 = F.kl_div(
                                #     F.log_softmax(kl_clean_logits / T, dim=1),
                                #     F.softmax(logits_view_map[attack_key] / T, dim=1),
                                #     reduction='batchmean'
                                # )
                                # ori_loss += kl1 * (T * T)
                           
                            if "FGSM_RS" in self.attack_types:
                                if 'FGSM_RS_Small' in self.view_types:
                                    attack_key = 'FGSM_RS_Small'
                                else:
                                    attack_key = 'FGSM_RS'
                                # kl2 = F.kl_div(
                                #     F.log_softmax(logits_view_map[attack_key] / T, dim=1),
                                #     F.softmax(kl_clean_logits.detach() / T, dim=1),
                                #     reduction='batchmean'
                                # )
                                # kl2 = F.kl_div(
                                #     F.log_softmax(kl_clean_logits / T, dim=1),
                                #     F.softmax(logits_view_map[attack_key] / T, dim=1),
                                #     reduction='batchmean'
                                # )
                                # ori_loss += kl2 * (T * T)
                      
                            if "PGD" in self.attack_types:
                                # kl3 = F.kl_div(
                                #     F.log_softmax(logits_view_map['PGD'] / T, dim=1),
                                #     F.softmax(kl_clean_logits.detach() / T, dim=1),
                                #     reduction='batchmean'
                                # )

                                kl3 = F.kl_div(
                                    F.log_softmax(kl_clean_logits / T, dim=1),
                                    F.softmax(logits_view_map['PGD'] / T, dim=1),
                                    reduction='batchmean'
                                )
                           
                                kl_weight = self.attack_types.get('PGD', {}).get('kl_weight', 1.0)
                                ori_loss += kl3 * (T * T) * kl_weight

                            if "TRADES" in self.attack_types:
                                # kl4 = F.kl_div(
                                #     F.log_softmax(logits_view_map['TRADES'] / T, dim=1),
                                #     F.softmax(kl_clean_logits.detach() / T, dim=1),
                                #     reduction='batchmean'
                                # )

                                kl4 = F.kl_div(
                                    F.log_softmax(kl_clean_logits / T, dim=1),
                                    F.softmax(logits_view_map['TRADES'] / T, dim=1),
                                    reduction='batchmean'
                                )
                                beta = self.attack_types.get('TRADES', {}).get('beta', 1.0)
                                ori_loss += (kl4 * (T * T)) * beta

                                clean_margin = margin(logits_view_map['Clean'], target_view_map['Clean'])
                                trades_margin = margin(logits_view_map['TRADES'], target_view_map['TRADES'])
                                ori_loss +=  F.smooth_l1_loss(clean_margin, trades_margin.detach())


                            if len(self.Trainer_config.get('Soft_Margin_Loss', {})) != 0:
                                sml_name = self.Trainer_config['Soft_Margin_Loss'].get('logits_name', 'Clean')
                                ori_loss += top_pred_correction_loss(logits_view_map[sml_name], target_view_map[sml_name], T=1.5)
                                ori_loss += 1 * soft_margin_loss_V2(logits_view_map[sml_name], target_view_map[sml_name], T=1.5, target_margin=4.0, focal=True)
                                ori_loss += 5 * soft_margin_loss_V1(logits_view_map[sml_name], target_view_map[sml_name], T=1.5, target_margin=4.0, focal=True)

                        if self.ema is not None and isinstance(self.ema, EMA) and len(self.Trainer_config.get("EMA_Proximal_Loss", {})) != 0 and epoch >= self.Trainer_config.get("EMA_Proximal_Loss", {}).get("Start_Epoch", 6):
                            rho = self.Trainer_config.get("EMA_Proximal_Loss", {}).get("rho", 5e-4)

                            prox = self.ema.prox_term(self.engine)
                            ori_loss += rho / 2 * prox

                        if len(self.Trainer_config.get("L1_Sparse_Loss", {})) != 0 and self.l1_act is not None:
                            trust_ratio = self.Trainer_config['L1_Sparse_Loss'].get('trust_ratio', 0.001)
                            l1_s_loss = self.l1_act.abs().mean()
                            ori_loss += trust_ratio * l1_s_loss


                            if 'TRADES' in self.attack_types:
                                act_chunks = self.l1_act.chunk(self.num_chunks, dim=0)
                                act_view_map = dict(zip(self.view_types,act_chunks))

                                freq_loss = freq_match_loss(
                                    feat_clean=act_view_map['Clean'],
                                    feat_adv=act_view_map['PGD'],
                                    spectrum_mode="amp",
                                    loss_mode="wasserstein",
                                    low=0.0,
                                    high=1.0,
                                    detach_clean=True,
                                )
                                ori_loss += freq_loss

                        if epoch >= EMA_teacher_controller.get("Start_Epoch", 6) and EMA_teacher_controller.get("full_logits", False):
                            ori_loss += dl_loss

                    # else:
                    #     raise TypeError("Current type of the loss function is not support, if you want to support it, please open a issue.")


                    if isinstance(self.engine, nn.parallel.DistributedDataParallel) and self.grad_acc_step > 1:
                        self.engine.require_backward_grad_sync = grad_step
                    loss = ori_loss / self.grad_acc_step
                    if self.amp_enable and self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

            # Use CUDA Graph
            else:
                if data.shape != self.static_x.shape or target.shape != self.static_y.shape:
                    raise RuntimeError(
                        f"CUDA Graph expects fixed shapes. "
                        f"Got data {tuple(data.shape)} vs {tuple(self.static_x.shape)}, "
                        f"target {tuple(target.shape)} vs {tuple(self.static_y.shape)}."
                    )

                with torch.cuda.stream(self.copy_stream):
                    self.static_x.copy_(data, non_blocking=True)
                    self.cri.set_batch_target(grad_acc_step=self.grad_acc_step, labels=target,) if self.teacher_model is None else self.cri.set_batch_target(grad_acc_step=self.grad_acc_step, labels=target, teacher_logits=teacher_logits)
                    self.copy_event.record(self.copy_stream)

                with torch.cuda.stream(self.compute_stream):
                    self.compute_stream.wait_event(self.copy_event)
                    if grad_step:
                        self.graph_sync.replay()
                    else:
                        self.graph_no_sync.replay()

                logits = self.static_logits
                ori_loss = self.static_loss
            
            if grad_step:
                if self.amp_enable and self.scaler is not None:
                    self.scaler.unscale_(self.opt)
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.engine.parameters(), max_norm=1.0)

                    if self.Trainer_config.get("SAM", {}) and self.Trainer_config.get("SAM", {}).get('turn_on', False):
                            self._de_sam(backup)

                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.engine.parameters(), max_norm=1.0)

                    if self.Trainer_config.get("SAM", {}) and self.Trainer_config.get("SAM", {}).get('turn_on', False):
                        self._de_sam(backup)

                    self.opt.step()

                if self.awp is not None:
                    self.awp.restore(self.engine)

        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update_parameters(self.engine)
        return logits, ori_loss

    @torch.no_grad()
    def _update_metrics(self, logits, target):
        if self.Trainer_config.get('Multi_View', False) and self.Trainer_config.get('Multi_Acc', False):
            logits_chunks = logits.chunk(self.num_chunks, dim=0)
            target_chunks = target.chunk(self.num_chunks, dim=0)

            logits_view_map = dict(zip(self.view_types, logits_chunks))
            target_view_map = dict(zip(self.view_types, target_chunks))

            for k, v in self.metrics.items():
                if k.endswith('_Accuracy'):
                    key = k.split('_Accuracy')[0]
                    v.update(logits_view_map[key].argmax(dim=1), target_view_map[key])
                elif k == 'AUROC':
                    if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
                        v.update(torch.sigmoid(logits), target)
                    else:
                        v.update(F.softmax(logits, dim=1), target)
                else:
                    v.update(logits.argmax(dim=1), target)

        else:
            for k, v in self.metrics.items():
                if k == 'AUROC':
                    if logits.ndim == 1 or (logits.ndim == 2 and logits.size(1) == 1):
                        v.update(torch.sigmoid(logits), target)
                    else:
                        v.update(F.softmax(logits, dim=1), target)
                else:
                    v.update(logits.argmax(dim=1), target)





    def _training(self,
                 epoch_idx: int,
                 turned_on: bool,
                 epoch:int = 0):
        total_loss, data_len = torch.tensor(0.0, dtype=torch.float32, device=self.device), torch.tensor(0, dtype=torch.long, device=self.device)
        computed_metrics = {}

        self.engine.train()

        is_wrapped = isinstance(self.engine, (DDP, deepspeed.DeepSpeedEngine))
        (self.engine.module if is_wrapped else self.engine).train()

        for v in self.metrics.values():
            v.reset()

        if isinstance(self.train_dataloader.sampler, DistributedSampler):
            self.train_dataloader.sampler.set_epoch(epoch_idx)


        cuda_time = 0
        start_time = time.time()
        for step, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            if self.Adversarial_Attack is not None and len(self.Adversarial_Attack.get('Attack_Type', {})) != 0:
                mu = self.Adversarial_Attack.get('mu', (0.5, 0.5, 0.5))
                std = self.Adversarial_Attack.get('std', (0.5, 0.5, 0.5))


                if self.Trainer_config.get("Multi_View", False) and "FGSM" in self.attack_types and "FGSM_RS" in self.attack_types:
                    both = True
                else:
                    both = False
                
                if "FGSM" in self.attack_types or "FGSM_RS" in self.attack_types:
                    eps = as_tuple(self.attack_types.get('FGSM', {}).get('eps', 8/255))
                    alpha = as_tuple(self.attack_types.get('FGSM_RS', {}).get('alpha', 10/255))
                    random_eps = as_tuple(self.attack_types.get('FGSM_RS', {}).get('random_eps', 8/255))
                    LI = self.attack_types.get('LIET', {}).get('LI', False,)
                    num_class = self.attack_types.get('LIET', {}).get('num_class', 10)


                    if both:
                        len_target = len(eps) + len(alpha)
                    else:
                        len_target = max(len(eps), len(alpha))
                        assert len_target < 2, 'Single view does not suppot multi-attacks, pls turn on Multi_View'

                    FGSM_attacked_data_chunks = FGSM_attack(self.engine, self.cri['Valid'], data, target, 
                                                                  both=both, LI=LI, num_class=num_class,
                                                                  eps=eps, random_eps=random_eps, alpha=alpha,
                                                                  mu=mu,
                                                                  std=std,
                                                                  target_top2=False,
                                                                  device=self.device)
                else:
                    FGSM_attacked_data_chunks = None
                    len_target = 1
                
                if "PGD" in self.attack_types:
                    random_eps = self.attack_types.get('PGD', {}).get('random_eps', 8/255)
                    alpha = self.attack_types.get('PGD', {}).get('alpha', 2/255)
                    steps = self.attack_types.get('PGD', {}).get('steps', 7)

                    PGD_attacked_data_chunks = PGD_attack(self.engine, self.cri['Valid'], data, target,
                                                    random_eps=random_eps, alpha=alpha, num_iters=steps,
                                                    mu=mu, std=std,
                                                    target_top2=False, valid=False,
                                                    device=self.device
                                                    )
                else:
                    PGD_attacked_data_chunks = None

                if "TRADES" in self.attack_types:
                    random_eps = self.attack_types.get('TRADES', {}).get('random_eps', 8/255)
                    alpha = self.attack_types.get('TRADES', {}).get('alpha', 2/255)
                    num_iters = self.attack_types.get('TRADES', {}).get('num_iters', 7)

                    TRADES_attacked_data_chunks = TRADES_attack(self.engine, data, labels=target,
                                                                num_iters=num_iters, random_eps=random_eps, alpha=alpha,
                                                                mu=mu, std=std,
                                                                valid=False)
                else:
                    TRADES_attacked_data_chunks = None
                
                if self.Trainer_config.get("Multi_View", False):
                    data = denorm(data, mu, std)

                    temp_data_chunks = [data]
                    temp_target_chunks = [target]
                    if FGSM_attacked_data_chunks is not None:
                        if isinstance(FGSM_attacked_data_chunks, list):
                            temp_data_chunks.extend(FGSM_attacked_data_chunks)
                            temp_target_chunks.extend([target] * len_target)
                        else:
                            temp_data_chunks.append(FGSM_attacked_data_chunks)
                            temp_target_chunks.append(target)

                    if PGD_attacked_data_chunks is not None:
                        temp_data_chunks.append(PGD_attacked_data_chunks)
                        temp_target_chunks.append(target)
                    if TRADES_attacked_data_chunks is not None:
                        temp_data_chunks.append(TRADES_attacked_data_chunks)
                        temp_target_chunks.append(target)
                    if self.Trainer_config.get('Freq_View', False):
                        freq_view = generate_freq_view(data, mu, std)
                        temp_data_chunks.append(freq_view)
                        temp_target_chunks.append(target)


                    data = torch.cat(temp_data_chunks, dim=0)
                    data = transforms.Normalize(mu, std)(data)

                    target = torch.cat(temp_target_chunks, dim=0)
                else:
                    if FGSM_attacked_data_chunks is not None:
                        data = FGSM_attacked_data_chunks
                        data = transforms.Normalize(mu, std)(data)
                    elif PGD_attacked_data_chunks is not None:
                        data = PGD_attacked_data_chunks
                        data = transforms.Normalize(mu, std)(data)
                    elif TRADES_attacked_data_chunks is not None:
                        data = TRADES_attacked_data_chunks
                        data = transforms.Normalize(mu, std)(data)


            grad_step = ((step + 1) % self.grad_acc_step == 0 or (step + 1) == len(self.train_dataloader))

            # if epoch > 2:
            #     # remove_parametrizations(self.engine)

            #     real_model = unwrap_model(self.engine)

            #     old_named_params = named_param_dict(real_model)

            #     # remove_parametrizations(real_model, leave_parametrized=True)

            #     self.opt = remap_optimizer_state_by_name(
            #         self.opt,
            #         old_named_params,
            #         real_model
            #     )

            self.cuda_timer_start.record()
            if self.Adversarial_Attack is not None:
                logits, loss = self._training_step(data, target, grad_step, epoch=epoch, step=step)
            else:
                logits, loss = self._training_step(data, target, grad_step, epoch=epoch, step=step)

            self._update_metrics(logits, target)
            self.cuda_timer_end.record()
            torch.cuda.synchronize()
            cuda_time += self.cuda_timer_start.elapsed_time(self.cuda_timer_end)
        
            total_loss += loss.detach() * data.size(0)
            data_len += len(data)

        end_time = time.time()
        total_loss = self._guard_all_reduce_SUM(total_loss)
        data_len = self._guard_all_reduce_SUM(data_len)
        computed_metrics['Loss'] = total_loss.item() / data_len.item()
        computed_metrics['Time'] = end_time - start_time
        computed_metrics['Total_Step_Time'] = (cuda_time / 1000)
        computed_metrics['Throughput'] = len(self.train_dataloader.dataset) / (cuda_time / 1000)
        for k, v in self.metrics.items():
            computed_metrics[k] = v.compute()

        if isinstance(self.opt, SGD_NS_Overshoot_Noise):
            self.opt.move_to_base()
        
        return computed_metrics

    @torch.no_grad()
    def _validation(self,
                    dataloader: DataLoader,
                    attack: bool = False, 
                    rs: bool =False,
                    target_top2: bool = False,
                    PGD: bool = False,
                    num_iters: int = 7,
                    eps: float = 8/255,
                    random_eps: float = 8/255,
                    alpha: float = 10/255,
                    LI: bool = True, num_class=10,
                    use_auto: bool = False,
                    last_valid: bool = False):
        total_loss, data_len = torch.tensor(0.0, dtype=torch.float32, device=self.device), torch.tensor(0, dtype=torch.long, device=self.device)
        computed_metrics = {}

        self.engine.eval()

        is_wrapped = isinstance(self.engine, (DDP, deepspeed.DeepSpeedEngine))
        (self.engine.module if is_wrapped else self.engine).eval()

        if attack:
            mu = self.Adversarial_Attack.get('mu', (0.5, 0.5, 0.5))
            std = self.Adversarial_Attack.get('std', (0.5, 0.5, 0.5))

        if attack and use_auto:
            if self.ema is not None and isinstance(self.ema, EMA):
                with self.ema.average_parameters(self.engine):
                    atk = torchattacks.AutoAttack(self.engine, eps=8/255)
            else:
                atk = torchattacks.AutoAttack(self.engine, eps=8/255)
            atk.set_normalization_used(mu, std)

        for v in self.metrics.values():
            v.reset()

        start_time = time.time()
        for data, target in dataloader:
            if not attack:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.ema is not None and isinstance(self.ema, EMA):
                        with self.ema.average_parameters(self.engine):
                            logits = self.engine(data)
                    else:
                        logits = self.engine(data)
            else:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                with torch.enable_grad():
                        if self.ema is not None and isinstance(self.ema, EMA):
                            with self.ema.average_parameters(self.engine):
                                if not PGD:
                                    data = FGSM_attack(self.engine, self.cri['Valid'], data, target, 
                                                            random_start=rs, mu=mu, std=std,
                                                            eps=as_tuple(eps), alpha=as_tuple(alpha), LI=LI, num_class=num_class,
                                                            target_top2=target_top2,
                                                            device=self.device)
                                if PGD:
                                    data = PGD_attack(self.engine, self.cri['Valid'], data, target, num_iters=num_iters,
                                                            target_top2=target_top2, random_eps=random_eps, 
                                                            alpha=alpha, mu=mu, std=std, valid=True,
                                                            device=self.device)
                                    
                                if use_auto:
                                    adv_data = atk(data.clone(), target)
                                    logits = self.engine(adv_data)
                                else:
                                    data = transforms.Normalize(mu, std)(data)
                                    with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                                        logits = self.engine(data)
                        else:
                            if not PGD:
                                data = FGSM_attack(self.engine, self.cri['Valid'], data, target, 
                                                        random_start=rs, mu=mu, std=std,
                                                        eps=as_tuple(eps), alpha=as_tuple(alpha),
                                                        target_top2=target_top2,
                                                        device=self.device)
                            if PGD:
                                data = PGD_attack(self.engine, self.cri['Valid'], data, target, num_iters=num_iters,
                                                        target_top2=target_top2, random_eps=random_eps, 
                                                        alpha=alpha, mu=mu, std=std, valid=True,
                                                        device=self.device)
                                
                            if use_auto:
                                adv_data = atk(data.clone(), target)
                                logits = self.engine(adv_data)
                            else:
                                data = transforms.Normalize(mu, std)(data)
                                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                                    logits = self.engine(data)

            if isinstance(self.cri['Valid'], Setup_Criterion):
                loss = self.cri['Valid'](logits, labels=target, valid=True)
            elif isinstance(self.cri['Valid'], nn.CrossEntropyLoss):
                loss = self.cri['Valid'](logits, target)
            else:
                raise TypeError("Current type of the loss function is not support, if you want to support it, please open a issue.")


            self._update_metrics(logits, target)
            total_loss += loss.detach() * data.size(0)
            data_len += len(data)
            
        
        end_time = time.time()
        total_loss = self._guard_all_reduce_SUM(total_loss)
        data_len = self._guard_all_reduce_SUM(data_len)
        computed_metrics['Loss'] = total_loss.item() / data_len.item()
        computed_metrics['Time'] = end_time - start_time
        for k, v in self.metrics.items():
            computed_metrics[k] = v.compute()

        if last_valid and isinstance(self.opt, SGD_NS_Overshoot_Noise):
            self.opt.move_to_overshoot()

        return computed_metrics
    
    def _compute_views_and_counts(self):
        num_chunks = 1
        view_types = ['Clean']
        if self.attack_types:
            if 'FGSM' in self.attack_types:
                if len(as_tuple(self.attack_types['FGSM'].get('eps', 8/255))) == 2:
                    num_chunks += 2
                    view_types.extend(['FGSM_Small', 'FGSM_Large'])
                else:
                    view_types.append('FGSM')
                    num_chunks += 1
            if 'FGSM_RS' in self.attack_types:
                if len(as_tuple(self.attack_types['FGSM_RS'].get('alpha', 10/255))) == 2:
                    num_chunks += 2
                    view_types.extend(['FGSM_RS_Small', 'FGSM_RS_Large'])
                else:
                    view_types.append('FGSM_RS')
                    num_chunks += 1
            if 'PGD' in self.attack_types:
                view_types.append('PGD')
                num_chunks += 1
            if 'TRADES' in self.attack_types:
                view_types.append('TRADES')
                num_chunks += 1

        if self.Trainer_config.get('Freq_View', False):
            view_types.append('Freq')
            num_chunks += 1

        return view_types, num_chunks
    

    def _sam(self, step, data, target):
        if step <= 1:
            return None
        
        logits = self.engine(data)
        loss = self.cri['Train'](logits, target)
        loss.backward()

        rho = self.Trainer_config.get("SAM", {}).get('rho', 0.05)
        use_opt = self.Trainer_config.get("SAM", {}).get('use_optim', False)
        adaptive = self.Trainer_config.get("SAM", {}).get('adaptive', False)
        norm_only = self.Trainer_config.get("SAM", {}).get('norm_only', False)
        backup = {}

        cache = []

        # Compute grad norm
        if isinstance(self.opt, (SGD_NS_Overshoot, SGD_NS_Overshoot_Noise)) and use_opt:
            grad_cache = []
            for group in self.opt.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if norm_only and p.ndim != 1:
                        continue

                    grad = self.opt.tiny_max_step(p,
                                                  momentum=group['momentum'],
                                                  dampening=group['dampening'],
                                                  rms_beta=group['rms_beta'],
                                                  nesterov=group['nesterov'],
                                                  eps=group['eps'])
                    grad_cache.append(((p.abs() if adaptive else 1.0) * grad).norm(p=2))
                    cache.append((p, p, grad))
        else:
            grad_cache = []
            for p in self.engine.parameters():
                if p.grad is not None:
                    if norm_only and p.ndim != 1:
                        continue

                    grad_cache.append(((p.abs() if adaptive else 1.0) * p.grad).norm(p=2))
                    cache.append((p, p, p.grad))

        grad_norm = torch.norm(torch.stack(grad_cache), p=2)
        grad_cache = None
        self.engine.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            for key, p, grad in cache:
                backup[key] = p.data.clone()
                e_w = rho * (((p.pow(2) if adaptive else 1.0) * grad) / (grad_norm + 1e-8))
                p.add_(e_w)

        return backup
    
    @torch.no_grad()
    def _de_sam(self, backup):
        if backup is None:
            return 

        for p in self.engine.parameters():
            if p in backup:
                p.copy_(backup[p])

    def compute_AWP_diff_and_perturbate(self, data, target, epoch=0):
        if self.awp is None:
            raise ValueError('AWP cannot be None!')

        if self.Trainer_config.get('Multi_View', False):
            data_chunks = data.chunk(self.num_chunks, dim=0)
            target_chunks = target.chunk(self.num_chunks, dim=0)

            data_view_map = dict(zip(self.view_types, data_chunks))
            target_view_map = dict(zip(self.view_types, target_chunks))

            if 'TRADES' in self.attack_types:
                keys = 'TRADES'
            elif 'PGD' in self.attack_types:
                keys = 'PGD'
            elif 'FGSM_RS' in self.attack_types:
                keys = 'FGSM_RS'
            elif 'FGSM_RS_Small' in self.attack_types:
                keys = 'FGSM_RS_Small'
            elif 'FGSM_RS_Large' in self.attack_types:
                keys = 'FGSM_RS_Large'
            elif 'FGSM' in self.attack_types:
                keys = 'FGSM'
            elif 'FGSM_Small' in self.attack_types:
                keys = 'FGSM_Small'
            elif 'FGSM_Large' in self.attack_types:
                keys = 'FGSM_Large'

            self.awp.compute_diff(data_view_map[keys], target_view_map[keys], self.engine, epoch, clean_data=data_view_map['Clean'])
        else:
            self.awp.compute_diff(data, target, self.engine, epoch)
        
        self.awp.perturbate(self.engine)
            

    def get_final_engine(self,):
        self.engine.eval()

        if self.ema is None:
            return self.engine

        # with self.ema.average_parameters(self.engine):
        self.ema.store(self.engine)
        self.ema.copy_to(self.engine)
        temp_engine = copy.deepcopy(self.engine.module)
        self.ema.restore(self.engine)
        return temp_engine


def norm(data, mu, std):
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, device=data.device).view(3, 1, 1)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=data.device).view(3, 1, 1)

    return (data - mu) / std
    

def denorm(data, mu, std):
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, device=data.device).view(3, 1, 1)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=data.device).view(3, 1, 1)

    return data * std + mu


def as_tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x,)







class Symmetric(nn.Module):
    """
    Symmetrize the last two dimensions.

    Supports:
        2D: [N, N]
        4D: [C_out, C_in, K, K]

    For Conv2d weights, this makes each spatial kernel symmetric.
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim not in (2, 4):
            raise ValueError(
                f"Symmetric only supports 2D or 4D tensors, but got shape {tuple(X.shape)}"
            )

        if X.shape[-1] != X.shape[-2]:
            raise ValueError(
                f"Last two dims must be square, but got shape {tuple(X.shape)}"
            )

        upper = torch.triu(X, diagonal=0)
        upper_no_diag = torch.triu(X, diagonal=1)

        return upper + upper_no_diag.transpose(-1, -2)

    def right_inverse(self, S: torch.Tensor) -> torch.Tensor:
        """
        Needed for torch.nn.utils.parametrize.
        Stores only the upper-triangular part.
        """
        if S.ndim not in (2, 4):
            raise ValueError(
                f"Symmetric only supports 2D or 4D tensors, but got shape {tuple(S.shape)}"
            )

        if S.shape[-1] != S.shape[-2]:
            raise ValueError(
                f"Last two dims must be square, but got shape {tuple(S.shape)}"
            )

        return torch.triu(S, diagonal=0)


def apply_parametrizations(model):
    for m in model.modules():
        if isinstance(m, (
            nn.Linear, nn.Conv1d, nn.Conv2d,
            layers.DOLinear, layers.DOConv1d, layers.DOConv2d
        )):
            if not nn.utils.parametrize.is_parametrized(m, "weight"):
                # parametrizations.spectral_norm(m, name="weight")
                if m.weight.shape[-1] == m.weight.shape[-2]:
                    nn.utils.parametrize.register_parametrization(m, "weight", Symmetric())



def remove_parametrizations(model, leave_parametrized=True):
    for m in model.modules():
        if parametrize.is_parametrized(m, "weight"):
            parametrize.remove_parametrizations(
                m,
                "weight",
                leave_parametrized=leave_parametrized
            )



def margin(logits, y):
    true = logits.gather(1, y[:, None]).squeeze(1)
    wrong = logits.clone()
    wrong.scatter_(1, y[:, None], -float("inf"))
    max_wrong = wrong.max(dim=1).values
    return true - max_wrong


def soft_margin_loss_V2(logits, target, target_margin=1.0, T=1.0, only_hard=True, focal=False, gamma=3.0): 
    prob = F.log_softmax(logits / T, dim=1) 
    true_prob = prob.gather(1, target[:, None]).squeeze(1) 

    wrong_prob = prob.clone() 
    wrong_prob.scatter_(1, target[:, None], value=-1.0) 
    max_wrong_prob = wrong_prob.max(1).values 

    margin = true_prob - max_wrong_prob
    gap = target_margin - margin
    
    # smooth hinge: approx max(0, target_margin - margin)
    loss_each = F.softplus(gap)

    if focal:
        pre_weight = 1.0 + torch.sigmoid(loss_each)
        weight = pre_weight.pow(gamma)
        weight = weight.detach()

        loss_each = weight * loss_each

    bad_mask = gap > 0
    if bad_mask.any():
        bad_margin = margin[bad_mask]
        var_loss = bad_margin.var(unbiased=False)
    else:
        var_loss = 0.0

    if only_hard:
        mask = (gap > 0)
        if mask.any():
            loss = loss_each[mask].mean() + var_loss
        else:
            loss = logits.sum() * 0.0
    else:
        loss = loss_each.mean() + var_loss
    return loss


def top_pred_correction_loss(logits, target, T=1.0, beta=10.0):
    """
    No-scatter version.

    This mainly affects samples where the top predicted class is not the true class.
    It does NOT enforce p_true - max_wrong >= margin.
    """

    prob = F.softmax(logits.float() / T, dim=1)

    true_prob = prob.gather(1, target[:, None]).squeeze(1)

    top_prob, pred = prob.max(dim=1)

    # If correct: margin = 0
    # If wrong:   margin < 0
    margin = true_prob - top_prob

    gap = -margin  # only positive when prediction is wrong

    loss_each = F.softplus(beta * gap) / beta

    wrong_mask = pred.ne(target).detach()

    if wrong_mask.any():
        return loss_each[wrong_mask].mean()

    return logits.sum() * 0.0

def soft_margin_loss_V1(logits, target, target_margin=1.0, T=1.0, focal=False, gamma=3.0):
    prob = F.log_softmax(logits / T, dim=1)

    true_prob = prob.gather(1, target[:, None]).squeeze(1)

    wrong_prob = prob.clone()
    max_wrong_prob = wrong_prob.max(1).values

    margin = true_prob - max_wrong_prob
    margin_loss = F.softplus(target_margin - margin)


    if focal:
        pre_weight = 1.0 + torch.sigmoid(margin_loss)
        weight = pre_weight.pow(gamma)
        weight = weight.detach()

        margin_loss = weight * margin_loss

    bad_mask = margin > 0
    if bad_mask.any():
        bad_margin = margin[bad_mask]
        var_loss = bad_margin.var(unbiased=False)
    else:
        var_loss = 0.0


    return margin_loss.mean() + var_loss


def make_low_mid_mask(h, w, ratio=0.6, device='cuda'):
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.fftfreq(w, device=device)

    radius = torch.sqrt(fy ** 2 + fx ** 2)
    radius = radius / radius.max().clamp_min(1e-8)

    mask = (radius <= ratio)
    return mask

def generate_freq_view(data, mu, std, sigma=0.1):
    mu = torch.tensor(mu, device=data.device).view(3,1,1)
    std = torch.tensor(std, device=data.device).view(3,1,1)

    # data = torch.fft.fft2(data.float(), dim=(-2, -1), norm='ortho')
    # amp = torch.log1p(data.abs())
    # phase = torch.angle(data)

    # amp = (amp - amp.mean(dim=(2,3), keepdim=True)) / (amp.std(dim=(2,3), keepdim=True) + 1e-7)
    # phase = phase / torch.pi

    # data = torch.cat([amp, phase], dim=0)

    data = (data - mu) / std
    # X = dct.dct_2d(data, norm='ortho')

    # B, C, H, W = X.shape
    # mask = make_low_mid_mask(H, W, ratio=0.6, device='cuda')
    # X = X * mask
    # data = dct.idct_2d(X, norm='ortho')

    noise = torch.randn_like(data) * sigma
    data = (data + noise).clamp(0, 1)

    return denorm(data, mu, std)


def make_radial_freq_mask(
    h,
    w,
    device,
    low=0.0,
    high=1.0,
    return_radius=False,
):
    """
    Create radial mask over FFT frequency coordinates.

    low/high are normalized radius values in [0, 1].

    Returns:
        mask:   [1, 1, H, W]
        radius: [H, W] if return_radius=True
    """
    assert 0.0 <= low <= 1.0, f"low must be in [0, 1], got {low}"
    assert 0.0 <= high <= 1.0, f"high must be in [0, 1], got {high}"
    assert low <= high, f"low must be <= high, got low={low}, high={high}"

    fy = torch.fft.fftfreq(h, device=device).view(h, 1)
    fx = torch.fft.fftfreq(w, device=device).view(1, w)

    radius = torch.sqrt(fx ** 2 + fy ** 2)
    radius = radius / radius.max().clamp_min(1e-8)

    mask_2d = (radius >= low) & (radius <= high)
    mask = mask_2d.float().view(1, 1, h, w)

    if return_radius:
        return mask, radius, mask_2d

    return mask


def _spectrum(feat, mode="log_amp", eps=1e-8):
    """
    Convert feature map to frequency spectrum.

    Args:
        feat: [B, C, H, W]

    mode:
        "amp"     -> amplitude spectrum
        "log_amp" -> log amplitude spectrum
        "power"   -> power spectrum
    """

    fft = torch.fft.fft2(feat.float(), dim=(-2, -1), norm="ortho")
    amp = torch.abs(fft)

    if mode == "amp":
        return amp

    elif mode == "log_amp":
        return torch.log(amp + eps)

    elif mode == "power":
        return amp.pow(2)

    else:
        raise ValueError(f"Unknown spectrum mode: {mode}")


def _wasserstein_1d_radial_from_spectrum(
    spec_adv,
    spec_clean,
    radius,
    mask_2d,
    eps=1e-8,
):
    """
    Approximate 1D Wasserstein distance over radial frequency order.

    Args:
        spec_adv:   [B, C, H, W]
        spec_clean: [B, C, H, W]
        radius:     [H, W], normalized radial frequency
        mask_2d:    [H, W], bool mask selecting frequency band

    This is not full 2D optimal transport.
    It treats spectrum energy as a 1D distribution sorted by radial frequency.
    """

    B, C, H, W = spec_adv.shape

    spec_adv_flat = spec_adv.reshape(B, C, -1)
    spec_clean_flat = spec_clean.reshape(B, C, -1)

    radius_flat = radius.reshape(-1)
    mask_flat = mask_2d.reshape(-1)

    # Select only frequencies inside [low, high].
    valid_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)

    radius_valid = radius_flat[valid_idx]
    sort_idx = torch.argsort(radius_valid)

    valid_sorted_idx = valid_idx[sort_idx]

    p = spec_adv_flat[..., valid_sorted_idx]
    q = spec_clean_flat[..., valid_sorted_idx]

    # Wasserstein needs non-negative mass.
    # amp/power are already non-negative.
    # For safety, clamp anyway.
    p = p.clamp_min(0.0) + eps
    q = q.clamp_min(0.0) + eps

    # Normalize into distributions over selected frequency band.
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)

    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)

    return torch.mean(torch.abs(cdf_p - cdf_q))


def freq_match_loss(
    feat_clean,
    feat_adv,
    spectrum_mode="log_amp",
    loss_mode="l1",
    low=0.0,
    high=1.0,
    detach_clean=True,
    eps=1e-8,
):
    """
    Frequency matching loss between clean and adversarial feature maps.

    Goal:
        Match adv feature frequency spectrum toward clean feature spectrum.

    Args:
        feat_clean: [B, C, H, W]
        feat_adv:   [B, C, H, W]

        spectrum_mode:
            "amp"
            "log_amp"
            "power"

        loss_mode:
            "l1"
            "mse"
            "wasserstein"

        low/high:
            radial frequency band to match.
            Example:
                low=0.0, high=0.35 -> match low/mid frequencies only

        detach_clean:
            If True, clean feature is treated as target/teacher.

    Returns:
        scalar loss
    """

    assert feat_clean.shape == feat_adv.shape, (
        f"Shape mismatch: clean {feat_clean.shape}, adv {feat_adv.shape}"
    )
    assert feat_clean.dim() == 4, (
        f"Expected [B, C, H, W], got {feat_clean.shape}"
    )

    if detach_clean:
        feat_clean = feat_clean.detach()

    B, C, H, W = feat_clean.shape

    spec_clean = _spectrum(feat_clean, mode=spectrum_mode, eps=eps)
    spec_adv = _spectrum(feat_adv, mode=spectrum_mode, eps=eps)

    mask, radius, mask_2d = make_radial_freq_mask(
        h=H,
        w=W,
        device=feat_clean.device,
        low=low,
        high=high,
        return_radius=True,
    )

    if loss_mode == "l1":
        spec_clean = spec_clean * mask
        spec_adv = spec_adv * mask

        return F.l1_loss(spec_adv, spec_clean)

    elif loss_mode == "mse":
        spec_clean = spec_clean * mask
        spec_adv = spec_adv * mask

        return F.mse_loss(spec_adv, spec_clean)

    elif loss_mode == "wasserstein":
        if spectrum_mode == "log_amp":
            raise ValueError(
                "For Wasserstein mode, use spectrum_mode='amp' or 'power', "
                "because Wasserstein expects non-negative spectrum mass."
            )

        return _wasserstein_1d_radial_from_spectrum(
            spec_adv=spec_adv,
            spec_clean=spec_clean,
            radius=radius,
            mask_2d=mask_2d,
            eps=eps,
        )

    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")
    



@torch.no_grad()
def log_logit_stats(logits, labels=None, name="logits", logger=None):
    # Important for AMP/bfloat16/fp16 training
    logits = logits.detach().float()

    probs = torch.softmax(logits, dim=1)

    norm = logits.norm(p=2, dim=1).float()
    max_abs = logits.abs().max(dim=1).values.float()
    conf = probs.max(dim=1).values.float()
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1).float()

    msg = (
        f"[{name}] "
        f"L2 mean={norm.mean().item():.3f}, "
        f"L2 p95={norm.quantile(0.95).item():.3f}, "
        f"L2 max={norm.max().item():.3f}, "
        f"max|logit| mean={max_abs.mean().item():.3f}, "
        f"max|logit| p95={max_abs.quantile(0.95).item():.3f}, "
        f"conf mean={conf.mean().item():.4f}, "
        f"conf p95={conf.quantile(0.95).item():.4f}, "
        f"entropy mean={entropy.mean().item():.4f}"
    )

    if labels is not None:
        labels = labels.detach()

        true_logit = logits.gather(1, labels[:, None]).squeeze(1)

        wrong_logits = logits.clone()
        wrong_logits.scatter_(1, labels[:, None], float("-inf"))
        max_wrong_logit = wrong_logits.max(dim=1).values

        margin = (true_logit - max_wrong_logit).float()

        msg += (
            f", margin mean={margin.mean().item():.3f}, "
            f"margin p05={margin.quantile(0.05).item():.3f}, "
            f"margin min={margin.min().item():.3f}"
        )

    if logger is not None:
        logger.info(msg)
    else:
        print(msg)




def unwrap_model(model):
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def canonical_param_name(name: str) -> str:
    return name.replace(".parametrizations.weight.original", ".weight")


def named_param_dict(model):
    model = unwrap_model(model)
    return {
        canonical_param_name(name): p
        for name, p in model.named_parameters()
    }


def remap_optimizer_state_by_name(optimizer, old_named_params, new_model):
    """
    old_named_params: dict[canonical_name -> old Parameter]
    collected BEFORE removing parametrization.

    new_model: model AFTER removing parametrization.
    """

    new_named_params = named_param_dict(new_model)

    old_to_new = {}

    for name, old_p in old_named_params.items():
        name = canonical_param_name(name)

        if name not in new_named_params:
            continue

        new_p = new_named_params[name]

        if old_p.shape != new_p.shape:
            print(f"Skip shape mismatch: {name}, old={old_p.shape}, new={new_p.shape}")
            continue

        old_to_new[old_p] = new_p

    # remap optimizer.state keys
    new_state = {}
    for old_p, state in optimizer.state.items():
        new_p = old_to_new.get(old_p, old_p)
        new_state[new_p] = state

    optimizer.state = new_state

    # remap param_groups
    for group in optimizer.param_groups:
        group["params"] = [
            old_to_new.get(p, p)
            for p in group["params"]
        ]

    return optimizer