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

from Deep_Optimization.Train.log import get_logger
from Deep_Optimization.Train.utils_train import warmup, build_CUDA_Graph, wrap_model_prepare_qat, Setup_Criterion, EMA
from Deep_Optimization.Train.utils_ddp import rank0, setup_ddp


from Deep_Optimization.Activation_Compression.controller import Controller
import Deep_Optimization.Activation_Compression.modules.layers  as layers
from Deep_Optimization.Activation_Compression.modules.normalization.norm_layer_utils import convert_do_sync_batchnorm

from Deep_Optimization.Adversarial_Attack.FGSM import FGSM_attack, PGD_attack
from Deep_Optimization.Optimizer.SGD_geometry import SGD_NS_Overshoot, SGD_NS_Overshoot_Noise

import torchattacks

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
                 Adversarial_Attack: Dict = None,
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
        self.ema = ema
        self.num_epochs = num_epochs
        self.grad_acc_step = grad_acc_step
        self.grad_norm_clip = grad_norm_clip
        self.device = device.type if isinstance(device, torch.device) else device
        self.teacher_model = teacher_model

        self.cuda_timer_start = torch.cuda.Event(enable_timing=True)
        self.cuda_timer_end = torch.cuda.Event(enable_timing=True)


        # Check confliction
        assert (self.QAT != self.amp_enable) or (not self.QAT and not self.amp_enable), "Please choose either QAT=True, or amp_enable=True!"
        assert not (self.DS_config is not None and self.DDP_config is not None) , "Please choose either Deep Speed, or DDP!"
        assert not (self.DS_config is not None and self.ACT_config is not None), "Please choose either Deep Speed, or Activation Compression!"
        assert not (self.ACT_config is not None and self.train_dataloader is None), "Please also pass the train_dataloader when ACT is enabled!"

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
        # If TP2 AMP enabled, auto check the best cast dtype
        # ---------------------------------------------------
        if self.DS_config is None and amp_enable and self.device.startswith('cuda'):
            major, _ = torch.cuda.get_device_capability(torch.device(device))
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
        return self._training(epoch_idx, turned_on=turned_on, epoch=epoch)
    
    def valid(self, dataloader, attack = False, rs=False, target_top2=False, PGD=False, num_iters=7, eps=8/255, random_eps=8/255, alpha=10/255, use_auto=False, last_valid=False):
        return self._validation(dataloader, attack=attack, rs=rs, target_top2=target_top2, PGD=PGD, num_iters=num_iters, eps=eps, random_eps=random_eps, alpha=alpha, use_auto=use_auto, last_valid=last_valid)
    
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
            if isinstance(m, (nn.Linear, nn.Conv2d, layers.DOLinear, layers.DOConv2d)):
                modules.append(m)
        
        modules[-1].register_forward_hook(forward_hook())

    
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

        else:
            engine = model
            if rank0():
                logger.info("Model Wrap Type: Raw")

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
                        if epoch >= EMA_teacher_controller.get("Start_Epoch", 6):
                            with torch.no_grad():
                                with self.ema.average_parameters(self.engine):
                                    ema_logits = self.engine(data)

                        if self.Trainer_config.get("SAM", False):
                            backup = self._sam(step, data, target)

                        logits = self.engine(data)

                        if ema_logits is not None and EMA_teacher_controller.get("full_logits", False):
                            log_p = F.log_softmax(logits, dim=1)
                            q = F.softmax(ema_logits.detach(), dim=1)
                            dl_loss = F.kl_div(log_p, q, reduction='batchmean')
                    else:
                        if self.Trainer_config.get("SAM", False):
                            backup = self._sam(step, data, target)

                        logits = self.engine(data)

                    if isinstance(self.cri, Setup_Criterion):
                        ori_loss = self.cri['Train'](logits, labels=target) if self.teacher_model is None else self.cri['Train'](logits, labels=target, teacher_logits=teacher_logits)
                    # elif isinstance(self.cri, nn.CrossEntropyLoss):
                    else:
                        # ori_loss = self.cri['Train'](logits, target) + soft_margin_loss(logits, target)
                        ori_loss = self.cri['Train'](logits, target)

                        if self.Trainer_config.get("Multi_View", False):
                            num_chunks = 1
                            view_types = ['Clean']
                            attack_types = self.Adversarial_Attack.get('Attack_Type', {})
                            if 'FGSM' in attack_types:
                                if len(as_tuple(attack_types['FGSM'].get('eps', 8/255))) == 2:
                                    num_chunks += 2
                                    view_types.extend(['FGSM_Small', 'FGSM_Large'])
                                else:
                                    view_types.append('FGSM')
                                    num_chunks += 1
                            if 'FGSM_RS' in attack_types:
                                if len(as_tuple(attack_types['FGSM_RS'].get('alpha', 10/255))) == 2:
                                    num_chunks += 2
                                    view_types.extend(['FGSM_RS_Small', 'FGSM_RS_Large'])
                                else:
                                    view_types.append('FGSM_RS')
                                    num_chunks += 1
                            if 'PGD' in attack_types:
                                view_types.append('PGD')
                                num_chunks += 1
                                
                            assert logits.size(0) == target.size(0)
                            assert logits.size(0) % num_chunks == 0, (
                                f"Bad multiview chunking: logits={logits.size(0)}, "
                                f"target={target.size(0)}, num_chunks={num_chunks}, views={view_types}"
                            )

                            logits_chunks = logits.chunk(num_chunks, dim=0)
                            # target_chunks = target.chunk(num_chunks, dim=0)

                            logits_view_map = dict(zip(view_types, logits_chunks))
                            # target_view_map = dict(zip(view_types, target_chunks))

                            if ema_logits is not None and EMA_teacher_controller.get("clean_logits", False):
                                target_ema_logits = ema_logits.chunk(num_chunks, dim=0)
                                kl_clean_logits = target_ema_logits[0]
                            else:
                                kl_clean_logits = logits_view_map['Clean']
                            
                            T = float(self.Adversarial_Attack.get("KL_temperature", 1.0))
                            assert T > 0, f"KL_temperature must be > 0, got {T}"
                            if "FGSM" in attack_types:
                                if 'FGSM_Small' in view_types:
                                    attack_key = 'FGSM_Small'
                                else:
                                    attack_key = 'FGSM'
                                kl1 = F.kl_div(
                                    F.log_softmax(logits_view_map[attack_key] / T, dim=1),
                                    F.softmax(kl_clean_logits.detach() / T, dim=1),
                                    reduction='batchmean'
                                )
                                ori_loss += kl1 * (T * T)
                            if "FGSM_RS" in attack_types:
                                if 'FGSM_RS_Small' in view_types:
                                    attack_key = 'FGSM_RS_Small'
                                else:
                                    attack_key = 'FGSM_RS'
                                kl2 = F.kl_div(
                                    F.log_softmax(logits_view_map[attack_key] / T, dim=1),
                                    F.softmax(kl_clean_logits.detach() / T, dim=1),
                                    reduction='batchmean'
                                )
                                ori_loss += kl2 * (T * T)
                            if "PGD" in attack_types:
                                kl3 = F.kl_div(
                                    F.log_softmax(logits_view_map['PGD'] / T, dim=1),
                                    F.softmax(kl_clean_logits.detach() / T, dim=1),
                                    reduction='batchmean'
                                )
                                kl_weight = attack_types.get('PGD', {}).get('kl_weight', 1.0)
                                ori_loss += kl3 * (T * T) * kl_weight



                        if self.ema is not None and isinstance(self.ema, EMA) and len(self.Trainer_config.get("EMA_Proximal_Loss", {})) != 0 and epoch >= self.Trainer_config.get("EMA_Proximal_Loss", {}).get("Start_Epoch", 6):
                            rho = self.Trainer_config.get("EMA_Proximal_Loss", {}).get("rho", 5e-4)

                            prox = self.ema.prox_term(self.engine)
                            ori_loss += 0.5 * rho * prox

                        if len(self.Trainer_config.get("L1_Sparse_Loss", {})) != 0:
                            trust_ratio = self.Trainer_config['L1_Sparse_Loss'].get('trust_ratio', 0.001)
                            l1_s_loss = self.l1_act.abs().mean()
                            ori_loss += trust_ratio * l1_s_loss


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

                    if self.Trainer_config.get("SAM", False):
                            self._de_sam(backup)

                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.engine.parameters(), max_norm=1.0)

                    if self.Trainer_config.get("SAM", False):
                        self._de_sam(backup)

                    self.opt.step()

        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update_parameters(self.engine)
        return logits, ori_loss

    @torch.no_grad()
    def _update_metrics(self, logits, target):
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


                attack_types = self.Adversarial_Attack.get("Attack_Type", {})
                if self.Trainer_config.get("Multi_View", False) and "FGSM" in attack_types and "FGSM_RS" in attack_types:
                    both = True
                else:
                    both = False
                
                if "FGSM" in attack_types or "FGSM_RS" in attack_types:
                    eps = as_tuple(attack_types.get('FGSM', {}).get('eps', 8/255))
                    alpha = as_tuple(attack_types.get('FGSM_RS', {}).get('alpha', 10/255))
                    random_eps = as_tuple(attack_types.get('FGSM_RS', {}).get('random_eps', 8/255))


                    if both:
                        len_target = len(eps) + len(alpha)
                    else:
                        len_target = max(len(eps), len(alpha))

                    FGSM_attacked_data_chunks = FGSM_attack(self.engine, self.cri['Valid'], data, target, 
                                                                  both=both, 
                                                                  eps=eps, random_eps=random_eps, alpha=alpha,
                                                                  mu=mu,
                                                                  std=std,
                                                                  target_top2=False,
                                                                  device=self.device)
                else:
                    FGSM_attacked_data_chunks = None
                    len_target = 1
                
                if "PGD" in attack_types:
                    random_eps = attack_types.get('PGD', {}).get('random_eps', 8/255)
                    alpha = attack_types.get('PGD', {}).get('alpha', 2/255)
                    steps = attack_types.get('PGD', {}).get('steps', 7)

                    PGD_attacked_data_chunks = PGD_attack(self.engine, self.cri['Valid'], data, target,
                                                    random_eps=random_eps, alpha=alpha, num_iters=steps,
                                                    mu=mu, std=std,
                                                    target_top2=False,
                                                    device=self.device
                                                    )
                else:
                    PGD_attacked_data_chunks = None
                
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


            grad_step = ((step + 1) % self.grad_acc_step == 0 or (step + 1) == len(self.train_dataloader))

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
                                                            eps=as_tuple(eps), alpha=as_tuple(alpha),
                                                            target_top2=target_top2,
                                                            device=self.device)
                                if PGD:
                                    data = PGD_attack(self.engine, self.cri['Valid'], data, target, num_iters=num_iters,
                                                            target_top2=target_top2, random_eps=random_eps, 
                                                            alpha=alpha, mu=mu, std=std,
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
                                                        alpha=alpha, mu=mu, std=std,
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

    def _sam(self, step, data, target):
        if step <= 1:
            return None

        logits = self.engine(data)
        loss = self.cri['Train'](logits, target)
        loss.backward()

        rho = 0.1
        backup = {}

        # compute grad norm
        if not isinstance(self.opt, (SGD_NS_Overshoot, SGD_NS_Overshoot_Noise)):
            grad_norm = torch.norm(
                torch.stack([
                    p.grad.norm(p=2)
                    for p in self.engine.parameters()
                    if p.grad is not None
                ])
            )
        else:
            cache_grad = []
            for group in self.opt.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = self.opt.tiny_max_step(p, momentum=group['momentum'], 
                                                  dampening=group['dampening'], 
                                                  rms_beta=group['rms_beta'], 
                                                  nesterov=group['nesterov'], 
                                                  eps=group['eps'])
                    
                    cache_grad.append(grad.norm(p=2))
            grad_norm = torch.norm(torch.stack(cache_grad))

        # perturb weights
        if not isinstance(self.opt, (SGD_NS_Overshoot, SGD_NS_Overshoot_Noise)):
            for name, p in self.engine.named_parameters():
                if p.grad is None:
                    continue

                backup[name] = p.data.clone()

                e_w = p.grad / (grad_norm + 1e-12)
                p.data.add_(rho * e_w)
        else:
            for group in self.opt.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    backup[p] = p.data.clone()

                    grad = self.opt.tiny_max_step(p, momentum=group['momentum'], 
                                                  dampening=group['dampening'], 
                                                  rms_beta=group['rms_beta'], 
                                                  nesterov=group['nesterov'], 
                                                  eps=group['eps'])
                    
                    e_w = grad / (grad_norm + 1e-12)
                    p.data.add_(rho * e_w)

        self.engine.zero_grad()

        return backup
    
    def _de_sam(self, backup):
        if backup is not None and not isinstance(self.opt, (SGD_NS_Overshoot, SGD_NS_Overshoot_Noise)):
            for name, p in self.engine.named_parameters():
                if name in backup:
                    p.data = backup[name]

        if backup is not None and isinstance(self.opt, (SGD_NS_Overshoot, SGD_NS_Overshoot_Noise)):
            for p in self.engine.parameters():
                if p in backup:
                    p.data = backup[p]

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


def soft_margin_loss(logits, target, target_margin=3.0):
    true_logits = logits.gather(1, target[:, None]).squeeze(1)

    wrong_logits = logits.clone()
    wrong_logits.scatter(1, target[:, None], value=1e-9)
    max_wrong_logits = wrong_logits.max(1).values

    margin = true_logits - max_wrong_logits
    margin_loss = F.softplus(target_margin - margin)

    return margin_loss.mean()