# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import copy
import math
import os
import sys
import time
from argparse import Namespace
from typing import Iterable
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset, MammothDatasetWrapper
from datasets.utils.gcl_dataset import GCLDataset
from models.utils.continual_model import ContinualModel
from models.utils.future_model import FutureModel

from utils import disable_logging
from utils.checkpoints import mammoth_load_checkpoint, save_mammoth_checkpoint
from utils.loggers import log_extra_metrics, Logger
from utils.schedulers import get_scheduler
from utils.stats import track_system_stats
from utils.integrated_dissect_score import IcarlDissectandScore
from utils.concept_units_regularizer import ConceptUnitsRegularizer
import utils.similarity as similarity
import clip
try:
    import wandb
    from wandb.errors import CommError
except ImportError:
    wandb = None
    CommError = Exception  # fallback when wandb is not installed
except Exception:
    CommError = Exception  # unexpected wandb error types fall back to generic exception

HAS_TORCH_PROFILER = False
HAS_FVCORE = False
try:
    from torch.profiler import ProfilerActivity, profile
    HAS_TORCH_PROFILER = True
except ImportError:
    try:
        from torch.autograd import profiler as autograd_profiler  # type: ignore[attr-defined]
        from fvcore.nn import FlopCountAnalysis
        HAS_FVCORE = True
    except ImportError:
        HAS_FVCORE = False
        autograd_profiler = None  # type: ignore[assignment]


class RuntimeMetricsTracker:
    """
    Tracks per-task training and evaluation runtimes and estimates GFLOPs when enabled.
    """
    def __init__(self, args: Namespace):
        self.args = args
        self.enabled = bool(getattr(args, 'log_perf_metrics', 0))
        self.records = []
        self._current = None
        self._warned_no_flops = False

    def start_task(self, task_idx: int) -> None:
        if not self.enabled:
            return
        self._current = {
            'task_index': task_idx,
            'train_time_s': 0.0,
            'eval_time_s': 0.0,
            'train_gflops': None,
            'train_gflops_per_sample': None,
            '_gflops_measured': False
        }

    def time_training_block(self):
        if not self.enabled or self._current is None:
            return _NullContext()
        return _AccumulatingTimer(self._current, 'train_time_s')

    def time_evaluation(self):
        if not self.enabled or self._current is None:
            return _NullContext()
        return _AccumulatingTimer(self._current, 'eval_time_s')

    def should_profile_step(self) -> bool:
        return bool(self.enabled and self._current is not None and not self._current['_gflops_measured'])

    def run_profiled_step(self, model: ContinualModel, inputs, labels, not_aug_inputs, epoch, **extra_fields):
        """
        Executes a training step under profiling to collect GFLOPs.
        """
        if not self.should_profile_step():
            raise RuntimeError("Attempted to profile training step when profiling is unavailable or already complete.")

        gflops = None
        if HAS_TORCH_PROFILER:
            activities = [ProfilerActivity.CPU]
            if model.device.type == 'cuda':
                activities.append(ProfilerActivity.CUDA)
            with profile(activities=activities, record_shapes=False, with_flops=True) as prof:
                loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch, **extra_fields)
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)
            flop_total = 0
            for evt in prof.key_averages():
                if getattr(evt, 'flops', None):
                    flop_total += evt.flops
            gflops = flop_total / 1e9 if flop_total else 0.0
        elif HAS_FVCORE and autograd_profiler is not None:
            net_copy = copy.deepcopy(model.net).to('cpu')
            net_copy.eval()
            with torch.no_grad():
                analysis = FlopCountAnalysis(net_copy, inputs.detach().cpu())
                forward_flops = analysis.total()
            del net_copy
            backward_multiplier = 3.0
            gflops = (forward_flops * backward_multiplier) / 1e9 if forward_flops else 0.0
            with autograd_profiler.profile(use_cuda=model.device.type == 'cuda'):
                loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch, **extra_fields)
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)
            if not self._warned_no_flops:
                logging.warning("torch.profiler not available; approximating GFLOPs via fvcore with multiplier 3.0.")
                self._warned_no_flops = True
        else:
            loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch, **extra_fields)
            if not self._warned_no_flops:
                logging.warning("Runtime profiling requested but torch.profiler/fvcore are unavailable. Skipping GFLOP logging.")
                self._warned_no_flops = True
            self._current['_gflops_measured'] = True
            return loss

        self._current['_gflops_measured'] = True
        self._current['train_gflops'] = gflops
        batch_size = inputs.size(0) if hasattr(inputs, 'size') else None
        if gflops is not None and batch_size:
            self._current['train_gflops_per_sample'] = gflops / max(batch_size, 1)
        return loss

    def finish_task(self) -> None:
        if not self.enabled or self._current is None:
            return
        record = self._current.copy()
        record.pop('_gflops_measured', None)
        self.records.append(record)
        task_display = record['task_index'] + 1
        log_msg = (f"[Runtime] Task {task_display}: "
                   f"train_time={record['train_time_s']:.2f}s, "
                   f"eval_time={record['eval_time_s']:.2f}s")
        if record['train_gflops'] is not None:
            log_msg += f", iter_gflops={record['train_gflops']:.3f}"
            if record['train_gflops_per_sample'] is not None:
                log_msg += f", per_sample_gflops={record['train_gflops_per_sample']:.4f}"
        logging.info(log_msg)
        if wandb is not None and not getattr(self.args, 'nowand', 1):
            wandb.log({
                'runtime/train_time_s': record['train_time_s'],
                'runtime/eval_time_s': record['eval_time_s'],
                'runtime/iter_gflops': record['train_gflops'],
                'runtime/per_sample_gflops': record['train_gflops_per_sample'],
                'runtime/task_index': record['task_index']
            })
        self._current = None

    def summarize(self) -> None:
        if not self.enabled or not self.records:
            return
        logging.info("===== Runtime Summary =====")
        for r in self.records:
            task_display = r['task_index'] + 1
            msg = (f"Task {task_display}: train_time={r['train_time_s']:.2f}s | "
                   f"eval_time={r['eval_time_s']:.2f}s")
            if r['train_gflops'] is not None:
                msg += f" | iter_gflops={r['train_gflops']:.3f}"
            logging.info(msg)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _AccumulatingTimer:
    """
    Context manager that accumulates elapsed time into the provided record.
    """
    def __init__(self, record: dict, key: str):
        self.record = record
        self.key = key
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return None

    def __exit__(self, exc_type, exc, tb):
        end = time.perf_counter()
        self.record[self.key] = self.record.get(self.key, 0.0) + (end - self.start)
        return False


def initialize_wandb(args: Namespace) -> None:
    """
    Initializes wandb, if installed.

    Args:
        args: the arguments of the current execution
    """
    if wandb is None:
        logging.warning("Weights & Biases not installed. Continuing without wandb logging.")
        args.nowand = 1
        args.wandb_url = None
        return

    run_name = args.wandb_name if args.wandb_name is not None else args.model

    run_id = args.conf_jobnum.split('-')[0]
    name = f'{run_name}_{run_id}'
    mode = 'disabled' if os.getenv('MAMMOTH_TEST', '0') == '1' else os.getenv('WANDB_MODE', 'online')
    try:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=name, mode=mode)
        args.wandb_url = wandb.run.get_url()
    except CommError as exc:
        logging.warning("Weights & Biases communication error (%s). Disabling wandb for this run.", exc)
        args.nowand = 1
        args.wandb_url = None
    except Exception as exc:
        logging.warning("Weights & Biases failed to initialize (%s). Disabling wandb for this run.", exc)
        args.nowand = 1
        args.wandb_url = None


def _to_device(name: str, x, device):
    if isinstance(x, torch.Tensor):
        if 'label' in name.lower() or 'target' in name.lower():
            return x.to(device, dtype=torch.long)
        return x.to(device)
    return x


def _compute_uncertainty_scores(model: ContinualModel,
                                dataset,
                                device: torch.device,
                                batch_size: int,
                                num_workers: int = 4) -> torch.Tensor:
    """
    Estimates predictive uncertainty for each sample in a dataset by
    computing the entropy of the model's softmax probabilities.
    """
    was_training = model.net.training
    model.net.eval()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda'
    )

    uncertainties = []
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (list, tuple)):
                inputs = batch
            else:
                inputs = batch[0]
            inputs = inputs.to(device)
            logits = None
            try:
                logits = model.net(inputs)
            except Exception:
                logits = model(inputs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
            uncertainties.append(entropy.cpu())

    if was_training:
        model.net.train()

    return torch.cat(uncertainties)


class UncertaintyBasedSampler(torch.utils.data.Sampler[int]):
    """
    Samples indices proportionally to their uncertainty scores.
    """
    def __init__(self, scores: torch.Tensor, num_samples: int, replacement: bool = True):
        if scores.ndim != 1:
            raise ValueError('Uncertainty scores must be a 1D tensor.')
        probs = scores.clone()
        probs = probs - probs.min()
        probs = probs.clamp_min(1e-12)
        self.probabilities = probs / probs.sum()
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        sampled = torch.multinomial(self.probabilities, self.num_samples, self.replacement)
        return iter(sampled.tolist())

    def __len__(self) -> int:
        return self.num_samples


def train_single_epoch(model: ContinualModel,
                       train_loader: Iterable,
                       args: Namespace,
                       epoch: int,
                       pbar: tqdm,
                       scores=None,
                       exp=None,
                       system_tracker=None,
                       scheduler=None,
                       perf_tracker: RuntimeMetricsTracker = None,
                       class_grad_scale: torch.Tensor = None) -> int:
    """
    Trains the model for a single epoch.

    Args:
        model: the model to be trained
        train_loader: the data loader for the training set
        args: the arguments from the command line
        epoch: the current epoch
        system_tracker: the system tracker to monitor the system stats
        scheduler: the scheduler for the current epoch

    Returns:
        the number of iterations performed in the current epoch
    """
    train_iter = iter(train_loader)
    i = 0

    timing_ctx = perf_tracker.time_training_block() if perf_tracker else _NullContext()
    with timing_ctx:
        while True:
            try:
                data = next(train_iter)
            except StopIteration:
                break
            if args.debug_mode and i > model.get_debug_iters():
                break
            if args.fitting_mode == 'iters' and model.task_iteration >= model.args.n_iters:
                break

            inputs, labels, not_aug_inputs = data[0], data[1], data[2]
            inputs, labels = inputs.to(model.device), labels.to(model.device, dtype=torch.long)
            not_aug_inputs = not_aug_inputs.to(model.device)

            extra_fields = {
                train_loader.dataset.extra_return_fields[k]: _to_device(train_loader.dataset.extra_return_fields[k], data[3 + k], model.device)
                for k in range(len(data) - 3)
            }

            batch_scales = None
            if class_grad_scale is not None and getattr(args, "weighted_gradient", 0):
                # Map each label in the batch to its corresponding class-wise gradient scale.
                # class_grad_scale is expected to have shape [N_CLASSES].
                batch_scales = class_grad_scale[labels]

            if perf_tracker and perf_tracker.should_profile_step():
                loss = perf_tracker.run_profiled_step(model, inputs, labels, not_aug_inputs, epoch,
                                                      grad_scales=batch_scales, **extra_fields)
            else:
                loss = model.meta_observe(inputs, labels, not_aug_inputs, epoch=epoch,
                                          grad_scales=batch_scales, **extra_fields)
        # if args.cog_cl==1 and exp>0:
        #     print("before:", loss)
        #     loss = loss * (sum(scores))*exp
            # print("after:", loss)
            # import pdb
            # pdb.set_trace()
            assert not math.isnan(loss)

            if scheduler is not None and args.scheduler_mode == 'iter':
                scheduler.step()

            if args.code_optimization == 0 and 'cuda' in str(args.device):
                torch.cuda.synchronize()
            system_tracker()
            i += 1

            pbar.set_postfix({'loss': loss, 'lr': model.opt.param_groups[0]['lr']}, refresh=False)
            pbar.update()

    if scheduler is not None and args.scheduler_mode == 'epoch':
        scheduler.step()


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.

    Args:
        model: the module to be trained
        dataset: the continual dataset at hand
        args: the arguments of the current execution
    """
    print(args)

    is_fwd_enabled = False
    can_compute_fwd_beforetask = False
    random_results_class, random_results_task = [], []

    if not args.nowand:
        initialize_wandb(args)

    if not args.disable_log:
        logger = Logger(args, dataset.SETTING, dataset.NAME, model.NAME)

    model.net.to(model.device)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=model.device)    
    torch.cuda.empty_cache()
    perf_tracker = RuntimeMetricsTracker(args)

    with track_system_stats(logger) as system_tracker:
        results, results_mask_classes = [], []

        if args.eval_future:
            results_transf, results_mask_classes_transf = [], []

        if args.start_from is not None:
            for i in range(args.start_from):
                train_loader, _ = dataset.get_data_loaders()
                model.meta_begin_task(dataset)
                model.meta_end_task(dataset)

        if args.loadcheck is not None:
            model, past_res = mammoth_load_checkpoint(args, model)

            if not args.disable_log and past_res is not None:
                (results, results_mask_classes, csvdump) = past_res
                logger.load(csvdump)

            print('Checkpoint Loaded!')

        print(file=sys.stderr)
        start_task = 0 if args.start_from is None else args.start_from
        end_task = dataset.N_TASKS if args.stop_after is None else args.stop_after

        if args.eval_future:
            assert isinstance(model, FutureModel), "Model must be an instance of FutureModel to evaluate on future tasks"
            eval_dataset = get_dataset(args)

            # disable logging for this loop
            with disable_logging(logging.WARNING):
                for _ in range(dataset.N_TASKS):
                    eval_dataset.get_data_loaders()
                    model.change_transform(eval_dataset)
                    del eval_dataset.train_loader
        else:
            eval_dataset = dataset

        torch.cuda.empty_cache()
        for t in range(start_task, end_task):
            perf_tracker.start_task(t)
            model.net.train()
            train_loader, _ = dataset.get_data_loaders()

            if not issubclass(dataset.__class__, GCLDataset):
                assert issubclass(train_loader.dataset.__class__, MammothDatasetWrapper), "Dataset must be an instance of MammothDatasetWrapper (did you forget to call the `store_masked_loaders`?)"

            if can_compute_fwd_beforetask and is_fwd_enabled and args.enable_other_metrics:
                # try to compute accuracy at the beginning of the task
                try:
                    logging.info("Evaluating model before task (for Forward Transfer metric)...")
                    random_res_class, random_res_task = dataset.evaluate(model, dataset, last=True)  # the ugliness of this line is for backward compatibility
                    random_results_class.append(random_res_class)
                    random_results_task.append(random_res_task)
                except Exception:
                    logging.info("Could not evaluate before `begin_task`, will try after")
                    # will try after the begin_task in case the model needs to setup something
                    can_compute_fwd_beforetask = False

            model.meta_begin_task(dataset)

            # Concept units regularizer: load per-unit stats from the last completed task
            if getattr(args, 'concept_units_reg', 0) == 1:
                if not hasattr(model, '_concept_reg'):
                    model._concept_reg = ConceptUnitsRegularizer(args, model.device)
                # t is 0-based; last completed task index is t (1-based) when t >= 1
                if t >= 1:
                    try:
                        model._concept_reg.load_for_task(model, task_idx_1based=t)
                    except Exception as e:
                        logging.warning("ConceptUnitsRegularizer load failed: %s", e)

            if not can_compute_fwd_beforetask and is_fwd_enabled and args.enable_other_metrics:
                if train_loader.dataset.num_times_iterated == 0:  # compute only if the model has not been trained yet
                    try:
                        logging.info("Evaluating model before task (for Forward Transfer metric)...")
                        random_res_class, random_res_task = dataset.evaluate(model, dataset, last=True)
                        random_results_class.append(random_res_class)
                        random_results_task.append(random_res_task)
                    except Exception as e:
                        logging.error(f"Model `{model.NAME}` does not support pre-evaluation, will not compute Forward Transfer metric\n{e}")
                        is_fwd_enabled = False
                else:
                    logging.info("Model used the training data, skipping Forward Transfer metric compute")
                    is_fwd_enabled = False

            if not args.inference_only and args.n_epochs > 0:
                if t and args.enable_other_metrics:
                    accs = eval_dataset.evaluate(model, eval_dataset, last=True)
                    results[t - 1] = results[t - 1] + accs[0]
                    if dataset.SETTING == 'class-il':
                        results_mask_classes[t - 1] = results_mask_classes[t - 1] + accs[1]

                # Scheduler is automatically reloaded after each task if defined in the dataset.
                # If the model defines it, it becomes the job of the model to reload it.
                scheduler = get_scheduler(model, args, reload_optim=True) if not hasattr(model, 'scheduler') else model.scheduler

                epoch = 0
                best_ea_metric = None
                best_ea_model = None
                cur_stopping_patience = args.early_stopping_patience
                per_label_weights=None
                scores=None
                n_iterations = None
                
                if not isinstance(dataset, GCLDataset):
                    n_iterations = model.args.n_epochs * len(train_loader) if model.args.fitting_mode == 'epochs' else model.args.n_iters
                mininterval = 0.2 if n_iterations is not None and n_iterations > 1000 else 0.1
                train_pbar = tqdm(train_loader, total=n_iterations,  # train_loader is actually ignored, will update the progress bar manually
                                  disable=args.non_verbose, mininterval=mininterval)
                if args.non_verbose:
                    logging.info(f"Task {t + 1}")  # at least print the task number
                class_grad_scale = None
                if args.cog_cl==1 and t>0:
                        if args.dataset =='seq-cifar100' or  args.dataset =='seq-cifar100-224' :
                            concept_dict_path="YOUR_PATH"
                            n_classes = 10
                        if args.dataset == 'seq-core50' :
                            n_classes = 10
                            concept_dict_path="YOUR_PATH"
                        elif args.dataset == 'seq-cub200' :
                            n_classes = 20
                            concept_dict_path="YOUR_PATH"
                        elif args.dataset == 'seq-inaturalist-300' :
                            n_classes = 30
                            concept_dict_path="YOUR_PATH"    
                        elif args.dataset == 'seq-imagenet-r':
                            n_classes = 20
                            concept_dict_path="YOUR_PATH"
                        if args.dataset == 'seq-core50' :
                            each_labels = [train_loader.dataset[i][1] for i in range(len(train_loader.dataset))]
                            print("N_classes:", n_classes)
                            task_labels =args.class_order[0:n_classes]
                            up_task_labels = list(range(0,n_classes))
                        else:
                            each_labels = [train_loader.dataset[i][1].item() for i in range(len(train_loader.dataset))]
                            print("N_classes:", n_classes)
                            task_labels =args.class_order[t*n_classes:(t+1)*n_classes]
                            up_task_labels = list(range(t*n_classes,(t+1)*n_classes))
                        # print("Each_labels:", each_labels)
                        # print(args.dataset)
                        # n_classes = int(len(np.unique(np.array(each_labels))))
                        # print("N_classes:", n_classes)
                        # task_labels =args.class_order[t*n_classes:(t+1)*n_classes]
                        # up_task_labels = list(range(t*n_classes,(t+1)*n_classes))
                        model_copy=copy.deepcopy(model.net)
                        IcarlDissect_Score = IcarlDissectandScore(args=args, model=model_copy, protocol=train_loader.dataset, similarity_fn= similarity.soft_wpmi, class_order=args.class_order, curent_experience=t, current_experience_classes=task_labels, device=model.device)
                        IcarlDissect_Score.dissect("ours_saved_activation")
                        scores = IcarlDissect_Score.scoring_function(clip_model=clip_model, filtered_concept_set=None, next_exp_concept_set_path=concept_dict_path, indices=task_labels, device=model.device)

                        # Build per-class gradient scaling factors from concept similarity scores
                        if getattr(args, "weighted_gradient", 0) and scores is not None:
                            with torch.no_grad():
                                scores_t = torch.tensor(scores, dtype=torch.float, device=model.device)
                                # Normalize scores to [0, 1] to interpret them as similarities
                                scores_t = (scores_t - scores_t.min()) / (scores_t.max() - scores_t.min() + 1e-8)

                                # Map similarity to gradient scale: high similarity -> small update
                                # low similarity -> larger update. Keep scales in [min_scale, max_scale].
                                min_scale = 0.2
                                max_scale = 1.0
                                class_grad_scale = torch.ones(model.N_CLASSES, device=model.device)
                                for cls_idx, sim in zip(up_task_labels, scores_t):
                                    # dissim = 1.0 - sim
                                    dissim = sim
                                    scale = min_scale + (max_scale - min_scale) * dissim
                                    class_grad_scale[cls_idx] = scale
                while True:

                    if args.cog_cl==1 and t>0:
                        # if args.dataset=="perm-mnist":
                        #     each_labels = [train_loader.dataset[i][1] for i in range(len(train_loader.dataset))]
                        # else:
                        # each_labels = [train_loader.dataset[i][1].item() for i in range(len(train_loader.dataset))]
                        # # print("Each_labels:", each_labels)
                        # # print(args.dataset)
                        # n_classes = int(len(np.unique(np.array(each_labels))))
                        # task_labels =args.class_order[t*10:(t+1)*10]
                        # up_task_labels = list(range(t*10,(t+1)*10))
                        # print("Task_labels:", task_labels)  
                        # print("Up_task_labels:", up_task_labels)
                        # if args.dataset=="perm-mnist":
                        #     task_labels = args.class_order
                        #     up_task_labels = list(range(0,10))

                        # if args.dataset=="seq-cifar10":
                        #     task_labels = args.class_order[t*2:(t+1)*2]
                        #     up_task_labels = list(range(t*2,(t+1)*2)
                        #     )    
                        
                        # print("Scores:", scores)
                        # print("labels",up_task_labels)
                        # scores = scores.tolist()
                        scores = torch.tensor(scores, dtype=torch.float)
                        # scores = rank_based_normalize(scores)
                        scores = [score.item() for score in scores]
                            
                        scores_dict = dict(zip(up_task_labels, scores))
                        # print("Scores_dict:", scores_dict)
                        #SACK(W->U)
                        if args.sack_scores_type ==0:
                            per_label_weights  = [(scores_dict[j] *(1-  (epoch/(args.n_epochs-1)) ) )  + ((epoch/(args.n_epochs-1))*(1/n_classes)) for j in up_task_labels]
                        #SACK(U->W)
                        elif args.sack_scores_type ==1:
                            per_label_weights = [((1/n_classes) *(1-  (epoch/(args.n_epochs-1)) ) )  + ((epoch/(args.n_epochs-1))* scores_dict[j]) for j in up_task_labels]
                        #random scores
                        elif args.sack_scores_type ==2:
                            random_scores = [np.random.uniform(0, 1) for j in task_labels]
                            random_scores_dict = dict(zip(task_labels, random_scores))
                            # print("random scores:",random_scores_dict)
                            per_label_weights = [((1/n_classes) *(1-  (epoch/(args.n_epochs-1)) ) )  + ((epoch/(args.n_epochs-1))* random_scores_dict[j]) for j in up_task_labels]
                            
                        per_label_weights_dict = dict(zip(up_task_labels, per_label_weights))
                        sample_weights = []
                        # I am assigning the weights to the each samples based on the scores of the concepts
                        for label in each_labels:
                            if label in up_task_labels:
                                # print("0:yess")
                                sample_weights.append(per_label_weights_dict[label])
                            else:
                                sample_weights.append(1)
                        base_dataset = train_loader.dataset
                        num_workers = getattr(args, "num_workers", 4)
                        if num_workers is None:
                            num_workers = 4
                        if getattr(args, "use_uncertainty_sampling", 0):
                            unc_batch_size = getattr(args, "uncertainty_eval_batch_size", None) or args.batch_size
                            uncertainties = _compute_uncertainty_scores(
                                model=model,
                                dataset=base_dataset,
                                device=model.device,
                                batch_size=unc_batch_size,
                                num_workers=num_workers
                            )
                            sampler = UncertaintyBasedSampler(uncertainties, len(base_dataset))
                            train_loader = torch.utils.data.DataLoader(
                                base_dataset,
                                batch_size=args.batch_size,
                                sampler=sampler,
                                num_workers=num_workers
                            )
                        else:
                            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
                            train_loader = torch.utils.data.DataLoader(
                                base_dataset,
                                batch_size=args.batch_size,
                                sampler=sampler,
                                num_workers=num_workers
                            )

                    model.begin_epoch(epoch, dataset)

                    train_pbar.set_description(f"Task {t + 1} - Epoch {epoch + 1}")

                    train_single_epoch(model, train_loader, args, pbar=train_pbar, epoch=epoch,
                                       system_tracker=system_tracker, scheduler=scheduler, scores=None,exp=t,
                                       perf_tracker=perf_tracker, class_grad_scale=class_grad_scale)

                    model.end_epoch(epoch, dataset)

                    epoch += 1
                    if args.fitting_mode == 'epochs' and epoch >= model.args.n_epochs:
                        break
                    elif args.fitting_mode == 'iters' and model.task_iteration >= model.args.n_iters:
                        break
                    elif args.fitting_mode == 'early_stopping' and epoch % args.early_stopping_freq == 0 and epoch > 0:
                        epoch_accs, _, epoch_loss = eval_dataset.evaluate(model, eval_dataset, return_loss=True, last=True)

                        if args.early_stopping_metric == 'accuracy':
                            ea_metric = np.mean(epoch_accs)  # Higher accuracy is better
                        elif args.early_stopping_metric == 'loss':
                            ea_metric = -epoch_loss  # Lower loss is better
                        else:
                            raise ValueError(f'Unknown early stopping metric {args.early_stopping_metric}')

                        # Higher accuracy is better
                        if best_ea_metric is not None and ea_metric - best_ea_metric < args.early_stopping_epsilon:
                            cur_stopping_patience -= args.early_stopping_freq
                            if cur_stopping_patience <= 0:
                                print(f"\nEarly stopping at epoch {epoch} with metric {abs(ea_metric)}", file=sys.stderr)
                                model.load_state_dict({k: v.to(model.device) for k, v in best_ea_model.items()})
                                break
                            print(f"\nNo improvement at epoch {epoch} (best {abs(best_ea_metric)} | current {abs(ea_metric)}). "
                                  f"Waiting for {cur_stopping_patience} epochs to stop.", file=sys.stderr)
                        else:
                            print(f"\nFound better model with metric {abs(ea_metric)} at epoch {epoch}. "
                                  f"Previous value was {abs(best_ea_metric) if best_ea_metric is not None else 'None'}", file=sys.stderr)
                            best_ea_metric = ea_metric
                            best_ea_model = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                            cur_stopping_patience = args.early_stopping_patience

                    if args.eval_epochs is not None and (epoch > 0 or args.eval_epochs) and epoch % args.eval_epochs == 0 and epoch < model.args.n_epochs:
                        epoch_accs = eval_dataset.evaluate(model, eval_dataset)

                        eval_dataset.log(args, logger, epoch_accs, t, dataset.SETTING, epoch=epoch)

                train_pbar.close()

            model.meta_end_task(dataset)

            with perf_tracker.time_evaluation() if perf_tracker.enabled else _NullContext():
                accs = eval_dataset.evaluate(model, eval_dataset)

            if args.eval_future and t < dataset.N_TASKS - 1:
                transf_accs = accs[0][t + 1:], accs[1][t + 1:]
                accs = accs[0][:t + 1], accs[1][:t + 1]
                results_transf.append(transf_accs[0])
                results_mask_classes_transf.append(transf_accs[1])

            logged_accs = eval_dataset.log(args, logger, accs, t, dataset.SETTING)

            if dataset.SETTING != 'biased-class-il':
                results.append(accs[0])
                results_mask_classes.append(accs[1])
            else:
                results.append(logged_accs[0])  # avg
                results_mask_classes.append(logged_accs[1])  # worst

            if args.eval_future:
                avg_transf = np.mean([np.mean(task_) for task_ in results_transf])
                print(f"Transfer Metrics  -  AVG Transfer {avg_transf:.2f}", file=sys.stderr)
                if t < dataset.N_TASKS - 1:
                    eval_dataset.log(args, logger, transf_accs, t, dataset.SETTING, future=True)

            if args.savecheck:
                save_mammoth_checkpoint(t, end_task, args,
                                        model,
                                        results=[results, results_mask_classes, logger.dump()],
                                        optimizer_st=model.opt.state_dict() if hasattr(model, 'opt') else None,
                                        scheduler_st=scheduler.state_dict() if scheduler is not None else None)
            perf_tracker.finish_task()

        if args.validation:
            # Final evaluation on the real test set
            print("Starting final evaluation on the real test set...", file=sys.stderr)
            del dataset
            args.validation = None
            args.validation_mode = 'current'

            final_dataset = get_dataset(args)
            for _ in range(final_dataset.N_TASKS):
                final_dataset.get_data_loaders()
            accs = final_dataset.evaluate(model, final_dataset)

            final_dataset.log(args, logger, accs, 'final', final_dataset.SETTING, prefix="FINAL")

        if args.enable_other_metrics:
            bwt, bwt_mask_class = logger.add_bwt(results, results_mask_classes)
            log_extra_metrics(args, bwt, bwt_mask_class, 'Backward Transfer', t)
            forgetting, forgetting_mask_class = logger.add_forgetting(results, results_mask_classes)
            log_extra_metrics(args, forgetting, forgetting_mask_class, 'Forgetting', t)
            if is_fwd_enabled:
                fwt, fwt_mask_class = logger.add_fwt(results, random_results_class,
                                                     results_mask_classes, random_results_task)
                log_extra_metrics(args, fwt, fwt_mask_class, 'Forward Transfer', t)
            else:
                logging.warning("Forward Transfer metric incompatible with the current model, skipped.")

        system_tracker.print_stats()

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()
    perf_tracker.summarize()
