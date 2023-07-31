from torch_xla import runtime as xr
import args_parse
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

import os
import schedulers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

import torch.distributed as dist
import torch_xla.distributed.xla_backend

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops



SUPPORTED_MODELS = ['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small']

MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': 'vit_small',
    },
    '--test_set_batch_size': {
        'type': int,
    },
    '--lr_scheduler_type': {
        'type': str,
    },
    '--lr_scheduler_divide_every_n_epochs': {
        'type': int,
    },
    '--lr_scheduler_divisor': {
        'type': int,
    },
    '--test_only_at_end': {
        'action': 'store_true',
    },
    '--ddp': {
        'action': 'store_true',
    },
    # Use pjrt:// init_method instead of env:// for `torch.distributed`.
    # Required for DDP on TPU v2/v3 when using PJRT.
    '--pjrt_distributed': {
        'action': 'store_true',
    },
    '--profile': {
        'action': 'store_true',
    },
    '--persistent_workers': {
        'action': 'store_true',
    },
    '--prefetch_factor': {
        'type': int,
    },
    '--loader_prefetch_size': {
        'type': int,
    },
    '--device_prefetch_size': {
        'type': int,
    },
    '--host_to_device_transfer_threads': {
        'type': int,
    },
    '--use_optimized_kwargs': {
        'type': str,
    },
}

FLAGS = args_parse.parse_common_options(
    datadir='~/imagenet',
    patch_size=None,
    out_dim=None,
    norm_last_layer=None,
    momentum_teacher=None,
    use_bn_in_head=None,
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    warmup_teacher_temp=None,
    teacher_temp=None,
    warmup_teacher_temp_epochs=None,
    target_accuracy=None,
    profiler_port=9012,
    local_crops_number=None,
    global_crops_scale=None,
    local_crops_scale=None,
    output_dir=None,
    saveckp_freq=None,
    num_workers=None,
    dist_url=None,
    local_rank=None,
    fp16_scaler=None,
    opts=MODEL_OPTS.items(),
)

DEFAULT_KWARGS = dict(
    batch_size=128,
    patch_size=16,
    out_dim=65536,
    norm_last_layer=True,
    momentum_teacher=0.996,
    use_bn_in_head=False,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    warmup_teacher_temp=0.04,
    teacher_temp=0.04,
    warmup_teacher_temp_epochs=0,
    target_accuracy=0.0,
    persistent_workers=False,
    prefetch_factor=16,
    loader_prefetch_size=8,
    device_prefetch_size=4,
    local_crops_number=8,
    global_crops_scale=(0.4, 1.),
    local_crops_scale=(0.05, 0.4),
    num_workers=8,
    output_dir="./",
    saveckp_freq=20,
    dist_url="env://",
    local_rank="local_rank",
    host_to_device_transfer_threads=1,
)

#  Best config to achieve peak performance based on TPU version
#    1. It is recommended to use this config in conjuntion with XLA_USE_BF16=1 Flag.
#    2. Hyperparameters can be tuned to further improve the accuracy.
#  usage: python3 /usr/share/pytorch/xla/test/test_train_mp_imagenet.py --model=resnet50 \
#         --fake_data --num_epochs=10 --log_steps=300 \
#         --profile   --use_optimized_kwargs=tpuv4  --drop_last
OPTIMIZED_KWARGS = {
    'tpuv4':
        dict(
            batch_size=128,
            test_set_batch_size=128,
            num_epochs=18,
            momentum=0.9,
            lr=0.1,
            target_accuracy=0.0,
            persistent_workers=True,
            prefetch_factor=32,
            loader_prefetch_size=128,
            device_prefetch_size=1,
            num_workers=16,
            host_to_device_transfer_threads=4,
        )
}

MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS/OPTIMIZED_KWARGS, or add them to the dict
    # if they don't exist.
    'resnet50':
        dict(
            OPTIMIZED_KWARGS.get(FLAGS.use_optimized_kwargs, DEFAULT_KWARGS),
            **{
                'lr': 0.5,
                'lr_scheduler_divide_every_n_epochs': 20,
                'lr_scheduler_divisor': 5,
                'lr_scheduler_type': 'WarmupAndExponentialDecayScheduler',
            })
}

# Set any args that were not explicitly given by the user.
default_value_dict = MODEL_SPECIFIC_DEFAULTS.get(FLAGS.model, DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
    if getattr(FLAGS, arg) is None:
        setattr(FLAGS, arg, value)


def get_model_property(key):
    default_model_property = {
        'img_dim': 224,
        'model_fn': getattr(torchvision.models, FLAGS.model)
    }
    model_properties = {
        'inception_v3': {
        'img_dim': 299,
        'model_fn': lambda: torchvision.models.inception_v3(aux_logits=False)
        },
    }
    model_fn = model_properties.get(FLAGS.model, default_model_property)[key]
    return model_fn


def _train_update(device, step, loss, tracker, epoch, writer):
    test_utils.print_training_update(
        device,
        step,
        loss.item(),
        tracker.rate(),
        tracker.global_rate(),
        epoch,
        summary_writer=writer)

def train_imagenet():
    if FLAGS.pjrt_distributed:
        import torch_xla.experimental.pjrt_backend
        dist.init_process_group('xla', init_method='pjrt://')
    elif FLAGS.ddp:
        dist.init_process_group(
            'xla', world_size=xm.xrt_world_size(), rank=xm.get_ordinal())

    print('==> Preparing data..')
    img_dim = get_model_property('img_dim')
    if FLAGS.fake_data:
        train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
        train_loader = xu.SampleGenerator(
            data=(torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
                torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
            sample_count=train_dataset_len // FLAGS.batch_size //
            xm.xrt_world_size())
        test_loader = xu.SampleGenerator(
            data=(torch.zeros(FLAGS.test_set_batch_size, 3, img_dim, img_dim),
                    torch.zeros(FLAGS.test_set_batch_size, dtype=torch.int64)),
                    sample_count=50000 // FLAGS.batch_size // xm.xrt_world_size())
    else:
        transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        )
        dataset = datasets.ImageFolder(args.data_path, transform=transforms.Compose([transforms.Resize((256,256)),transform]))
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=FLAGS.batch_size_per_gpu,
            num_workers=FLAGS.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Data loaded: there are {len(dataset)} images.")
        
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.datadir, 'train'),
            transforms.Compose([
                transforms.Resize((img_dim,img_dim)),
                transform
            ]))
        train_dataset_len = len(train_dataset.imgs)
        resize_dim = max(img_dim, 224)
        test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.datadir, 'validation'),
            # Matches Torchvision's eval transforms except Torchvision uses size
            # 256 resize for all models both here and in the train loader. Their
            # version crashes during training on 299x299 images, e.g. inception.
            transforms.Compose([
                transforms.Resize((resize_dim,resize_dim)),
                transform
            ]))

        train_sampler, test_sampler = None, None
        if xm.xrt_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=FLAGS.batch_size,
            sampler=train_sampler,
            drop_last=FLAGS.drop_last,
            shuffle=False if train_sampler else True,
            num_workers=FLAGS.num_workers,
            persistent_workers=FLAGS.persistent_workers,
            prefetch_factor=FLAGS.prefetch_factor)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=FLAGS.test_set_batch_size,
            sampler=test_sampler,
            drop_last=FLAGS.drop_last,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            persistent_workers=FLAGS.persistent_workers,
            prefetch_factor=FLAGS.prefetch_factor)

    torch.manual_seed(42)

    device = xm.xla_device()
    
    student = get_model_property('model_fn')().to(device)
    teacher = get_model_property('model_fn')().to(device)
    embed_dim = student.fc.weight.shape[1]

    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # Initialization is nondeterministic with multiple threads in PjRt.
    # Synchronize model parameters across replicas manually.
    if xr.using_pjrt():
        xm.broadcast_master_param(model)

    if FLAGS.ddp:
        student = DDP(student, gradient_as_bucket_view=True, broadcast_buffers=False)
        teacher = DDP(teacher, gradient_as_bucket_view=True, broadcast_buffers=False)

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        FLAGS.out_dim,
        FLAGS.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        FLAGS.warmup_teacher_temp,
        FLAGS.teacher_temp,
        FLAGS.warmup_teacher_temp_epochs,
        FLAGS.epochs,
    )

    if FLAGS.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(FLAGS.logdir)
    optimizer = optim.SGD(
        model.parameters(),
        lr=FLAGS.lr,
        momentum=FLAGS.momentum,
        weight_decay=1e-4)
    num_training_steps_per_epoch = train_dataset_len // (
        FLAGS.batch_size * xm.xrt_world_size())
    
    
    """lr_scheduler = schedulers.wrap_optimizer_with_scheduler(
        optimizer,
        scheduler_type=getattr(FLAGS, 'lr_scheduler_type', None),
        scheduler_divisor=getattr(FLAGS, 'lr_scheduler_divisor', None),
        scheduler_divide_every_n_epochs=getattr(
        FLAGS, 'lr_scheduler_divide_every_n_epochs', None),
        num_steps_per_epoch=num_training_steps_per_epoch,
        summary_writer=writer)"""
    
    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        FLAGS.lr * (FLAGS.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        FLAGS.min_lr,
        FLAGS.epochs, len(data_loader),
        warmup_epochs=FLAGS.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        FLAGS.weight_decay,
        FLAGS.weight_decay_end,
        FLAGS.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(FLAGS.momentum_teacher, 1,
                                               FLAGS.epochs, len(data_loader))
    
    loss_fn = dino_loss

    if FLAGS.profile:
        server = xp.start_server(FLAGS.profiler_port)
    
    #Train function
    def train_loop_fn(loader, epoch):
        tracker = xm.RateTracker()
        model.train()
        for step, (data, target) in enumerate(loader):
            with xp.StepTrace('train_imagenet'):
                with xp.Trace('build_graph'):
                    teacher_output = teacher(data[:2])
                    student_output = student(data)
                    loss = dino_loss(student_output, teacher_output, step)
                
                    if not math.isfinite(loss.item()):
                        print("Loss is {}, stopping training".format(loss.item()), force=True)
                        sys.exit(1)
                
                    optimizer.zero_grad()
                    
                    param_norms = None

                    if fp16_scaler is None:
                        loss.backward()
                        if args.clip_grad:
                            param_norms = utils.clip_gradients(student, args.clip_grad)
                        utils.cancel_gradients_last_layer(epoch, student,
                                                        args.freeze_last_layer)
                        if FLAGS.ddp:
                            optimizer.step()
                        else:
                            xm.optimizer_step(optimizer)
                            tracker.add(FLAGS.batch_size)
                    else:
                        fp16_scaler.scale(loss).backward()
                        if args.clip_grad:
                            fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                            param_norms = utils.clip_gradients(student, args.clip_grad)
                        utils.cancel_gradients_last_layer(epoch, student,
                                                        args.freeze_last_layer)
                        fp16_scaler.step(optimizer)
                        fp16_scaler.update()


                    with torch.no_grad():
                        m = momentum_schedule[step]  # momentum parameter
                        for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                    
                    """if lr_scheduler:
                        lr_scheduler.step()"""
                if step % FLAGS.log_steps == 0:
                    xm.add_step_closure(
                        _train_update, args=(device, step, loss, tracker, epoch, writer))
    #Test Functions
    def test_loop_fn(loader, epoch):
        total_samples, correct = 0, 0
        model.eval()
        for step, (data, target) in enumerate(loader):
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total_samples += data.size()[0]
            if step % FLAGS.log_steps == 0:
                xm.add_step_closure(
                test_utils.print_test_update, args=(device, None, epoch, step))
        accuracy = 100.0 * correct.item() / total_samples
        accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
        return accuracy

    #Devices
    train_device_loader = pl.MpDeviceLoader(
        train_loader,
        device,
        loader_prefetch_size=FLAGS.loader_prefetch_size,
        device_prefetch_size=FLAGS.device_prefetch_size,
        host_to_device_transfer_threads=FLAGS.host_to_device_transfer_threads)
    test_device_loader = pl.MpDeviceLoader(
        test_loader,
        device,
        loader_prefetch_size=FLAGS.loader_prefetch_size,
        device_prefetch_size=FLAGS.device_prefetch_size,
        host_to_device_transfer_threads=FLAGS.host_to_device_transfer_threads)
        
    accuracy, max_accuracy = 0.0, 0.0
    for epoch in range(1, FLAGS.num_epochs + 1):
        xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
        train_loop_fn(train_device_loader, epoch)
        xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
        if not FLAGS.test_only_at_end or epoch == FLAGS.num_epochs:
            accuracy = test_loop_fn(test_device_loader, epoch)
            xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
            epoch, test_utils.now(), accuracy))
            max_accuracy = max(accuracy, max_accuracy)
            test_utils.write_to_summary(
                writer,
                epoch,
                dict_to_write={'Accuracy/test': accuracy},
                write_xla_metrics=True)
        if FLAGS.metrics_debug:
            xm.master_print(met.metrics_report())

    test_utils.close_summary_writer(writer)
    xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
    return max_accuracy


def _mp_fn(index, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type('torch.FloatTensor')
    accuracy = train_imagenet()
    if accuracy < FLAGS.target_accuracy:
        print('Accuracy {} is below target {}'.format(accuracy,
                                                  FLAGS.target_accuracy))
        sys.exit(21)


if __name__ == '__main__':
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)
