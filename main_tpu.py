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

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

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
import torch_xla.experimental.pjrt_backend
import torch_xla.experimental.pjrt as pjrt

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
        '''with torch.no_grad():
            batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
            #dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output)) #* dist.get_world_size())

            # ema update
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        '''
        return total_loss

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

def get_args_parser(datadir=None,
                         logdir=None,
                         num_cores=None,
                         batch_size=128,
                         num_epochs=10,
                         num_workers=4,
                         log_steps=20,
                         lr=None,
                         momentum=None,
                         target_accuracy=None,
                         profiler_port=9012,
                         opts=None):
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_tiny', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--datadir', type=str, default=datadir)
    parser.add_argument('--logdir', type=str, default=logdir)
    parser.add_argument('--num_cores', type=int, default=num_cores)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--num_epochs', type=int, default=num_epochs)
    parser.add_argument('--num_workers', type=int, default=num_workers)
    parser.add_argument('--log_steps', type=int, default=log_steps)
    parser.add_argument('--profiler_port', type=int, default=profiler_port)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--target_accuracy', type=float, default=target_accuracy)
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--fake_data', action='store_true')
    parser.add_argument('--tidy', action='store_true')
    parser.add_argument('--metrics_debug', action='store_true')
    parser.add_argument('--async_closures', action='store_true')
    parser.add_argument('--debug', action='store_true')
    if opts:
        for name, aopts in opts:
            parser.add_argument(name, **aopts)
    args, leftovers = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + leftovers
    # Setup import folders.
    xla_folder = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    sys.path.append(os.path.join(os.path.dirname(xla_folder), 'test'))
    
    return args

FLAGS = get_args_parser(
    datadir='~/imagenet',
    batch_size=None,
    num_epochs=None,
    momentum=None,
    lr=None,
    target_accuracy=None,
    profiler_port=9012,
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
    '''if FLAGS.pjrt_distributed:
        import torch_xla.experimental.pjrt_backend
        dist.init_process_group('xla', init_method='pjrt://')
        print('PJRT execution')
    elif FLAGS.ddp:
        print('DDP execution')
        dist.init_process_group(
            'xla', world_size=xm.xrt_world_size(), rank=xm.get_ordinal())
    '''

    dist.init_process_group('xla', init_method='pjrt://')
    print('PJRT execution')
    
    print('==> Preparing data..')
    
    img_dim = 224
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
        FLAGS.global_crops_scale,
        FLAGS.local_crops_scale,
        FLAGS.local_crops_number,
        )
        train_dataset = datasets.ImageFolder(FLAGS.datadir, transform=transforms.Compose([transforms.Resize((256,256)),transform]))
        train_dataset_len = len(train_dataset.imgs)
        train_sampler, test_sampler = None, None
        if xm.xrt_world_size() > 1:
            print('Split data')
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True)
            """test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=False)"""
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=FLAGS.batch_size,
            sampler=train_sampler,
            drop_last=FLAGS.drop_last,
            shuffle=False if train_sampler else True,
            num_workers=FLAGS.num_workers,
            persistent_workers=FLAGS.persistent_workers,
            prefetch_factor=FLAGS.prefetch_factor)
        """test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=FLAGS.test_set_batch_size,
            sampler=test_sampler,
            drop_last=FLAGS.drop_last,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            persistent_workers=FLAGS.persistent_workers,
            prefetch_factor=FLAGS.prefetch_factor)
            print(f"Data loaded: there are {len(dataset)} images.")"""
        
    torch.manual_seed(42)
    FLAGS.arch = FLAGS.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if FLAGS.arch in vits.__dict__.keys():
        student = vits.__dict__[FLAGS.arch](
            patch_size=FLAGS.patch_size,
            drop_path_rate=FLAGS.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[FLAGS.arch](patch_size=FLAGS.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif FLAGS.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', FLAGS.arch,
                                pretrained=False, drop_path_rate=FLAGS.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', FLAGS.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif FLAGS.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[FLAGS.arch]()
        teacher = torchvision_models.__dict__[FLAGS.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {FLAGS.arch}")


    torch.manual_seed(42)
    

    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        FLAGS.out_dim,
        use_bn=FLAGS.use_bn_in_head,
        norm_last_layer=FLAGS.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, FLAGS.out_dim, FLAGS.use_bn_in_head),
    )

    device = xm.xla_device()
    student = student.to(device)
    teacher = teacher.to(device)
    # Initialization is nondeterministic with multiple threads in PjRt.
    # Synchronize model parameters across replicas manually.
    #if xr.using_pjrt():
    #pjrt.broadcast_master_paramm(student)
    #pjrt.broadcast_master_param(teacher)

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
    # ============ preparing optimizer ... ============
    writer = None
    if xm.is_master_ordinal():
        writer = test_utils.get_summary_writer(FLAGS.logdir)
    
    params_groups = utils.get_params_groups(student)
    if FLAGS.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif FLAGS.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif FLAGS.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    
    # for mixed precision training
    fp16_scaler = None
    if FLAGS.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    
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
        FLAGS.epochs, len(train_loader),
        warmup_epochs=FLAGS.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        FLAGS.weight_decay,
        FLAGS.weight_decay_end,
        FLAGS.epochs, len(train_loader)
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(FLAGS.momentum_teacher, 1,
                                               FLAGS.epochs, len(train_loader))

    if FLAGS.profile:
        server = xp.start_server(FLAGS.profiler_port)
    
    #Train function
    def train_loop_fn(loader, epoch):
        tracker = xm.RateTracker()
        teacher.train()
        student.train()
        for step, (data, target) in enumerate(loader):
            with xp.StepTrace('train_imagenet'):
                with xp.Trace('build_graph'):
                    print('Antes do predict train\n\n')
                    teacher_output = teacher(data[:2])
                    student_output = student(data)
                    print('Depois predict train\n\n')
                    loss = dino_loss(student_output, teacher_output, step)
                    print(loss)
                    if not math.isfinite(loss.item()):
                        print("Loss is {}, stopping training".format(loss.item()), force=True)
                        sys.exit(1)
                
                    optimizer.zero_grad()
                    param_norms = None

                    if fp16_scaler is None:
                        loss.backward()
                        if FLAGS.clip_grad:
                            param_norms = utils.clip_gradients(student, FLAGS.clip_grad)
                        utils.cancel_gradients_last_layer(epoch, student,
                                                        FLAGS.freeze_last_layer)
                        if FLAGS.ddp:
                            optimizer.step()
                        else:
                            xm.optimizer_step(optimizer)
                            tracker.add(FLAGS.batch_size)
                    else:
                        fp16_scaler.scale(loss).backward()
                        if FLAGS.clip_grad:
                            fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                            param_norms = utils.clip_gradients(student, FLAGS.clip_grad)
                        utils.cancel_gradients_last_layer(epoch, student,
                                                        FLAGS.freeze_last_layer)
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
    #test_device_loader = pl.MpDeviceLoader(
    #    test_loader,
    #    device,
    #    loader_prefetch_size=FLAGS.loader_prefetch_size,
    #    device_prefetch_size=FLAGS.device_prefetch_size,
    #    host_to_device_transfer_threads=FLAGS.host_to_device_transfer_threads)
        
    accuracy, max_accuracy = 0.0, 0.0
    for epoch in range(1, FLAGS.num_epochs + 1):
        xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
        train_loop_fn(train_device_loader, epoch)
        xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))
        '''if not FLAGS.test_only_at_end or epoch == FLAGS.num_epochs:
            accuracy = test_loop_fn(test_device_loader, epoch)
            xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
            epoch, test_utils.now(), accuracy))
            max_accuracy = max(accuracy, max_accuracy)
            test_utils.write_to_summary(
                writer,
                epoch,
                dict_to_write={'Accuracy/test': accuracy},
                write_xla_metrics=True)'''
        if FLAGS.metrics_debug:
            xm.master_print(met.metrics_report())

    #test_utils.close_summary_writer(writer)
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
