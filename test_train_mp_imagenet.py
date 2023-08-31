"""Fork of test_train_mp_mnist.py to demonstrate how to profile workloads."""
import args_parse

FLAGS = args_parse.parse_common_options(
    datadir='/tmp/cifar10-data',
    batch_size=1,
    momentum=0.5,
    lr=0.01,
    target_accuracy=98.0,
    num_epochs=18,
    profiler_port=9012)

FLAGS.arch = 'vit_tiny'
FLAGS.patch_size = 8
FLAGS.drop_path_rate = 0.1
FLAGS.out_dim=65536
FLAGS.local_crops_number=8
FLAGS.warmup_teacher_temp=0.04
FLAGS.teacher_temp=0.04
FLAGS.warmup_teacher_temp_epochs=0
FLAGS.global_crops_scale=(0.4, 1.)
FLAGS.local_crops_scale=(0.05, 0.4)
FLAGS.use_bn_in_head=False
FLAGS.norm_last_layer=True

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
import os
import shutil
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torchvision import datasets, transforms
import torch.distributed as dist
from torchvision import models as torchvision_models
import utils
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

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
        #self.update_center(teacher_output)

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

class MNIST(nn.Module):

  def __init__(self):
    super(MNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.bn1 = nn.BatchNorm2d(10)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.bn2 = nn.BatchNorm2d(20)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    with xp.Trace('conv1'):
      x = F.relu(F.max_pool2d(self.conv1(x), 2))
      x = self.bn1(x)
    with xp.Trace('conv2'):
      x = F.relu(F.max_pool2d(self.conv2(x), 2))
      x = self.bn2(x)
    with xp.Trace('dense'):
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
    return F.log_softmax(x, dim=1)


def _train_update(device, x, loss, tracker, writer):
  test_utils.print_training_update(
      device,
      x,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      summary_writer=writer)


def train_mnist(flags,
                training_started=None,
                dynamic_graph=False,
                fetch_often=False):
  torch.manual_seed(1)
  #import torch_xla.experimental.pjrt_backend
  #dist.init_process_group('xla', init_method='pjrt://')
  if flags.fake_data:
    img_dim = 224
    train_dataset_len = 12000  # Roughly the size of Imagenet dataset.
    
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
            torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
        sample_count=train_dataset_len // FLAGS.batch_size //
        xm.xrt_world_size(), transform=transform)
    
    #test_loader = xu.SampleGenerator(
    #    data=(torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
    #            torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
    #            sample_count=50000 // FLAGS.batch_size // xm.xrt_world_size())
  else:
    transform = DataAugmentationDINO(
        FLAGS.global_crops_scale,
        FLAGS.local_crops_scale,
        FLAGS.local_crops_number,
    )

    train_dataset = datasets.CIFAR10(
        os.path.join(flags.datadir, str(xm.get_ordinal())),
        train=True,
        download=True,
        transform=transform)
    
    test_dataset = datasets.CIFAR10(
        os.path.join(flags.datadir, str(xm.get_ordinal())),
        train=False,
        download=True,
        transform=transform)
    
    train_sampler = None
    if xm.xrt_world_size() > 1:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=flags.batch_size,
        sampler=train_sampler,
        drop_last=flags.drop_last,
        shuffle=False if train_sampler else True,
        num_workers=flags.num_workers)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=flags.batch_size,
        drop_last=flags.drop_last,
        shuffle=False,
        num_workers=flags.num_workers)

  # Scale learning rate to num cores
  lr = flags.lr * xm.xrt_world_size()

  
  student = vits.__dict__[FLAGS.arch](
            patch_size=FLAGS.patch_size,
            drop_path_rate=FLAGS.drop_path_rate,  # stochastic depth
        )
  teacher = vits.__dict__[FLAGS.arch](patch_size=FLAGS.patch_size)
  embed_dim = student.embed_dim
  student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        flags.out_dim,
        use_bn=flags.use_bn_in_head,
        norm_last_layer=flags.norm_last_layer,
    ))
  teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, flags.out_dim, flags.use_bn_in_head),
    )
  device = xm.xla_device()
  student.to(device)
  teacher.to(device)
  
  
  writer = None
  if xm.is_master_ordinal():
    writer = test_utils.get_summary_writer(flags.logdir)
  optimizer = optim.SGD(student.parameters(), lr=lr, momentum=flags.momentum)
  
  dino_loss = DINOLoss(
        FLAGS.out_dim,
        FLAGS.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        FLAGS.warmup_teacher_temp,
        FLAGS.teacher_temp,
        FLAGS.warmup_teacher_temp_epochs,
        FLAGS.num_epochs,
  )

  server = xp.start_server(flags.profiler_port)

  def train_loop_fn(loader, epoch):
    tracker = xm.RateTracker()
    student.train()
    teacher.train()
    for step, (data, target) in enumerate(loader):
      with xp.StepTrace('train_mnist', step_num=step):
        with xp.Trace('build_graph'):
          optimizer.zero_grad()
          s_out = student(data)
          t_out = teacher(data[:2])
          loss = dino_loss(s_out, t_out, epoch)
          loss.backward()
        xm.optimizer_step(optimizer)
        if fetch_often:
          # testing purpose only: fetch XLA tensors to CPU.
          loss_i = loss.item()
        tracker.add(flags.batch_size)
        if step % flags.log_steps == 0:
          xm.add_step_closure(
              _train_update, args=(device, step, loss, tracker, writer))

  def test_loop_fn(loader):
    total_samples = 0
    correct = 0
    student.eval()
    for data, target in loader:
      with xp.StepTrace('test_mnist'):
        output = student(data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()
        total_samples += data.size()[0]

    accuracy = 100.0 * correct.item() / total_samples
    accuracy = xm.mesh_reduce('test_accuracy', accuracy, np.mean)
    return accuracy

  train_device_loader = pl.MpDeviceLoader(train_loader, device)
  test_device_loader = pl.MpDeviceLoader(test_loader, device)
  accuracy, max_accuracy = 0.0, 0.0
  for epoch in range(0, flags.num_epochs):
    xm.master_print('Epoch {} train begin {}'.format(epoch, test_utils.now()))
    train_loop_fn(train_device_loader, epoch)
    xm.master_print('Epoch {} train end {}'.format(epoch, test_utils.now()))

    accuracy = test_loop_fn(test_device_loader)
    xm.master_print('Epoch {} test end {}, Accuracy={:.2f}'.format(
        epoch, test_utils.now(), accuracy))
    max_accuracy = max(accuracy, max_accuracy)
    test_utils.write_to_summary(
        writer,
        epoch,
        dict_to_write={'Accuracy/test': accuracy},
        write_xla_metrics=True)
    if flags.metrics_debug:
      xm.master_print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  xm.master_print('Max Accuracy: {:.2f}%'.format(max_accuracy))
  return max_accuracy


def _mp_fn(rank, flags):
  global FLAGS
  FLAGS = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  #print("Starting train method on rank: {}".format(rank))
  #dist.init_process_group(
  #      backend='nccl', world_size=1, init_method='env://',
  #      rank=rank)
  accuracy = train_mnist(flags, dynamic_graph=True, fetch_often=True)
  if flags.tidy and os.path.isdir(flags.datadir):
    shutil.rmtree(flags.datadir)
  if accuracy < flags.target_accuracy:
    print('Accuracy {} is below target {}'.format(accuracy,
                                                  flags.target_accuracy))
    sys.exit(21)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores)