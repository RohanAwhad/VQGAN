import dataclasses
import inspect
import math
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from dataset import Dataset
from logger import Logger

# ===
# Configure Optimizer
# ===
def configure_optimizer(model, weight_decay, lr, device_type):
  # get all parameters that require grad
  params = filter(lambda p: p.requires_grad, model.parameters())
  # create param groups based on ndim
  optim_groups = [
    {'params': [p for p in params if p.ndim >= 2], 'weight_decay': weight_decay},
    {'params': [p for p in params if p.ndim < 1], 'weight_decay': 0.0},
  ]
  # use fused optimizer if available
  fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
  use_fused = fused_available and device_type == 'cuda'
  return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)


# ===
# LR Scheduler
# ===
class CosineLRScheduler:
  def __init__(self, warmup_steps, max_steps, max_lr, min_lr):
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps
    self.max_lr = max_lr
    self.min_lr = min_lr

  def get_lr(self, step):
    # linear warmup
    if step < self.warmup_steps:
      return self.max_lr * (step+1) / self.warmup_steps

    # constant lr
    if step > self.max_steps:
      return self.min_lr

    # cosine annealing
    decay_ratio = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return self.min_lr + coeff * (self.max_lr - self.min_lr)



# Perceptual Loss
# Didn't seem to work quite well on CIFAR10 in colab
class PerceptualLoss(nn.Module):
  def __init__(self, device):
    import timm
    super().__init__()
    self.model = timm.create_model(model_name='vgg19_bn.tv_in1k', pretrained=True, features_only=True)
    self.model = self.model.eval()
    self.model.to(device)
    for param in self.model.parameters(): param.requires_grad = False

  def forward(self, predicted, targets):
    predicted = self.model(predicted)[-1]
    targets = self.model(targets)[-1]
    return F.mse_loss(predicted, targets)
    

# Lambda
def calculate_lambda(perceptual_loss, gan_loss, gen_last_layer):
  last_layer_weight = gen_last_layer.weight
  perceptual_loss_grad = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
  gan_loss_grad = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]
  lambda_ = torch.norm(perceptual_loss_grad) / (torch.norm(gan_loss_grad) + 1e-6)
  return torch.clamp(lambda_, 0, 1e4).detach()

# ===
# Normalization
# ===
DS_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DS_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def normalize(x):
  return (x - DS_MEAN) / DS_STD

@dataclasses.dataclass
class EngineConfig:
  train_ds: Dataset
  test_ds: Dataset
  vqgan: nn.Module
  disc: nn.Module
  N_STEPS: int
  device: str
  logger: Logger
  lr_scheduler: CosineLRScheduler
  grad_accum_steps: int
  checkpoint_every: int
  checkpoint_dir: str
  last_step: int
  # ddp vars
  is_ddp: bool
  ddp_rank: int
  ddp_local_rank: int
  ddp_world_size: int
  is_master_process: bool


def turn_off_grad(model: nn.Module):
  for param in model.parameters(): param.requires_grad = False
def turn_on_grad(model: nn.Module):
  for param in model.parameters(): param.requires_grad = True



def run(config: EngineConfig):
  device_type = 'cuda' if config.device.startswith('cuda') else config.device

  config.vqgan.to(config.device)
  config.disc.to(config.device)

  # was facing with batchnorm throwing errors in DDP because we had 2 forward passes and 1 backward.
  # solved it by setting broadcast_buffer=False. Ref: https://github.com/pytorch/pytorch/issues/66504
  if config.is_ddp:
    config.vqgan = nn.parallel.DistributedDataParallel(config.vqgan, device_ids=[config.ddp_local_rank], broadcast_buffers=False)
    config.disc = nn.parallel.DistributedDataParallel(config.disc, device_ids=[config.ddp_local_rank], broadcast_buffers=False)

  if config.last_step != -1:
    print("Resuming training from step:", config.last_step)
    # load model and optimizer state
    config.vqgan.module.load_state_dict(torch.load(f'{config.checkpoint_dir}/vqgan.pth'))
    config.disc.module.load_state_dict(torch.load(f'{config.checkpoint_dir}/disc.pth'))
    with open(f'{config.checkpoint_dir}/last_step.txt', 'r') as f: start_step = int(f.read())
  else:
    if config.is_master_process: print("Starting training from scratch")
    start_step = 0

  raw_vqgan = config.vqgan.module if config.is_ddp else config.vqgan
  raw_disc = config.disc.module if config.is_ddp else config.disc

  vqgan_opt = configure_optimizer(raw_vqgan, 0.1, config.lr_scheduler.max_lr, device_type) # lr is placeholder
  disc_opt = configure_optimizer(raw_disc, 0.1, config.lr_scheduler.max_lr, device_type)

  vqgan_opt.zero_grad()
  disc_opt.zero_grad()

  if config.last_step != -1:
    # load optimizer state
    vqgan_opt.load_state_dict(torch.load(f'{config.checkpoint_dir}/vqgan_opt.pth'))
    disc_opt.load_state_dict(torch.load(f'{config.checkpoint_dir}/disc_opt.pth'))

  test_samples = config.test_ds.next_batch()
  test_images = test_samples['images'].to(config.device)

  if config.is_master_process: print("Training for steps:", config.N_STEPS)
  config.vqgan.train()
  config.disc.train()

  for step in range(start_step, config.N_STEPS):
    start = time.monotonic()
    log_data = dict(
      disc=dict(total_loss = 0.0),
      gen=dict(
        total_loss = 0.0,
        commitment_loss = 0.0,
        reconstruction_loss = 0.0,
        lambda_ = 0.0,
        gan_loss = 0.0,
      )
    )

    vqgan_opt.zero_grad()
    disc_opt.zero_grad()
    for micro_step in range(config.grad_accum_steps):
      batch = config.train_ds.next_batch()
      images = batch['images'].to(config.device)

      # train vqgan
      turn_on_grad(config.vqgan)
      turn_off_grad(config.disc)
      with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        fake_img, _, commitment_loss = config.vqgan(images)
        disc_out = config.disc(normalize(fake_img)).flatten(0, 2)

        # reconstruction_loss = get_perceptual_loss(fake_img, images)
        reconstruction_loss = ((fake_img - images) ** 2).mean()
        gan_loss = F.binary_cross_entropy_with_logits(disc_out, torch.zeros_like(disc_out))
        #lambda_ = calculate_lambda(reconstruction_loss, gan_loss, raw_vqgan.generator.deconv1)
        # lambda as a constant
        lambda_ = 1.0
        gen_loss = lambda_ * gan_loss + reconstruction_loss + commitment_loss
        gen_loss = gen_loss / config.grad_accum_steps
      # if ddp, sync gradients on last micro_step
      if config.is_ddp:
        config.vqgan.require_backward_grad_sync = micro_step == (config.grad_accum_steps - 1)
      gen_loss.backward()

      # train discriminator
      with torch.autograd.set_detect_anomaly(True):
        turn_off_grad(config.vqgan)
        turn_on_grad(config.disc)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
          fake_img, _, _ = config.vqgan(images)  # recompute fake_img because retain_graph=True doesn't work with ddp
          disc_out = config.disc(normalize(fake_img)).flatten(0, 2)
          real_img_disc_out = config.disc(normalize(images)).flatten(0, 2)

          y_hat = torch.cat([real_img_disc_out, disc_out])
          y_true = torch.cat([torch.ones_like(real_img_disc_out), torch.zeros_like(disc_out)])
          disc_loss = F.binary_cross_entropy_with_logits(y_hat, y_true) / 2
          disc_loss = disc_loss / config.grad_accum_steps
        # if ddp, sync gradients on last micro_step
        if config.is_ddp:
          config.disc.require_backward_grad_sync = micro_step == (config.grad_accum_steps - 1)
        disc_loss.backward()

      # log
      log_data['disc']['total_loss'] += disc_loss.detach()
      log_data['gen']['total_loss'] += gen_loss.detach()
      log_data['gen']['commitment_loss'] += commitment_loss.detach()
      log_data['gen']['reconstruction_loss'] += reconstruction_loss.detach()
      log_data['gen']['lambda_'] += (lambda_ / config.grad_accum_steps)
      log_data['gen']['gan_loss'] += gan_loss.detach()

    turn_on_grad(config.vqgan)
    turn_on_grad(config.disc)
    lr = config.lr_scheduler.get_lr(step)
    for param_group in vqgan_opt.param_groups: param_group['lr'] = lr
    for param_group in disc_opt.param_groups: param_group['lr'] = lr

    vqgan_opt.step()
    disc_opt.step()

    log_data['lr'] = lr
    if config.is_ddp:
      dist.all_reduce(log_data['disc']['total_loss'], op=dist.ReduceOp.AVG)
      dist.all_reduce(log_data['gen']['total_loss'], op=dist.ReduceOp.AVG)
      dist.all_reduce(log_data['gen']['commitment_loss'], op=dist.ReduceOp.AVG)
      dist.all_reduce(log_data['gen']['reconstruction_loss'], op=dist.ReduceOp.AVG)
      #dist.all_reduce(log_data['gen']['lambda_'], op=dist.ReduceOp.AVG)
      dist.all_reduce(log_data['gen']['gan_loss'], op=dist.ReduceOp.AVG)
    if config.is_master_process: config.logger.log(log_data, step=step)

    end = time.monotonic()
    if config.is_master_process: print(f'Time taken for step {step}: {end-start:0.2f} secs')

    # periodically plot test images
    if step % 1000 == 0 and config.is_master_process:

      with torch.no_grad():
        config.vqgan.eval()
        img, _, _ = config.vqgan(test_images)
        config.vqgan.train()
      
      # calculate mse between test_images, and fake_images
      loss = ((img - test_images) ** 2).mean()
      print('Test Loss:', loss.item())
      batch_size, n_channels, img_h, img_w = img.shape
      _n_cols = int(math.sqrt(batch_size)) + 1

      img = F.interpolate(img, (32, 32)).permute(0, 2, 3, 1).detach().cpu().numpy()
      resized_test_images = F.interpolate(test_images, (32, 32)).permute(0, 2, 3, 1).detach().cpu().numpy()

      fig, axs = plt.subplots(_n_cols * 2, _n_cols)
      for a in range(_n_cols * 2):
        for b in range(_n_cols):
          idx = (a % _n_cols) * _n_cols + b
          if a < _n_cols:
            # plot real iamges
            img_db = resized_test_images
          else:
            # plot fake images
            img_db = img

          if idx < batch_size:
            axs[a, b].imshow(img_db[idx])

          axs[a, b].axis('off')

      config.logger.log(dict(test_images = fig), step=step)
      plt.close()

    # periodically save model and optimizer state
    if (step + 1) % config.checkpoint_every == 0 and config.is_master_process:
      with open(f'{config.checkpoint_dir}/last_step.txt', 'w') as f: f.write(str(step))
      torch.save(raw_vqgan.state_dict(), f'{config.checkpoint_dir}/vqgan.pth')
      torch.save(raw_disc.state_dict(), f'{config.checkpoint_dir}/disc.pth')
      torch.save(vqgan_opt.state_dict(), f'{config.checkpoint_dir}/vqgan_opt.pth')
      torch.save(disc_opt.state_dict(), f'{config.checkpoint_dir}/disc_opt.pth')
