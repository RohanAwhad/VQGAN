import dataclasses
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import Dataset
from .logger import Logger


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

@dataclasses.dataclass
class EngineConfig:
  train_ds: Dataset
  test_ds: Dataset
  vqgan: nn.Module
  disc: nn.Module
  vqgan_opt: torch.optim.Optimizer
  disc_opt: torch.optim.Optimizer
  N_STEPS: int
  device: str
  logger: Logger
  lr_scheduler: CosineLRScheduler
  grad_accum_steps: int
  checkpoint_every: int
  checkpoint_dir: str
  last_step: int = -1

def run(config: EngineConfig):
  config.vqgan.to(config.device)
  config.disc.to(config.device)

  config.vqgan_opt.zero_grad()
  config.disc_opt.zero_grad()

  test_samples = config.test_ds.next_batch()
  test_images = test_samples['images'].to(config.device)

  print("Training for steps:", config.N_STEPS)
  config.vqgan.train()
  config.disc.train()

  if config.last_step != -1:
    print("Resuming training from step:", config.last_step)
    # load model and optimizer state
    config.vqgan.load_state_dict(torch.load(f'{config.checkpoint_dir}/vqgan.pth'))
    config.disc.load_state_dict(torch.load(f'{config.checkpoint_dir}/disc.pth'))
    config.vqgan_opt.load_state_dict(torch.load(f'{config.checkpoint_dir}/vqgan_opt.pth'))
    config.disc_opt.load_state_dict(torch.load(f'{config.checkpoint_dir}/disc_opt.pth'))
    with open(f'{config.checkpoint_dir}/last_step.txt', 'r') as f: start_step = int(f.read())
  else:
    print("Starting training from scratch")
    start_step = 0

  for step in range(start_step, config.N_STEPS):
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

    config.vqgan_opt.zero_grad()
    config.disc_opt.zero_grad()
    for micro_step in range(config.grad_accum_steps):
      batch = config.train_ds.next_batch()
      images = batch['images'].to(config.device)

      with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
        fake_img, _, commitment_loss = config.vqgan(images)
        disc_out = config.disc(fake_img).flatten(0, 2)

        # reconstruction_loss = get_perceptual_loss(fake_img, images)
        reconstruction_loss = ((fake_img - images) ** 2).mean()
        gan_loss = F.binary_cross_entropy_with_logits(disc_out, torch.zeros_like(disc_out))
        lambda_ = calculate_lambda(reconstruction_loss, gan_loss, config.vqgan.generator.deconv1)
        gen_loss = lambda_ * gan_loss + reconstruction_loss + commitment_loss
        gen_loss = gen_loss / config.grad_accum_steps

        real_img_disc_out = config.disc(images).flatten(0, 2)

        y_hat = torch.cat([real_img_disc_out, disc_out])
        y_true = torch.cat([torch.ones_like(real_img_disc_out), torch.zeros_like(disc_out)])
        disc_loss = F.binary_cross_entropy_with_logits(y_hat, y_true) / 2
        disc_loss = disc_loss / config.grad_accum_steps

        # log
        log_data['disc']['total_loss'] += disc_loss.item()
        log_data['gen']['total_loss'] += gen_loss.item()
        log_data['gen']['commitment_loss'] += commitment_loss.item()
        log_data['gen']['reconstruction_loss'] += reconstruction_loss.item()
        log_data['gen']['lambda_'] += lambda_.item()
        log_data['gen']['gan_loss'] += gan_loss.item()

    lr = config.lr_scheduler.get_lr(step)
    for param_group in config.vqgan_opt.param_groups: param_group['lr'] = lr
    for param_group in config.disc_opt.param_groups: param_group['lr'] = lr

    config.vqgan_opt.zero_grad()
    gen_loss.backward(retain_graph=True)

    config.disc_opt.zero_grad()
    disc_loss.backward()

    config.vqgan_opt.step()
    config.disc_opt.step()

    config.logger.log(log_data, step=step)

    # periodically plot test images
    if step % 1000 == 0:

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
    if step % config.checkpoint_every == 0:
      with open(f'{config.checkpoint_dir}/last_step.txt', 'w') as f: f.write(str(step))
      torch.save(config.vqgan.state_dict(), f'{config.checkpoint_dir}/vqgan.pth')
      torch.save(config.disc.state_dict(), f'{config.checkpoint_dir}/disc.pth')
      torch.save(config.vqgan_opt.state_dict(), f'{config.checkpoint_dir}/vqgan_opt.pth')
      torch.save(config.disc_opt.state_dict(), f'{config.checkpoint_dir}/disc_opt.pth')
