import math
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F

def calculate_lambda(perceptual_loss, gan_loss, gen_last_layer):
  last_layer_weight = gen_last_layer.weight
  perceptual_loss_grad = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
  gan_loss_grad = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]
  lambda_ = torch.norm(perceptual_loss_grad) / (torch.norm(gan_loss_grad) + 1e-4)
  return torch.clamp(lambda_, 0, 1e4).detach()

def run(train_ds, test_ds, vqgan, disc, vqgan_opt, disc_opt, N_STEPS, device, n_channels):
  vqgan.to(device)
  disc.to(device)

  vqgan_opt.zero_grad()
  disc_opt.zero_grad()

  test_samples = test_ds.next_batch()
  test_images = test_samples['images'].to(device)

  print("Training for steps:", N_STEPS)
  vqgan.train()
  disc.train()

  start = time.monotonic()
  for i in range(N_STEPS):

    batch = train_ds.next_batch()
    images = batch['images'].to(device)
    fake_img, commitment_loss = vqgan(images)
    disc_out = disc(fake_img).flatten(0, 2)

    reconstruction_loss = ((fake_img - images) ** 2).mean()
    gan_loss = F.binary_cross_entropy(disc_out, torch.zeros_like(disc_out))
    lambda_ = calculate_lambda(reconstruction_loss, gan_loss, vqgan.generator.last_layer)
    gen_loss = lambda_ * gan_loss + reconstruction_loss + commitment_loss

    vqgan_opt.zero_grad()
    gen_loss.backward(retain_graph=True)

    real_img_disc_out = disc(images).flatten(0, 2)

    y_hat = torch.cat([real_img_disc_out, disc_out])
    y_true = torch.cat([torch.ones_like(real_img_disc_out), torch.zeros_like(disc_out)])
    disc_loss = 0.3 * F.binary_cross_entropy(y_hat, y_true) / 2

    disc_opt.zero_grad()
    disc_loss.backward()

    vqgan_opt.step()
    disc_opt.step()

    vqgan_opt.zero_grad()
    disc_opt.zero_grad()

    end = time.monotonic()
    print(f"Step: {i:4d} | Disc Loss: {disc_loss.item():.4f} | Gen Loss: {gen_loss.item():.4f} | Commitment Loss: {commitment_loss:.4f} | Reconstruction loss: {reconstruction_loss:.4f} | Time: {end - start:.2f} s")
    start = time.monotonic()

    if i % 100 == 0:

      with torch.no_grad():
        vqgan.eval()
        img, _ = vqgan(test_images)
        vqgan.train()
      
      # calculate mse between test_images, and fake_images
      loss = ((img - test_images) ** 2).mean()
      print('Test Loss:', loss.item())
      batch_size, n_channels, img_h, img_w = img.shape
      # calculate euclidean distance between fake_images intra batch
      # dist_mat = torch.cdist(img.view(batch_size, -1), img.view(batch_size, -1))
      # # plot the distance matrix
      # plt.imshow(dist_mat.detach().cpu().numpy())
      # # add legend
      # plt.colorbar()
      # plt.savefig(f"images/dist_{i}.png")
      # plt.close()

      img = img.permute(0, 2, 3, 1).detach().cpu().numpy()
      _n_cols = int(math.sqrt(batch_size)) + 1

      resized_test_images = test_images.permute(0, 2, 3, 1).detach().cpu().numpy()

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

      plt.savefig(f"images/{i}.png")
      plt.close()
