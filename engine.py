import math
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F

def run(train_ds, test_ds, vqgan, disc, vqgan_opt, disc_opt, N_STEPS, device):
  vqgan.to(device)
  disc.to(device)

  vqgan_opt.zero_grad()
  disc_opt.zero_grad()

  test_samples = test_ds.next_batch().to(device)

  print("Training for steps:", N_STEPS)
  vqgan.train()
  disc.train()

  start = time.monotonic()
  for i in range(N_STEPS):

    batch = train_ds.next_batch().to(device)

    # train discriminator
    for params in vqgan.parameters(): params.requires_grad = False
    for params in disc.parameters(): params.requires_grad = True

    vqgan_opt.zero_grad()
    disc_opt.zero_grad()
    real_img_disc_out = disc(batch).sigmoid()
    
    with torch.no_grad(): fake_imgs, _ = vqgan(batch)
    fake_disc_out = disc(fake_imgs).sigmoid()

    y_hat = torch.cat([real_img_disc_out, fake_disc_out])
    y_true = torch.cat([torch.ones_like(real_img_disc_out), torch.zeros_like(fake_disc_out)])
    disc_loss = F.binary_cross_entropy(y_hat, y_true)
    disc_loss /= 2  # because we have 2x batch_size
    disc_loss.backward()
    disc_opt.step()
    disc_opt.zero_grad()

    # train encoder-generator
    for params in vqgan.parameters(): params.requires_grad = True
    for params in disc.parameters(): params.requires_grad = False

    vqgan_opt.zero_grad()
    disc_opt.zero_grad()
    
    fake_img, q_loss = vqgan(batch)
    disc_out = disc(fake_img).sigmoid()
    gen_loss = F.binary_cross_entropy(disc_out, torch.ones_like(disc_out)) + F.mse_loss(fake_img, batch) + q_loss
    gen_loss.backward()

    vqgan_opt.step()
    vqgan_opt.zero_grad()


    end = time.monotonic()
    print(f"Step: {i:4d} | Gen Loss: {gen_loss.item():.4f} | Disc Loss: {disc_loss.item():.4f} | Time: {end - start:.2f} s")
    start = time.monotonic()

    if i % 1000 == 0:

      vqgan.eval()
      img, _ = vqgan(test_samples)
      vqgan.train()
      img = img.detach().cpu().numpy()
      img = img.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
      n_cols = int(math.sqrt(img.shape[0])) + 1
      fig, axs = plt.subplots(n_cols, n_cols)
      for j, ax in enumerate(axs.flatten()):
        if j < img.shape[0]:
          ax.imshow(img[j], cmap='gray')
        ax.axis('off')

      plt.savefig(f"images/{i}.png")
      plt.close()
