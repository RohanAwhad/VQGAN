import math
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F

def run(train_ds, test_ds, vqgan, disc, vqgan_opt, disc_opt, N_STEPS, device, n_channels):
  vqgan.to(device)
  disc.to(device)

  vqgan_opt.zero_grad()
  disc_opt.zero_grad()

  test_samples = test_ds.next_batch()
  test_images = test_samples['imgs'].to(device)
  test_n_rows = test_samples['n_rows']
  test_n_cols = test_samples['n_cols']

  print("Training for steps:", N_STEPS)
  vqgan.train()
  disc.train()

  start = time.monotonic()
  for i in range(N_STEPS):

    batch = train_ds.next_batch()
    images = batch['imgs'].to(device)
    n_rows = batch['n_rows']
    n_cols = batch['n_cols']

    # train discriminator
    for params in vqgan.parameters(): params.requires_grad = False
    for params in disc.parameters(): params.requires_grad = True

    vqgan_opt.zero_grad()
    disc_opt.zero_grad()
    # real_img_disc_out = disc(images, n_rows, n_cols)[0].flatten(0, 1)
    real_img_disc_out = disc(images).flatten(0, 1)
    
    with torch.no_grad(): fake_imgs, _ = vqgan(images, n_rows=n_rows, n_cols=n_cols)
    # fake_disc_out = disc(fake_imgs, n_rows, n_cols)[0].flatten(0, 1)
    fake_disc_out = disc(fake_imgs).flatten(0, 1)

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
    
    fake_img, q_loss = vqgan(images, n_rows=n_rows, n_cols=n_cols)
    # disc_out = disc(fake_img, n_rows, n_cols)[0]
    disc_out = disc(fake_img)
    # print(disc_out)
    disc_out = disc_out.flatten(0, 1)
    mse_loss = ((fake_img - images) ** 2).mean()
    gen_loss = F.binary_cross_entropy(disc_out, torch.ones_like(disc_out)) + mse_loss + q_loss
    gen_loss.backward()

    vqgan_opt.step()
    vqgan_opt.zero_grad()


    end = time.monotonic()
    print(f"Step: {i:4d} | Disc Loss: {disc_loss.item():.4f} | Gen Loss: {gen_loss.item():.4f} | Q Loss: {q_loss:.4f} | mse loss: {mse_loss:.4f} | Time: {end - start:.2f} s")
    start = time.monotonic()

    if i % 100 == 0:

      # TODO (rohan): for the time being added here. Change it later
      img_h, img_w = 32, 32
      patch_size = 4

      print('Test images:', test_images.shape)
      print(torch.all(torch.eq(test_images[0], test_images)))

      vqgan.eval()
      img, _ = vqgan(test_images, n_rows=test_n_rows, n_cols=test_n_cols)
      # calculate mse between test_images, and fake_images
      loss = ((img - test_images) ** 2).mean()
      print('Test Loss:', loss.item())

      vqgan.train()
      batch_size, n_patches, embed_dim = img.shape
      img = img.view(batch_size, img_h // patch_size, img_w // patch_size, patch_size, patch_size, n_channels)
      img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, img_h, img_w, n_channels)
      # calculate euclidean distance between fake_images intra batch
      dist_mat = torch.cdist(img.view(batch_size, -1), img.view(batch_size, -1))
      # plot the distance matrix
      plt.imshow(dist_mat.detach().cpu().numpy())
      # add legend
      plt.colorbar()
      plt.savefig(f"images/dist_{i}.png")
      plt.close()


      img = img.detach().cpu().numpy()
      _n_cols = int(math.sqrt(batch_size)) + 1

      resized_test_images = test_images.view(batch_size, img_h // patch_size, img_w // patch_size, patch_size, patch_size, n_channels)
      resized_test_images = resized_test_images.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, img_h, img_w, n_channels)
      resized_test_images = resized_test_images.detach().cpu().numpy()

      fig, axs = plt.subplots(_n_cols * 2, _n_cols)
      for a in range(_n_cols * 2):
        for b in range(_n_cols):
          idx = (a % _n_cols) * _n_cols + b
          if a < _n_cols:
            img_db = resized_test_images
            # plot real iamges
          else:
            # plot fake images
            img_db = img

          if idx < batch_size:
            axs[a, b].imshow(img_db[idx].squeeze(), cmap='gray')

          axs[a, b].axis('off')

      plt.savefig(f"images/{i}.png")
      plt.close()
