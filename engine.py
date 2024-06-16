import math
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F

def run(train_ds, test_ds, enc, gen, disc, enc_gen_opt, disc_opt, N_STEPS, device):
  enc.to(device)
  gen.to(device)
  disc.to(device)

  enc_gen_opt.zero_grad()
  disc_opt.zero_grad()

  test_samples = test_ds.next_batch().to(device)

  print("Training for steps:", N_STEPS)
  enc.train()
  gen.train()
  disc.train()

  start = time.monotonic()
  for i in range(N_STEPS):

    batch = train_ds.next_batch().to(device)

    # train discriminator
    for params in enc.parameters(): params.requires_grad = False
    for params in gen.parameters(): params.requires_grad = False
    for params in disc.parameters(): params.requires_grad = True


    enc_gen_opt.zero_grad()
    disc_opt.zero_grad()
    real_img_disc_out = disc(batch).sigmoid()
    
    with torch.no_grad():
      latent_dim = enc(batch)
      fake_imgs = gen(latent_dim)

    fake_disc_out = disc(fake_imgs).sigmoid()

    y_hat = torch.cat([real_img_disc_out, fake_disc_out])
    y_true = torch.cat([torch.ones_like(real_img_disc_out), torch.zeros_like(fake_disc_out)])
    disc_loss = F.binary_cross_entropy(y_hat, y_true)
    disc_loss /= 2  # because we have 2x batch_size
    disc_loss.backward()
    disc_opt.step()
    disc_opt.zero_grad()

    # train encoder-generator
    for params in enc.parameters(): params.requires_grad = True
    for params in gen.parameters(): params.requires_grad = True
    for params in disc.parameters(): params.requires_grad = False

    enc_gen_opt.zero_grad()
    disc_opt.zero_grad()
    
    latent_dim = enc(batch)
    fake_img = gen(latent_dim)
    disc_out = disc(fake_img).sigmoid()
    gen_loss = 0.5 * (F.binary_cross_entropy(disc_out, torch.ones_like(disc_out)) + F.mse_loss(fake_img, batch))
    gen_loss.backward()

    enc_gen_opt.step()
    enc_gen_opt.zero_grad()


    end = time.monotonic()
    print(f"Step: {i:4d} | Gen Loss: {gen_loss.item():.4f} | Disc Loss: {disc_loss.item():.4f} | Time: {end - start:.2f} s")
    start = time.monotonic()

    if i % 100 == 0:

      enc.eval()
      gen.eval()
      latent_dim = enc(test_samples)
      img = gen(latent_dim)
      enc.train()
      gen.train()
      img = img.detach().cpu().numpy()
      img = img.reshape(-1, 28, 28)
      n_cols = int(math.sqrt(img.shape[0])) + 1
      fig, axs = plt.subplots(n_cols, n_cols)
      for j, ax in enumerate(axs.flatten()):
        if j < img.shape[0]:
          ax.imshow(img[j], cmap='gray')
        ax.axis('off')

      plt.savefig(f"images/{i}.png")
      plt.close()
