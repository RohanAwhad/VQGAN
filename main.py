import os
import torch

import engine
from dataset import MNISTDataset
from model import Encoder, Generator, Discriminator, Codebook, VQGAN

# ===
# Constants
# ===
GEN_LR = 3e-4
DISC_LR = 3e-4
BATCH_SIZE = 64
N_STEPS = 20000
DEVICE = 'cpu'
if torch.cuda.is_available(): DEVICE = 'cuda'
if torch.backends.mps.is_available(): DEVICE = 'mps'
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ===
# Intialization
# ===
train_ds = MNISTDataset(train=True, root='data', batch_size=BATCH_SIZE, shuffle=True)
test_ds = MNISTDataset(train=False, root='data', batch_size=BATCH_SIZE, shuffle=False)

num_embeddings = 10
input_dim = 28 * 28
latent_dim = 64
n_heads = 8
embed_dim = latent_dim // n_heads

codebook = Codebook(num_embeddings=num_embeddings, embedding_dim=embed_dim, n_heads=n_heads)
encoder = Encoder(input_dim, latent_dim)
generator = Generator(latent_dim, input_dim)
vqgan = VQGAN(encoder, codebook, generator)

discriminator = Discriminator(input_dim)

vqgan_opt = torch.optim.AdamW(vqgan.parameters(), lr=GEN_LR)
disc_opt = torch.optim.AdamW(discriminator.parameters(), lr=DISC_LR)

engine.run(train_ds, test_ds, vqgan, discriminator, vqgan_opt, disc_opt, N_STEPS, DEVICE)

# save models
codebook.to('cpu')
encoder.to('cpu')
generator.to('cpu')

torch.save(codebook.state_dict(), os.path.join(MODEL_DIR, 'codebook.pth'))
torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, 'encoder.pth'))
torch.save(generator.state_dict(), os.path.join(MODEL_DIR, 'generator.pth'))