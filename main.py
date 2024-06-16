import dataclasses
import os
import torch

import engine
from dataset import DatasetLoaderLite
from model import (Encoder, Generator, Discriminator, Codebook, VQGAN,
  EncoderConfig, GeneratorConfig, DiscriminatorConfig, CodebookConfig
)

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
patch_size = 4
train_ds = DatasetLoaderLite(train=True, root='data', batch_size=BATCH_SIZE, shuffle=True, patch_size=patch_size)
test_ds = DatasetLoaderLite(train=False, root='data', batch_size=BATCH_SIZE, shuffle=False, patch_size=patch_size)

num_embeddings = 128
input_dim = patch_size * patch_size * 3
n_heads = 8
embed_dim = 128

ENCODER_CONFIG = EncoderConfig(
  input_dim=input_dim,
  max_len=32,
  n_hidden_layers=4,
  n_hidden_dims=[embed_dim, embed_dim, embed_dim*2, embed_dim*2],
  merge_after=2,
  intermediate_scale=2,
  n_heads=n_heads
)
GENERATOR_CONFIG = GeneratorConfig(
  input_dim=input_dim,
  max_len=32,
  n_hidden_layers=4,
  n_hidden_dims=[embed_dim*2, embed_dim*2, embed_dim, embed_dim],
  merge_after=2,
  intermediate_scale=2,
  n_heads=n_heads
)
DISCRIMINATOR_CONFIG = DiscriminatorConfig(
  input_dim=input_dim,
  max_len=32,
  n_hidden_layers=4,
  embed_dim=embed_dim,
  intermediate_scale=2,
  n_heads=n_heads
)
CODEBOOK_CONFIG = CodebookConfig(
  num_embeddings=num_embeddings,
  embedding_dim=ENCODER_CONFIG.n_hidden_dims[-1],
)

codebook = Codebook(**dataclasses.asdict(CODEBOOK_CONFIG))
encoder = Encoder(**dataclasses.asdict(ENCODER_CONFIG))
generator = Generator(**dataclasses.asdict(GENERATOR_CONFIG))
vqgan = VQGAN(encoder, codebook, generator)

discriminator = Discriminator(**dataclasses.asdict(DISCRIMINATOR_CONFIG))

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