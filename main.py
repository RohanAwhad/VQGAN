import dataclasses
import os
import torch

import engine
from dataset import DatasetLoaderLite
from model import (Encoder, Generator, Discriminator, Codebook, VQGAN,
  # EncoderConfig, GeneratorConfig, DiscriminatorConfig, CodebookConfig
)

# ===
# Constants
# ===
GEN_LR = 3e-4
DISC_LR = 3e-4
BATCH_SIZE = 32
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

n_channels = 1
input_dim = patch_size * patch_size * n_channels
n_heads = 4
embed_dim = 16
max_len = 64
dropout_rate = 0.1
num_embeddings = 8
latent_dim = embed_dim

# ENCODER_CONFIG = EncoderConfig(
#   input_dim=input_dim,
#   output_dim=latent_dim,
#   max_len=max_len,
#   n_hidden_layers=6,
#   n_hidden_dims=[embed_dim] * 6,
#   merge_after=3,
#   intermediate_scale=2,
#   n_heads=n_heads,
#   dropout_rate=dropout_rate
# )
# GENERATOR_CONFIG = GeneratorConfig(
#   input_dim=latent_dim,
#   output_dim=input_dim,
#   max_len=max_len,
#   n_hidden_layers=6,
#   n_hidden_dims=[embed_dim] * 6,
#   merge_after=3,
#   intermediate_scale=2,
#   n_heads=n_heads,
#   dropout_rate=dropout_rate
# )
# DISCRIMINATOR_CONFIG = EncoderConfig(
#   input_dim=input_dim,
#   output_dim=1,
#   max_len=max_len,
#   n_hidden_layers=4,
#   n_hidden_dims=[embed_dim, embed_dim, embed_dim, embed_dim*2, embed_dim*2, embed_dim*2],
#   merge_after=3,
#   intermediate_scale=2,
#   n_heads=n_heads,
#   dropout_rate=dropout_rate
# )
# DISCRIMINATOR_CONFIG = DiscriminatorConfig(
#   input_dim=input_dim,
#   max_len=max_len,
#   n_hidden_layers=4,
#   embed_dim=embed_dim,
#   intermediate_scale=2,
#   n_heads=n_heads,
#   dropout_rate=dropout_rate
# )
# CODEBOOK_CONFIG = CodebookConfig(
#   num_embeddings=num_embeddings,
#   embedding_dim=latent_dim,
# )

# codebook = Codebook(**dataclasses.asdict(CODEBOOK_CONFIG))
# encoder = Encoder(**dataclasses.asdict(ENCODER_CONFIG))
# generator = Generator(**dataclasses.asdict(GENERATOR_CONFIG))
# vqgan = VQGAN(encoder, codebook, generator)

# discriminator = Discriminator(**dataclasses.asdict(DISCRIMINATOR_CONFIG))

codebook = Codebook(num_embeddings=16, embedding_dim=128)
encoder = Encoder()
generator = Generator()
discriminator = Discriminator()
vqgan = VQGAN(encoder, codebook, generator)



# compile 
# vqgan = torch.compile(vqgan)
# discriminator = torch.compile(discriminator)


vqgan_opt = torch.optim.AdamW(vqgan.parameters(), lr=GEN_LR)
disc_opt = torch.optim.AdamW(discriminator.parameters(), lr=DISC_LR)

engine.run(train_ds, test_ds, vqgan, discriminator, vqgan_opt, disc_opt, N_STEPS, DEVICE, n_channels)

# save models
codebook.to('cpu')
encoder.to('cpu')
generator.to('cpu')

torch.save(codebook.state_dict(), os.path.join(MODEL_DIR, 'codebook.pth'))
torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, 'encoder.pth'))
torch.save(generator.state_dict(), os.path.join(MODEL_DIR, 'generator.pth'))