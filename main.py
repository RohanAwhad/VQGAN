import os
import torch

import engine
from dataset import DatasetLoaderLite
from model import Encoder, Generator, Discriminator, Codebook, VQGAN
from logger import WandbLogger

# ===
# Constants
# ===
GEN_LR = 3e-4
DISC_LR = 3e-4
BATCH_SIZE = 64
N_STEPS = 5000

DEVICE = 'cpu'
if torch.cuda.is_available(): DEVICE = 'cuda'
if torch.backends.mps.is_available(): DEVICE = 'mps'

MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

LOGGER = WandbLogger(project_name='vqgan', run_name='test-mnist')

# ===
# Intialization
# ===
train_ds = DatasetLoaderLite(train=True, root='data', batch_size=BATCH_SIZE, shuffle=True)
test_ds = DatasetLoaderLite(train=False, root='data', batch_size=BATCH_SIZE, shuffle=False)

codebook = Codebook(num_embeddings=4, embedding_dim=128)
encoder = Encoder()
generator = Generator()
discriminator = Discriminator()
vqgan = VQGAN(encoder, codebook, generator)


vqgan_opt = torch.optim.AdamW(vqgan.parameters(), lr=GEN_LR)
disc_opt = torch.optim.AdamW(discriminator.parameters(), lr=DISC_LR)

engine.run(train_ds, test_ds, vqgan, discriminator, vqgan_opt, disc_opt, N_STEPS, DEVICE, LOGGER)

# save models
codebook.to('cpu')
encoder.to('cpu')
generator.to('cpu')

torch.save(codebook.state_dict(), os.path.join(MODEL_DIR, 'codebook.pth'))
torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, 'encoder.pth'))
torch.save(generator.state_dict(), os.path.join(MODEL_DIR, 'generator.pth'))