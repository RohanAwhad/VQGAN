import os
import torch

import engine, logger
from dataset import DatasetLoaderLite
from model import Encoder, Generator, Discriminator, Codebook, VQGAN

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

LOGGER = logger.ConsoleLogger(project_name='vqgan', run_name='test-mnist')

# ===
# Intialization
# ===
DATA_DIR = "/Users/rohan/3_Resources/ai_datasets/caltech_101/caltech101/101_ObjectCategories"
train_ds = DatasetLoaderLite(root=DATA_DIR, batch_size=BATCH_SIZE, shuffle=True)
test_ds = DatasetLoaderLite(root=DATA_DIR, batch_size=BATCH_SIZE, shuffle=False)

codebook = Codebook(num_embeddings=64, embedding_dim=2048)
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