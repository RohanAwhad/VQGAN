from dotenv import load_dotenv
load_dotenv('/home/rawhad/personal_jobs/VQGAN/dev.env')

import argparse
import os
import torch

import engine, logger
from dataset import ImageNetDatasetLoaderLite
from model import Encoder, Generator, Discriminator, Codebook, VQGAN

# ===
# DDP setup
# ===
from torch.distributed import init_process_group, destroy_process_group

is_ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

# torchrun cmd sets env vars: RANK, LOCAL_RANK, and WORLD_SIZE
if is_ddp: 
  assert torch.cuda.is_available(), "DDP requires CUDA"
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  DEVICE = f'cuda:{ddp_local_rank}'
  torch.cuda.set_device(DEVICE)
  is_master_process = ddp_rank == 0 # master process will do the logging, checkpointing, etc.
else:
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  DEVICE = 'cpu'
  if torch.cuda.is_available(): DEVICE = 'cuda'
  if torch.backends.mps.is_available(): DEVICE = 'mps'
  is_master_process = True


# ===
# Constants
# ===
argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=3e-4)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--micro_batch_size', type=int, default=0)
argparser.add_argument('--n_steps', type=int, default=5000)
argparser.add_argument('--num_embeddings', type=int, default=1024)
argparser.add_argument('--dropout_rate', type=float, default=0.1)
argparser.add_argument('--data_dir', type=str, default="/scratch/rawhad/datasets/preprocessed_tiny_imagenet")
argparser.add_argument('--last_step', type=int, default=-1)
argparser.add_argument('--save_model', action='store_true')
argparser.add_argument('--project_name', type=str, default='vqgan')
argparser.add_argument('--run_name', type=str, default='test-imagenet')
args = argparser.parse_args()

LR = args.lr
TOTAL_BATCH_SIZE = args.batch_size
MICRO_BATCH_SIZE = args.micro_batch_size if args.micro_batch_size > 0 else TOTAL_BATCH_SIZE
assert TOTAL_BATCH_SIZE % (MICRO_BATCH_SIZE * ddp_world_size) == 0, "Total batch size must be divisible by (micro batch size * world_size)"

GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // (MICRO_BATCH_SIZE * ddp_world_size)
N_STEPS = args.n_steps
NUM_EMBEDDINGS = args.num_embeddings
DROPOUT_RATE = args.dropout_rate
DATA_DIR = args.data_dir
SAVE_MODEL = args.save_model
PROJECT_NAME = args.project_name
RUN_NAME = args.run_name

WARMUP_STEPS = 5000
MAX_STEPS = N_STEPS // 3
MAX_LR = LR
MIN_LR = LR / 10

GEN_LR = LR
DISC_LR = LR

MODEL_DIR = '/scratch/rawhad/VQGAN/models'
if is_master_process:
  os.makedirs(MODEL_DIR, exist_ok=True)

if is_master_process:
  #LOGGER = logger.WandbLogger(project_name=PROJECT_NAME, run_name=RUN_NAME)
  LOGGER = logger.ConsoleLogger(project_name='vqgan', run_name='test-imagenet')
else:
  LOGGER = None

# ===
# Intialization
# ===
torch.manual_seed(1234)  # setting seed because we are using DDP
if torch.cuda.is_available(): torch.cuda.manual_seed(1234)

train_ds = ImageNetDatasetLoaderLite(split='train', batch_size=MICRO_BATCH_SIZE, root=DATA_DIR, process_rank=ddp_rank, num_processes=ddp_world_size)
test_ds = ImageNetDatasetLoaderLite(split='test', batch_size=MICRO_BATCH_SIZE, root=DATA_DIR, process_rank=ddp_rank, num_processes=ddp_world_size)

codebook = Codebook(num_embeddings=NUM_EMBEDDINGS, embedding_dim=2048)  # 2048 is the output dim of the encoder
encoder = Encoder(DROPOUT_RATE)
generator = Generator(DROPOUT_RATE)
discriminator = Discriminator(DROPOUT_RATE)
vqgan = VQGAN(encoder, codebook, generator)

# ===
# Configure Optimizers and LR Schedulers
# ===


# lr scheduler
lr_scheduler = engine.CosineLRScheduler(WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)

# Unable to compile
#vqgan = torch.compile(vqgan)
#discriminator = torch.compile(discriminator)
torch.set_float32_matmul_precision('high')

training_config = engine.EngineConfig(
  train_ds=train_ds,
  test_ds=test_ds,
  vqgan=vqgan,
  disc=discriminator,
  N_STEPS=N_STEPS,
  device=DEVICE,
  logger=LOGGER,
  lr_scheduler=lr_scheduler,
  grad_accum_steps=GRAD_ACCUM_STEPS,
  checkpoint_every=1000,
  checkpoint_dir=MODEL_DIR,
  last_step=args.last_step,
  is_ddp=is_ddp,
  ddp_rank=ddp_rank,
  ddp_local_rank=ddp_local_rank,
  ddp_world_size=ddp_world_size,
  is_master_process=is_master_process,
)
engine.run(training_config)

if is_master_process:
  # save models
  codebook.to('cpu')
  encoder.to('cpu')
  generator.to('cpu')

  if SAVE_MODEL:
    torch.save(codebook.state_dict(), os.path.join(MODEL_DIR, 'codebook_final.pth'))
    torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, 'encoder_final.pth'))
    torch.save(generator.state_dict(), os.path.join(MODEL_DIR, 'generator_final.pth'))


destroy_process_group()
