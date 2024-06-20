# Hyperparams: LR and DROPOUT_RATE
SLURM_JOBSCRIPT_TMPLT = """#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 4            # number of cores 
#SBATCH -t 0-00:30:00   # time in d-hh:mm:ss
#SBATCH --mem=10G
#SBATCH -G a30:1
#SBATCH -p general      # partition 
#SBATCH -q public       # QOS
#SBATCH -o hyperparam_search_jobs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e hyperparam_search_jobs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest

# Using python, so source activate an appropriate environment
source activate interest_region_cls
echo $LD_LIBRARY_PATH
nvidia-smi
source "/home/rawhad/personal_jobs/VQGAN/dev.env"

python -c "import torch; print(torch.cuda.is_available())"
python "/home/rawhad/personal_jobs/VQGAN/VQGAN/main.py" \\
  --lr "{lr}" \\
  --batch_size 64 \\
  --n_steps 500 \\
  --num_embeddings 1024 \\
  --dropout_rate "{dropout_rate}" \\
  --data_dir "/scratch/rawhad/datasets/preprocessed_tiny_imagenet" \\
  --project_name "vqgan_hyperparam_search" \\
  --run_name "{run_name}" \\
;
"""


import random
import os

ATTEMPT = 0
N = 20
for i in range(N):
  lr = 10**random.uniform(-5, -2)
  dropout_rate = 10**random.uniform(-2, -0.8)
  run_name = f'run-{ATTEMPT}-{i}'
  with open(f'hyperparam_scripts/{run_name}.sh', 'w') as f:
    f.write(SLURM_JOBSCRIPT_TMPLT.format(lr=lr, dropout_rate=dropout_rate, run_name=run_name))

  os.system(f'sbatch hyperparam_scripts/{run_name}.sh')
