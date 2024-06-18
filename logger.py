import wandb

class WandbLogger:
  def __init__(self, project_name, run_name):
    self.run = wandb.init(project=project_name, name=run_name)

  def log(self, data: dict, step: int):
    self.run.log(data, step=step)