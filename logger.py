import wandb

class WandbLogger:
  def __init__(self, project_name, run_name):
    self.run = wandb.init(project=project_name, name=run_name)

  def log(self, data: dict, step: int):
    self.run.log(data, step=step)


class ConsoleLogger:
  def __init__(self, project_name, run_name):
    pass

  def log(self, data: dict, step: int):
    # pretty print the data
    print(f"Step: {step}")
    for key, value in data.items():
      print(f" {key:20s}: {value}")