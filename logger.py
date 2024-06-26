import wandb
from abc import ABC, abstractmethod

class Logger(ABC):
  @abstractmethod
  def log(self, data: dict, step: int):
    pass

class WandbLogger(Logger):
  def __init__(self, project_name, run_name):
    self.run = wandb.init(project=project_name, name=run_name)

  def log(self, data: dict, step: int):
    self.run.log(data, step=step)

  def update_config(self, config: dict):
    self.run.config.update(config)


class ConsoleLogger(Logger):
  def __init__(self, project_name, run_name):
    pass

  def log(self, data: dict, step: int):
    # pretty print the data
    print(f"Step: {step}")
    for key, value in data.items():
      print(f" {key:20s}: {value}")

  def update_config(self, config: dict):
    # pretty print the config
    print("Config:")
    for key, value in config.items():
      print(f" {key:20s}: {value}")