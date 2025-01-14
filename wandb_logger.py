from collections import defaultdict

import numpy as np
import wandb

class WandbLogger:
    def __init__(self, save_val_metrics, project_name: str, run_name: str = None):
        wandb.init(project=project_name, name=run_name)
        self.best_epoch = None
        self.best_value = None
        self.save_val_metrics = save_val_metrics
        self.last_train_dict = defaultdict(lambda: None)
        self.last_val_dict = defaultdict(lambda: None)

        best_epoch_val_metrics = {}
        for metric in self.save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf

    def log(self, metrics: dict, step: int, train: bool = True):
        split = "train" if train else "val"
        metrics = {f"{key}/{split}": value.item() if hasattr(value, 'item') else value for key, value in metrics.items()}
        wandb.log(metrics, step=step)
        # if train:
        #     self.last_train_dict = metrics
        # else:
        #     self.last_val_dict = metrics
        # metrics = {f"{key}/train": value.item() if hasattr(value, 'item') else value for key, value in self.last_train_dict.items()}
        # metrics.update({f"{key}/val": value.item() if hasattr(value, 'item') else value for key, value in self.last_val_dict.items()})
        # wandb.log(metrics, step=step)

    def log_best(self, value: float, step: int):
        if self.best_value is None or value < self.best_value:
            self.best_value = value
            self.best_step = step
            wandb.log({"best_value": self.best_value, "best_step": self.best_step}, step=step)

    def set_config(self, config: dict):
        wandb.config.update(config)

    def finish(self):
        wandb.finish()

