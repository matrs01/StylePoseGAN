import yaml
import os

import wandb

from trainer.trainer import Trainer


def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    if not config["wandb_on"]:
        os.environ["WANDB_MODE"] = "offline"

    wandb.login()
    wandb.init()

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
