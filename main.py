"""Simple test for wandb."""

import wandb
import training as training_lib


def main():
    wandb.init(
        project="enwik8-test",
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    )
    training_lib.train()


if __name__ == "__main__":
    main()
