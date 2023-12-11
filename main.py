"""Simple test for wandb."""

import dataclasses

import wandb

import training as training_lib


def main():
    config = training_lib.TrainConfig(
        log_frequency=100,
        training_steps=100_000,
        batch_size=32,
        seq_length=128,
        learning_rate=1e-3,
        data_seed=1,
        model_params_seed=1,
    )
    wandb.init(
        project="enwik8-test",
        config=dataclasses.asdict(config),
    )
    training_lib.train(config)


if __name__ == "__main__":
    main()
