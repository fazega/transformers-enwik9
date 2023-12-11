"""Launch script for a given config."""

import dataclasses

import wandb

import training as training_lib


def main():
    """Trains a model with a config."""
    config = training_lib.TrainConfig(
        log_frequency=10,
        training_steps=10_000,
        batch_size=32,
        seq_length=128,
        learning_rate=3e-4,
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
