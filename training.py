"""Training script."""

import dataclasses
import math
import time

import numpy as np
import torch.nn
import wandb

import data as data_lib
import network as network_lib

# Change to use your own device.
_DEVICE = torch.device("mps")


@dataclasses.dataclass(kw_only=True)
class TrainConfig:
    """Config for the training script."""

    log_frequency: int
    training_steps: int
    batch_size: int
    seq_length: int
    learning_rate: float
    clip_grad_norm: float = 1.0
    data_seed: int
    model_params_seed: int


def train(config: TrainConfig):
    """Trains a neural network on enwik8 data."""
    model = network_lib.DecoderOnly(
        vocab_size=256,
        embedding_dim=256,
        d_model=256,
        num_heads=8,
        num_layers=4,
    )
    model = model.to(_DEVICE)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,
        gamma=0.99,
    )

    dataset = data_lib.load_dataset(seq_length=config.seq_length)
    rng = np.random.default_rng(seed=config.data_seed)

    total_loss = 0.0
    start_time = time.time()
    for step in range(config.training_steps):
        sequences = data_lib.fetch_random_batch(dataset, config.batch_size, rng)
        tensor_sequences = torch.from_numpy(sequences)
        tensor_sequences = tensor_sequences.long()
        tensor_sequences = tensor_sequences.to(_DEVICE)

        output = model.forward(tensor_sequences)
        output_flat = output.view(-1, model.vocab_size)
        sequences_flat = tensor_sequences.view(-1)
        loss = loss_fn(output_flat, sequences_flat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.clip_grad_norm
        )
        optimizer.step()

        total_loss += loss.item()
        if step % config.log_frequency == 0:
            lr = scheduler.get_last_lr()[0]
            steps_per_sec = config.log_frequency / (time.time() - start_time)
            avg_loss = total_loss / config.log_frequency
            avg_bpb = avg_loss / math.log(2)
            print(
                f"| step {step} | "
                f"lr {lr:02.2f} | steps/s {steps_per_sec:5.2f} | "
                f"avg_bpb {avg_bpb:5.3f}"
            )
            wandb.log({"bpb": avg_bpb})
            total_loss = 0
            start_time = time.time()

    wandb.finish()
