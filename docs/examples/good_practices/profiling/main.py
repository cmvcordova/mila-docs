"""Single-GPU training example."""
import argparse
import logging
import os
import time
from pathlib import Path

import rich.logging
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.models import resnet50
from tqdm import tqdm


def main():
    # Use an argument parser so we can pass hyperparameters from the command line.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batches", type=int, default=30)
    args = parser.parse_args()

    epochs: int = args.epochs
    learning_rate: float = args.learning_rate
    weight_decay: float = args.weight_decay
    batch_size: int = args.batch_size
    test_batches: int = args.test_batches

    # Check that the GPU is available
    assert torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda", 0)

    # Setup logging (optional, but much better than using print statements)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[rich.logging.RichHandler(markup=True)],  # Very pretty, uses the `rich` package.
    )

    logger = logging.getLogger(__name__)

    # Create a model and move it to the GPU.
    model = resnet50(num_classes=1000)
    model.to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Setup ImageNet
    logger.info("Setting up ImageNet")
    num_workers = get_num_workers()
    dataset_path = Path(os.environ.get("SLURM_TMPDIR", ".")) / "imagenet"
    train_dataset, valid_dataset, test_dataset = make_datasets(str(dataset_path))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    test_dataloader = DataLoader(  # NOTE: Not used in this example.
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    logger.info("Beginning bottleneck diagnosis.")
    logger.info("Starting dataloader loop without training.")
    ## TODO: Pass into function and call directly to illustrate the bottleneck
    ## example in a few lines of code. People who are interested in how the bottleneck is computed
    ## can then go and see how the function is implemented.
    
    dataloader_start_time = time.time()
    n_batches = 0
    for batch_idx, batch in enumerate(tqdm(
            train_dataloader,
            desc="Dataloader throughput test",
            # hint: look at unit_scale and unit params
            unit="batches",
            total=test_batches,
    )): 
        if batch_idx >= test_batches:
            break

        batch = tuple(item.to(device) for item in batch)
        n_batches += 1

    dataloader_end_time = time.time()
    dataloader_elapsed_time = dataloader_end_time - dataloader_start_time
    avg_time_per_batch = dataloader_elapsed_time / n_batches
    logger.info(f"Baseline dataloader speed: {avg_time_per_batch:.3f} s/batch")
    
    
    logger.info("Starting training loop.")
    for epoch in range(epochs):
        logger.debug(f"Starting epoch {epoch}/{epochs}")

        # Set the model in training mode (important for e.g. BatchNorm and Dropout layers)
        model.train()

        # NOTE: using a progress bar from tqdm because it's nicer than using `print`.
        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Train epoch {epoch}",
            # hint: look at unit_scale and unit params
            unit="images",
            unit_scale=train_dataloader.batch_size,
        )

        # Training loop
        for batch in train_dataloader:
            # Move the batch to the GPU before we pass it to the model
            batch = tuple(item.to(device) for item in batch)
            x, y = batch

            # Forward pass
            logits: Tensor = model(x)

            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate some metrics:
            n_correct_predictions = logits.detach().argmax(-1).eq(y).sum()
            n_samples = y.shape[0]
            accuracy = n_correct_predictions / n_samples

            logger.debug(f"Accuracy: {accuracy.item():.2%}")
            logger.debug(f"Average Loss: {loss.item()}")

            # Advance the progress bar one step and update the progress bar text.
            progress_bar.update()
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item())
        progress_bar.close()

        val_loss, val_accuracy = validation_loop(model, valid_dataloader, device)
        logger.info(f"Epoch {epoch}: Val loss: {val_loss:.3f} accuracy: {val_accuracy:.2%}")

    print("Done!")


@torch.no_grad()
def validation_loop(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()

    total_loss = 0.0
    n_samples = 0
    correct_predictions = 0

    for batch in dataloader:
        batch = tuple(item.to(device) for item in batch)
        x, y = batch

        logits: Tensor = model(x)
        loss = F.cross_entropy(logits, y)

        batch_n_samples = x.shape[0]
        batch_correct_predictions = logits.argmax(-1).eq(y).sum()

        total_loss += loss.item()
        n_samples += batch_n_samples
        correct_predictions += batch_correct_predictions

    accuracy = correct_predictions / n_samples
    return total_loss, accuracy

def dataloader_throughput_loop(dataloader: DataLoader, device: torch.device):
    pass

def make_datasets(
    dataset_path: str,
    val_split: float = 0.1,
    val_split_seed: int = 42,
    target_size: tuple = (224, 224),
):
    """Returns the training, validation, and test splits for ImageNet.

    NOTE: We don't use image transforms here for simplicity.
    Having different transformations for train and validation would complicate things a bit.
    Later examples will show how to do the train/val/test split properly when using transforms.
    """

    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'val')

    transform = Compose([
        Resize(target_size),
        ToTensor(),
    ])

    train_dataset = ImageFolder(
        root=train_dir,
        transform=transform, 
    )
    test_dataset = ImageFolder(
        root=test_dir,
        transform=transform,
    )

    # Split the training dataset into training and validation
    n_samples = len(train_dataset)
    n_valid = int(val_split * n_samples)
    n_train = n_samples - n_valid

    train_dataset, valid_dataset = random_split(
        train_dataset, [n_train, n_valid], 
        generator = torch.Generator().manual_seed(val_split_seed))                                                         

    return train_dataset, valid_dataset, test_dataset


def get_num_workers() -> int:
    """Gets the optimal number of DatLoader workers to use in the current job."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


if __name__ == "__main__":
    main()