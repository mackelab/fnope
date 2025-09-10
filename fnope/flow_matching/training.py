import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from pathlib import Path
import time
import csv
from fnope.utils.misc import get_output_dir
from torch.optim import Adam


out_dir = get_output_dir()

def get_dataloaders(theta, x, x_finite=None,batch_size=256, validation_fraction=0.1):
    """
    Function to create dataloaders for training and validation

    Args:
    theta (torch.Tensor): tensor of shape (num_samples, num_channels, seq_len)
        containing the training data
    x (torch.Tensor): tensor of shape (num_samples, num_channels, seq_len)
        containing the context data
    x_finite (torch.Tensor): tensor of shape (num_samples, num_channels, seq_len)
        containing the finite data
    batch_size (int): batch size for training
    validation_fraction (float): fraction of data to use for validation
    """
    # Split data into training and validation set
    if x_finite is not None:
        assert theta.shape[0] == x_finite.shape[0] == x.shape[0], "theta, x and x_finite must have the same number of samples"
        dataset = TensorDataset(theta, x, x_finite)
    else:
        assert theta.shape[0] == x.shape[0], "theta and x must have the same number of samples"
        dataset = TensorDataset(theta, x)


    num_samples = theta.shape[0]
    num_training_samples = int((1 - validation_fraction) * num_samples)
    num_validation_samples = num_samples - num_training_samples 

    # Split the dataset into training and validation
    train_dataset, val_dataset = random_split(dataset, [num_training_samples, num_validation_samples])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        drop_last=False,
    )

    return train_loader, val_loader


def train_fnope(model, cfg, theta, x, simulation_positions, ctx_simulation_positions=None, x_finite= None, save_path=".", device="cpu"):
    """
    Function to train the FNOPE model

    Args:
    model (FNOPE): FNOPE model to train
    train_cfg (DictConfig): training configuration
    theta (torch.Tensor): tensor of shape (num_samples, num_channels, seq_len) of parameters
    x (torch.Tensor): tensor of shape (num_samples, num_channels, seq_len) of context data
    simulation_positions (torch.Tensor): tensor of shape (num_samples, num_channels, seq_len) of simulation positions
    ctx_simulation_positions (torch.Tensor): tensor of shape (num_samples, num_channels, seq_len) of context simulation positions
    x_finite (torch.Tensor): tensor of shape (num_samples, finite_dim) of finite parameters
    save_path (str): path to save the model
    device (torch.device): device to use for training
    """


    # Set device
    model.to(device)
    train_cfg = cfg.model_config

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(theta, x, x_finite, batch_size=train_cfg.batch_size, validation_fraction=train_cfg.validation_fraction)


    # Prepare csv to write training progress into
    csv_file = Path(save_path) / 'losses.csv'
    fieldnames = ['epoch', 'train_loss', 'val_loss']

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    optimizer = Adam(model.parameters(), lr=train_cfg.learning_rate)

    converged = False
    epoch = 0

    while epoch <= cfg.model_config.max_num_epochs and not converged:

        # Training step
        model.train()
        avg_training_loss = 0.0
        epoch_start_time = time.time()
        for data_batch in train_loader:
            if x_finite is not None:
                theta_batch, x_batch, x_finite_batch = data_batch
            else:
                theta_batch, x_batch = data_batch
                x_finite_batch = None
            optimizer.zero_grad()
            loss = model.loss(
                theta_batch, ctx=x_batch, x_finite=x_finite_batch,simulation_positions=simulation_positions, ctx_simulation_positions=ctx_simulation_positions
            )
            avg_training_loss += loss.item() * theta_batch.shape[0] / len(train_loader.dataset)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, total loss: {avg_training_loss:.4f}")
        

        # Validation step
        model.eval()
        avg_val_loss = 0.0

        with torch.no_grad():
            for data_batch in val_loader:
                if x_finite is not None:
                    theta_batch, x_batch, x_finite_batch = data_batch
                else:
                    theta_batch, x_batch = data_batch
                    x_finite_batch = None
            
                # TODO: add possibility of fixed validation times to loss function 
                val_loss = model.loss(
                    theta_batch, ctx=x_batch,x_finite=x_finite_batch, simulation_positions = simulation_positions, ctx_simulation_positions=ctx_simulation_positions
                )

                avg_val_loss += val_loss.item() * theta_batch.shape[0] / len(val_loader.dataset)

        # Check convergence 
        if epoch == 0:
            best_val_loss = avg_val_loss
            epochs_since_last_improvement = 0 
        elif avg_val_loss < best_val_loss: 
            best_val_loss = avg_val_loss
            epochs_since_last_improvement = 0         
        else: 
            epochs_since_last_improvement += 1

        # Stop training if no improvement over several epochs
        if epochs_since_last_improvement > cfg.model_config.stop_after_epochs:
            converged = True

        # Write to csv
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerow({'epoch': epoch, 'train_loss': avg_training_loss, 'val_loss': avg_val_loss})

        epoch += 1
    
    return epoch, avg_training_loss, avg_val_loss