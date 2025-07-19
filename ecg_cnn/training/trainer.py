import torch


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch on the given dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.

    dataloader : torch.utils.data.DataLoader
        DataLoader yielding (inputs, targets) batches for training.

    optimizer : torch.optim.Optimizer
        Optimizer used for updating model weights.

    criterion : torch.nn.Module
        Loss function (e.g., BCEWithLogitsLoss).

    device : torch.device
        Device to perform training on ('cpu' or 'cuda').

    Returns
    -------
    float
        Average training loss across all batches.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise TypeError("dataloader must be a torch.utils.data.DataLoader")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if not callable(criterion):
        raise TypeError("criterion must be callable (e.g., a loss function)")
    if not isinstance(device, torch.device):
        raise TypeError("device must be a torch.device")

    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    return avg_loss
