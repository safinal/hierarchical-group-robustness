import torch
from tqdm import tqdm
import time
import os

from src.config import ConfigManager


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start_time = time.time()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update tqdm progress bar description
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{100. * correct / total:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_time = time.time() - epoch_start_time

    print(f"Epoch {epoch} | Training Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.4f} | Time: {epoch_time:.2f}s")
    return epoch_loss, epoch_acc

def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    print(f"Starting training for {num_epochs} epochs.")

    # Main training loop
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)


        # Save the latest model
        os.makedirs(ConfigManager().get("logs_dir"), exist_ok=True)
        latest_checkpoint_path = os.path.join(ConfigManager().get("logs_dir"), 'latest_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
        }, latest_checkpoint_path)

        print(f"Epoch [{epoch}/{num_epochs}] Summary: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
