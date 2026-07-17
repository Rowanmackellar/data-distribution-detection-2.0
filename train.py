import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.models import ResNetCIFAR
from src.datasets import get_clean_dataloaders

# Defining num of training loops globally
EPOCHS = 30

# Setup to handle multiple devices
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# Safe-guards & Enviornment
if __name__ == "__main__":
    device = get_device()
    writer = SummaryWriter(log_dir="runs/resnet_experiment")
    train_loader, test_loader = get_clean_dataloaders()
    
    # Deep learning components
    model = ResNetCIFAR().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Ensure directory exists
    os.makedirs("models/checkpoints", exist_ok=True)
    
    # Track the highest accuracy achieved to save the absolute best weights
    best_acc = 0.0
    
    print(f"Starting training on device: {device} for {EPOCHS} epochs...")
    
    # 30 Epoch training loop
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        print(f"Epoch {epoch}/{EPOCHS} complete. Loss: {loss:.3f}, Acc: {acc:.3f}")
        
        # Log values to TensorBoard to see graphs of your progress later
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        
        # Only overwrite the checkpoint if this epoch's accuracy is better than before
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/checkpoints/resnet18_best.pt")
            print(f"--> New best accuracy achieved ({acc:.3%})! Checkpoint updated.")
            
    writer.close()
    print("Training finished successfully.")