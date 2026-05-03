# train_one_epoch()   → one pass through train loader
# evaluate()          → one pass through val/test loader
# train()             → full training loop with checkpointing



import os
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item()

    return total_loss / len(loader)


def train(model, train_loader, val_loader,
          num_epochs=200, lr=1e-3,
          checkpoint_path="checkpoints/best_model.pt",
          device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader,
                                     optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        # print every 10 epochs
        print(f"Epoch {epoch:3d}/{num_epochs} "
              f"| train: {train_loss:.4f} "
              f"| val: {val_loss:.4f} "
              f"{'← best' if val_loss == best_val_loss else ''}")

    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")

    return train_losses, val_losses