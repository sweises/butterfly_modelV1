import torch

def train_model(model, train_loader, val_loader, device, criterion, optimizer, epochs):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch+1}/{epochs} "
              f"- Loss: {total_loss/len(train_loader):.4f} "
              f"- Val Acc: {correct/total:.4f}")

    return model
