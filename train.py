import torch.nn.functional as F
import torch


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for data, target,seq in train_loader:
        data, target,seq = data.to(device), target.float().to(device),seq.float().to(device)
        # target = (target >= 0.5).to(torch.int64)
        # target_onehot = F.one_hot((target >= 0.5).to(torch.int64), 2).to(device=device)
        optimizer.zero_grad()
        output = model(data,seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)
