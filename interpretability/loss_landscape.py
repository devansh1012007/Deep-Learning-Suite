def compute_loss(model, data, target, criterion):
    output = model(data)
    return criterion(output, target).item()