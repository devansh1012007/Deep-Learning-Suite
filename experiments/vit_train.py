from models.vit.vit import ViT
import torch.nn as nn
import torch.optim as optim

model = ViT().to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05) # AdamW is an optimization algorithm that combines the benefits of Adam and weight decay regularization. 
# It helps to prevent overfitting by adding a penalty to the loss function based on the magnitude of the model's weights, which encourages the model to learn simpler representations and generalize better to unseen data.

criterion = nn.CrossEntropyLoss() # CrossEntropyLoss is a commonly used loss function for classification tasks. 
# It measures the difference between the predicted class probabilities and the true class labels, and it encourages the model to output high probabilities for the correct class and low probabilities for the incorrect classes.