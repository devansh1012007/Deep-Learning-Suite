import torch.nn as nn

class Probe(nn.Module): # A probe is a simple model that is trained to predict a specific property or attribute of the input data, based on the intermediate features extracted from a pre-trained model.
    # Probes are often used in interpretability research to understand what information is encoded in the intermediate layers of a pre-trained model, and to evaluate the quality of the learned representations. 
    # By training a probe on the intermediate features, we can gain insights into what the model has learned and how it is processing the input data.
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)
'''
feature extraction and probing steps:
Freeze model
Extract intermediate features
Train probeFreeze model
Extract intermediate features
Train probe
'''