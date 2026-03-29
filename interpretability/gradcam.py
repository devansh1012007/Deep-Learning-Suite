import torch
import torch.nn.functional as F

class GradCAM: # Grad-CAM (Gradient-weighted Class Activation Mapping) is a technique used in computer vision to visualize the regions of an input 
    # image that are most important for a model's prediction. It works by computing the gradients of the target class with respect to the feature maps of a 
    # specific convolutional layer, and then using these gradients to weight the feature maps to create a heatmap that highlights the important regions in the input image.
    # this is used for interpretability, to understand which parts of the input image are contributing to the model's decision, and can be useful for debugging and improving model performance.
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation) # register_forward_hook is a method in PyTorch that allows you to register a function (hook) 
        # that will be called every time a forward pass is made through a specific layer of the model. In this case, the save_activation function will be called
        # every time a forward pass is made through the target_layer, allowing you to save the activations of that layer for later use in generating the Grad-CAM heatmap.
        
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x, class_idx): # generate is a method in the GradCAM class that takes an input tensor x and a class index class_idx, and generates a 
        # Grad-CAM heatmap for the specified class index.
        output = self.model(x)

        self.model.zero_grad()
        output[:, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[2:], mode='bilinear')

        return cam.squeeze()
    
