
import torch
import torch.nn as nn
from models.resnet.blocks import BasicBlock

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # By calling super().__init__(), we can ensure that the necessary initialization steps defined in the nn.Module class are executed,
        # which allows us to properly set up our ResNet18 model and take advantage of the functionalities provided by the nn.Module class, 
        # such as parameter management, forward pass definition, and other features that are essential for building and training neural networks in PyTorch.

        self.in_channels = 64 # channel size is important for ResNet design, as it determines the number of filters in each convolutional layer and 
        # affects the overall capacity and performance of the model. By setting self.in_channels to 64, we are defining the initial number of channels for 
        # the first convolutional layer in the ResNet architecture. This value is typically chosen based on the input data and the desired model complexity,
        #  and it can be adjusted to optimize the performance of the model on a specific task or dataset.

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)# this line is used to define the first convolutional layer of the ResNet18 model. 
        # The nn.Conv2d class is a PyTorch module that applies a 2D convolution operation to the input data. In this case, we are creating a convolutional 
        # layer that takes an input with 3 channels (e.g., RGB images) and produces an output with 64 channels (as defined by self.in_channels). 
        # The kernel_size argument specifies the size of the convolutional filter (in this case, a 3x3 filter), stride=1 indicates that the filter will move 
        # one pixel at a time across the input, padding=1 adds a one-pixel border around the input to maintain the spatial dimensions, and 
        # bias=False indicates that we do not want to include a bias term in the convolutional layer. This layer will be responsible for extracting 
        # features from the input images and will be followed by batch normalization and activation layers to further process the extracted features.
        
        self.bn1 = nn.BatchNorm2d(64) # this line is used to define a batch normalization layer that will be applied after the first convolutional layer in the ResNet18 model.
        
        self.relu = nn.ReLU(inplace=True) # The nn.ReLU class is a PyTorch module that applies the Rectified Linear Unit (ReLU) activation function to the input data. 
        # By setting inplace=True, we are specifying that the ReLU operation should be performed in-place, meaning that it will modify the input tensor directly without creating a new tensor for the output.
        # This can help save memory and improve performance during training. The ReLU activation function introduces non-linearity into the model, allowing it to learn complex patterns and relationships in the data, which is essential for achieving good performance on tasks such as image classification.

        ### convolutional layers and blocks

        # layer1 means a sequence of blocks. each block is a sequence of layers. the first layer in the first block of each layer is responsible for downsampling the input 
        # (except for layer1). the number of channels is doubled after each layer. the number of blocks in each layer is defined by the ResNet architecture 
        # (2 for ResNet18).
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1) # The BasicBlock modules will be responsible for learning and extracting features from the input data while maintaining the identity mapping through skip connections.
        #The stride of 1 indicates that there will be no downsampling in this layer, allowing the spatial dimensions of the input to be preserved.
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # adaptive average pooling layer that will be applied after the last convolutional layer in the ResNet18 model.
        #By passing (1, 1) as an argument, we are specifying that the output of this layer should have a spatial dimension of 1x1, regardless of the input size.
        # This means that the adaptive average pooling layer will compute the average value of each channel across the entire spatial dimensions of the input, 
        # effectively reducing the spatial dimensions to 1x1 while retaining the number of channels. 
        # This is typically used before the fully connected layer to prepare the features for classification.


        self.fc = nn.Linear(512, num_classes) # The nn.Linear class is a PyTorch module that applies a linear transformation to the input data. 
        # By passing 512 as the first argument, we are specifying that the input to this layer will have 512 features (which corresponds to the number of channels output by the last convolutional layer), 
        # and by passing num_classes as the second argument, we are specifying that the output of this layer will have a number of features equal to the number of classes in our classification task 
        # (e.g., 10 for CIFAR-10). This fully connected layer will be responsible for mapping the extracted features from the convolutional layers to the final class scores, which can then be used for classification.

    def _make_layer(self, block, out_channels, blocks, stride): # this method is used to create a layer consisting of a specified number of blocks (e.g., BasicBlock) in the ResNet architecture. 
        # The method takes the block type, the number of output channels, the number of blocks, and the stride as arguments. 
        # It initializes an empty list called layers and then appends the first block to the list, which is responsible for downsampling the input if necessary (based on the stride and channel size).
        # After that, it updates self.in_channels to reflect the new number of channels after the first block. 
        # Then, it iterates over the remaining number of blocks and appends additional blocks to the layers list without downsampling.
        # Finally, it returns a sequential container (nn.Sequential) that contains all the blocks in the layer, allowing us to easily apply them in sequence during the forward pass of the model.
        
        layers = []

        layers.append(block(self.in_channels, out_channels, stride)) # downsampling the input if necessary (based on the stride and channel size).
        self.in_channels = out_channels * block.expansion # channel means the number of filters in each convolutional layer and affects the overall capacity and performance of the model.

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x): # this method defines the forward pass of the ResNet18 model. It takes an input tensor x and processes it through the layers of the model to produce the final output.
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # faltten is important here because the output of the avgpool layer will have a shape of (batch_size, num_channels, 1, 1), 
        # and we need to flatten it to (batch_size, num_channels) before passing it to the fully connected layer. By using torch.flatten(x, 1),
        # we are flattening the tensor starting from the first dimension (which corresponds to the batch size) and keeping the number of channels intact. 
        # This allows us to prepare the features for classification in the fully connected layer.

        x = self.fc(x) # fc layer will be responsible for mapping the extracted features from the convolutional layers to the final class scores, which can then be used for classification.

# deffrence between channel, layers, blocks, stage, connections, etc. in ResNet architecture:
# - Channel: Refers to the number of filters in a convolutional layer, which determines the depth of the feature maps produced by that layer. In ResNet, the number of channels typically doubles after each stage (e.g., 64, 128, 256, 512).
# - Layer: Refers to a single operation in the neural network, such as a convolutional layer, batch normalization layer, or activation layer. In ResNet, layers are organized into blocks and stages.
# - Block: Refers to a group of layers that are connected together, often with skip connections. 
# In ResNet, there are two types of blocks: BasicBlock (used in ResNet18 and ResNet34) and BottleneckBlock (used in ResNet50, ResNet101, and ResNet152). 
# Each block contains multiple layers and is responsible for learning features at a specific level of abstraction.
# - Stage: Refers to a group of blocks that operate at the same spatial resolution. In ResNet, there are typically four stages, each containing a certain number of blocks. The spatial resolution is reduced after each stage through downsampling (e.g., using stride=2 in the convolutional layers).
# - Connections: Refers to the way layers and blocks are connected together in the ResNet architecture. 
# ResNet is known for its skip connections (also called identity connections), which allow the input to bypass certain layers and be added directly to the output of those layers. 
# This helps to mitigate the vanishing gradient problem and allows for the training of deeper networks. The connections can be either identity connections (where the input is added directly to the output) or projection connections (where a convolutional layer is used to match the dimensions before adding).