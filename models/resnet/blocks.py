import torch
import torch.nn as nn

class BasicBlock(nn.Module): # this class defines a basic building block for the ResNet architecture. 
    #It consists of two convolutional layers with batch normalization and ReLU activation functions. 
    # The block also includes a skip connection that allows the input to bypass the convolutional layers and be added to the output, 
    # which helps to mitigate the vanishing gradient problem and allows for deeper networks to be trained effectively. 
    # The expansion variable is set to 1, which indicates that the number of output channels is the same as the number of input channels. 
    # This block can be used as a fundamental component in constructing ResNet models of various depths.
    
    
    expansion = 1  # important for ResNet design --> coz it determines how the number of channels changes across the layers. 
    #In a ResNet architecture, the number of channels typically increases as you go deeper into the network.
    # The expansion variable helps to control this increase in channels, allowing for a more efficient and effective design of the network.
    # By setting expansion to 1, it indicates that the number of output channels is the same as the number of input channels,
    # which can be useful for certain layers in the ResNet architecture where you want to maintain the same number of channels.
    # if u sdet to 2 or 4, it means that the number of output channels will be 2 or 4 times the number of input channels, respectively, 
    # which can help to increase the capacity of the network and allow for more complex representations to be learned.

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, # here the first convolutional layer takes in the number of input channels (in_channels) and the number of output channels (out_channels) as arguments. 
                               #The kernel size is set to 3, which means that the convolutional filter will have a size of 3x3. The stride is set to the value of the stride argument passed to the constructor, which determines how much the filter moves across the input image during convolution. The padding is set to 1, which means that the input image will be padded with a border of 1 pixel on all sides to maintain the spatial dimensions after convolution. The bias is set to False, which means that no bias term will be added to the output of this convolutional layer.
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # here the batch normalization layer is applied to the output of the first convolutional layer. It takes the number of output channels (out_channels) as an argument, which is used to normalize the activations across the batch. Batch normalization helps to stabilize and accelerate the training process by normalizing the inputs to each layer, which can improve the convergence and generalization of the model.

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Skip connection adjustment
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):# this class defines a bottleneck building block for the ResNet architecture.
    # It consists of three convolutional layers with batch normalization and ReLU activation functions.
    # The first convolutional layer reduces the number of channels, the second layer performs the main convolution operation, 
    # and the third layer restores the number of channels. The block also includes a skip connection that allows the input to bypass the convolutional layers
    # and be added to the output, which helps to mitigate the vanishing gradient problem and allows for deeper networks to be trained effectively.
    # The expansion variable is set to 4, which indicates that the number of output channels is four times the number of input channels.

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out