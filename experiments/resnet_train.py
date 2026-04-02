
from core.utils import set_seed, get_device
from data.datasets import CIFAR10Dataset
from data.transforms import get_train_transforms, get_val_transforms
from data.dataloader import create_dataloader
from core.trainer import Trainer
from core.evaluator import Evaluator
from models.resnet.resnet18 import ResNet18
import torch.nn as nn
import torch.optim as optim
import torch

# ResNet solves: Vanishing gradient problem

def run():
    set_seed(42)# this line is used to set the random seed for reproducibility. By setting the seed to a specific value (in this case, 42), we can ensure that the random operations in our code (such as shuffling data, initializing model parameters, etc.) will produce the same results every time we run the code. This is important for debugging and comparing results across different runs, as it allows us to have consistent and repeatable outcomes when training and evaluating our models.
    device = get_device()

    train_ds = CIFAR10Dataset("data", train=True, transform=get_train_transforms())# this line is used to create an instance of the CIFAR10Dataset class for the training dataset. The CIFAR10Dataset class is a specific implementation of the BaseDataset class for the CIFAR-10 dataset. By creating an instance of this class, we can easily load and access the CIFAR-10 training dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The transform argument is used to specify any data transformations that should be applied to the training dataset, such as data augmentation techniques like random cropping, flipping, or color jittering. This can help increase the diversity of the training data and improve the generalization of our model.
    val_ds = CIFAR10Dataset("data", train=False, transform=get_val_transforms())# this line is used to create an instance of the CIFAR10Dataset class for the validation dataset. The CIFAR10Dataset class is a specific implementation of the BaseDataset class for the CIFAR-10 dataset. By creating an instance of this class, we can easily load and access the CIFAR-10 validation dataset, which consists of 10,000 32x32 color images in 10 classes, with 1,000 images per class. The transform argument is used to specify any data transformations that should be applied to the validation dataset, such as converting the images to tensors and normalizing the pixel values. This ensures that the input data for evaluation is in the same format as the training data and maintains consistency in the pixel value range for accurate performance assessment.

    train_loader = create_dataloader(train_ds, 128)
    val_loader = create_dataloader(val_ds, 128, shuffle=False)

    model = ResNet18(num_classes=10).to(device)  # Phase 2
    model.to(device)

    criterion = nn.CrossEntropyLoss() # CrossEntropyLoss is a commonly used loss function for multi-class classification problems. 
    # it helps to measure the difference between the predicted class probabilities and the true class labels. 
    # It combines a softmax activation function with a negative log-likelihood loss, which makes it suitable for training classification models.
    # (a negative log-likelihood loss means that it calculates the negative logarithm of the predicted probabilities for the true class labels, which encourages the model to assign higher probabilities to the correct classes and lower probabilities to the incorrect classes.)
    # It combines LogSoftmax and NLLLoss in one single class. 
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)# SGD (Stochastic Gradient Descent) is an optimization algorithm used to minimize the loss function during training.
    # It updates the model's parameters based on the gradients of the loss with respect to the parameters.
    # The learning rate (lr) determines the step size at each iteration while moving toward a minimum
    # Momentum helps to accelerate the optimization process by adding a fraction of the previous update to the current update, which can help to navigate through local minima and improve convergence.
    # Weight decay is a regularization technique that adds a penalty to the loss function based on the magnitude of the model's parameters, which helps to prevent overfitting by encouraging the model to learn simpler patterns in the data.
    

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler   
    )# this line is used to create an instance of the Trainer class, which is responsible for managing the training process of the model. 
    #The Trainer class takes several arguments, including the model to be trained, the optimizer for updating the model's parameters, 
    # the loss criterion for calculating the loss during training, the device on which the training will be performed (e.g., CPU or GPU), 
    # and an optional learning rate scheduler for adjusting the learning rate during training. By creating an instance of the Trainer class with these arguments, 
    # we can easily manage and execute the training process for our model, allowing us to focus on defining our model architecture and training logic while 
    # ensuring that we have a structured and organized way to handle the training process.

    evaluator = Evaluator(model, device) # this line is used to create an instance of the Evaluator class, which is responsible for evaluating the performance of the model on the validation data. The Evaluator class takes the model and device as arguments. By creating an instance of the Evaluator class, we can easily assess how well our model is performing on unseen data by passing the validation dataloader to the evaluate method. The Evaluator class provides a structured way to organize and execute the evaluation process, allowing us to focus on defining our model architecture and training logic while ensuring that we can effectively evaluate its performance on the validation data.

    trainer.fit(train_loader, val_loader, evaluator, epochs=10, ckpt_path="checkpoints/resnet.pth")
    '''
    dataset setup
    dataloader setup
    model creation
    optimizer + scheduler
    trainer + evaluator
    training execution
    '''



