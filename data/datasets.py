from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

class BaseDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform # it is used to specify any data transformations that should be applied to the dataset. For example, you might want to apply data augmentation techniques such as random cropping, flipping, or color jittering to increase the diversity of your training data and improve the generalization of your model. By passing a transform argument to the dataset, you can easily apply these transformations to your data during training or evaluation.
        # transform basic means that you can apply some transformations to your data, such as resizing, cropping, flipping, or normalizing the images. This can help improve the performance of your model by making it more robust to variations in the input data.

    def __len__(self): # this method is used to return the number of samples in the dataset. It is a required method for any class that inherits from the Dataset class in PyTorch. The __len__ method allows you to determine the size of your dataset, which can be useful for various purposes such as batching, shuffling, or splitting the data into training and validation sets. By implementing this method, you can ensure that your dataset can be properly utilized by PyTorch's data loading utilities.
        raise NotImplementedError

    def __getitem__(self, idx):# this method is used to retrieve a single sample from the dataset based on the provided index (idx). It is a required method for any class that inherits from the Dataset class in PyTorch. The __getitem__ method allows you to define how to access and return individual samples from your dataset, which can include loading the data, applying any necessary transformations, and returning the sample in a format suitable for training or evaluation. By implementing this method, you can ensure that your dataset can be properly utilized by PyTorch's data loading utilities.
        raise NotImplementedError


# if we want  the system to be mopduler such that it workas for any dataset we can create a base dataset class and then create specific dataset classes that inherit from the base class. This way, we can easily add new datasets by simply creating new classes that implement the necessary methods without having to modify the existing code. The base dataset class can provide common functionality and structure, while the specific dataset classes can handle the unique aspects of each dataset. This modular approach allows for greater flexibility and reusability in our codebase.
class CIFAR10Dataset(BaseDataset):# this class is a specific implementation of the BaseDataset class for the CIFAR-10 dataset. It inherits from the BaseDataset class and provides the necessary implementations for the __len__ and __getitem__ methods to work with the CIFAR-10 dataset. The CIFAR10Dataset class allows you to easily load and access the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. By using this class, you can efficiently work with the CIFAR-10 dataset in your machine learning projects.
    def __init__(self, root, train=True, transform=None):
        super().__init__(transform)
        self.dataset = CIFAR10(root=root, train=train, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
# u can change this data set to any dataset you want by just creating a new class that inherits from the BaseDataset class and implements the necessary methods. For example, if you want to work with the MNIST dataset, you can create a new class called MNISTDataset that inherits from BaseDataset and implements the __len__ and __getitem__ methods to load and access the MNIST dataset. This way, you can easily switch between different datasets without having to modify the existing code, making your project more flexible and reusable.
