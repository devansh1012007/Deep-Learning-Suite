import torchvision.transforms as T
# for traning
def get_train_transforms(): # this funcrionm is for defining the transformations that will be applied to the training data. It returns a composition of several transformations, including random cropping, random horizontal flipping, converting the image to a tensor, and normalizing the pixel values. These transformations are commonly used in data augmentation techniques to increase the diversity of the training data and improve the generalization of the model. By applying these transformations to the training data, we can help prevent overfitting and improve the performance of our model on unseen data.
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))# normanize is for scaling the pixel values of the images to a specific range. In this case, the mean is set to 0.5 and the standard deviation is set to 0.5 for each channel (assuming the images are in RGB format). This normalization helps to center the pixel values around zero and scale them to a range of [-1, 1]. Normalizing the pixel values can improve the convergence of the model during training and can also help to reduce the impact of varying lighting conditions or other factors that may affect the pixel values in the images.
    ])# there are several other transformations that can be applied to the training data, such as color jittering, random rotation, or random scaling. The specific transformations you choose will depend on the characteristics of your dataset and the requirements of your model. It is important to experiment with different transformations to find the ones that work best for your specific task and dataset.
# fir validation 
def get_val_transforms():# this fuction is for defining the transformations that will be applied to the validation data. It returns a composition of two transformations: converting the image to a tensor and normalizing the pixel values. These transformations are typically used for the validation data to ensure that the input data is in the same format as the training data and to maintain consistency in the pixel value range. By applying these transformations to the validation data, we can evaluate the performance of our model on unseen data and ensure that it generalizes well to new examples.
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])

# the difference between the training and validation transformations is that the training transformations include data augmentation techniques such as random cropping and random horizontal flipping, which are not applied to the validation data. The training transformations are designed to increase the diversity of the training data and improve the generalization of the model, while the validation transformations are focused on ensuring that the input data is in the same format as the training data and maintaining consistency in the pixel value range for evaluation purposes.
# U can make this moduler 