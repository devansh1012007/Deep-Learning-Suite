import torch
# v can other matrics such as precision, recall, F1-score, etc. by defining additional functions similar to the accuracy function. 
# These functions can be used to evaluate the performance of our model on different aspects and provide a more comprehensive understanding of its performance. 
# By implementing these metrics, we can gain insights into how well our model is performing and identify areas for improvement.

# make this more moduler in future by creating a separate file for metrics and then importing the necessary functions into the evaluator.py file. This way, we can easily add new metrics or modify existing ones without having to change the code in the evaluator.py file, making our codebase more organized and maintainable.

def accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)# argmax is used to get the predicted class labels from the model's output probabilities. It returns the indices of the maximum values along the specified dimension (dim=1 in this case, which corresponds to the class dimension). By applying argmax to the model's output, we can obtain the predicted class labels for each sample in the batch.
    return (preds == targets).float().mean().item() # this will return the average accuracy of the model's predictions compared to the true labels. The expression (preds == targets) creates a boolean tensor where each element is True if the predicted label matches the true label and False otherwise. 
                                                    #By converting this boolean tensor to a float tensor using .float(), we can calculate the mean accuracy by taking the average of the True values (which are treated as 1) and False values (which are treated as 0). Finally, .item() is used to extract the scalar value from the resulting tensor, giving us the overall accuracy of the model's predictions. 
