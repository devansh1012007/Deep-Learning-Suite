import torch
from core.metrics import accuracy

class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad() # this decorator is used to disable gradient calculation during the evaluation phase.
    #By using @torch.no_grad(), we can save memory and computational resources since we don't need to compute gradients for the model's parameters when we are only interested in evaluating its performance on the validation or test data.
    #This is particularly useful when we are working with large models or datasets, as it can help speed up the evaluation process and reduce memory usage.
    # the reason gradiant decent is not imp is bcoz during evaluation, we are not updating the model's parameters based on the loss or performance metrics(which happens in training btw for back prop and for optimization).
    # Instead, we are simply using the model to make predictions and evaluate its performance on the validation or test data. 
    # Since we are not performing any optimization steps or backpropagation during evaluation, there is no need to calculate gradients for the model's parameters. This allows us to save memory and computational resources, as we can skip the gradient calculation step and focus solely on evaluating the model's performance.
    
    def evaluate(self, dataloader):# when we call the evaluate method, we pass in a dataloader that contains the validation or test data. The method will iterate over the dataloader, passing each batch of data through the model to obtain predictions, and then calculate the accuracy of those predictions compared to the true labels. The total accuracy is accumulated across all batches and returned as the final evaluation result. This allows us to assess how well our model is performing on unseen data and make informed decisions about its performance and potential areas for improvement.
        # evaluation will need ur gpu as it basicly takes the model and pass the validation data through it to get the predictions, and then compare those predictions to the true labels to calculate metrics such as accuracy. 
        self.model.eval()# eval () is a method in PyTorch that sets the model to evaluation mode. When a model is in evaluation mode, certain layers such as dropout and batch normalization behave differently compared to training mode. For example, dropout layers will not randomly drop out units during evaluation, and batch normalization layers will use the running statistics instead of the batch statistics. This is important because it ensures that the model's behavior is consistent during evaluation and that the performance metrics are calculated correctly. By calling self.model.eval(), we can ensure that the model is in the appropriate mode for evaluation and that we get accurate results when evaluating its performance on the validation or test data.
        total_acc = 0 # this variable is used to accumulate the total accuracy across all batches in the dataloader. It is initialized to 0 at the beginning of the evaluation process and will be updated with the accuracy calculated for each batch of data. By keeping track of the total accuracy, we can calculate the average accuracy at the end of the evaluation process by dividing the total accuracy by the number of batches or samples in the dataloader. This allows us to get an overall measure of the model's performance on the validation or test data.

        for x, y in dataloader: # here x, y are the input data and corresponding labels from the dataloader. The dataloader is an iterable that provides batches of data during the evaluation process. By iterating over the dataloader, we can access each batch of input data (x) and its corresponding labels (y) to evaluate the model's performance on that batch. This allows us to calculate metrics such as accuracy for each batch and accumulate the results to get an overall evaluation of the model's performance on the entire dataset.
            x, y = x.to(self.device), y.to(self.device)

            outputs = self.model(x) # this line is used to pass the input data (x) through the model to obtain the predicted outputs.
            #The model takes the input data and processes it through its layers to generate predictions, which are stored in the variable outputs. 
            #These outputs can then be compared to the true labels (y) to calculate performance metrics such as accuracy, which can help us evaluate how well the model is performing on the given data.
            total_acc += accuracy(outputs, y)

        return total_acc / len(dataloader)
    