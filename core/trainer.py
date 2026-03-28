# so tv r making a flexable trainer that can be used for any model and any task
'''
goal for the project :
A reusable Dataset system
A flexible Trainer engine
A clean experiment pipeline
Logging + checkpointing + reproducibility
'''
import torch
from core.checkpoint import save_checkpoint

class Trainer:
    def __init__(self,model,optimizer,criterion,device,scheduler=None,hooks=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.hooks = hooks or []

    def train_epoch(self, dataloader): # this is for training the model for one epoch. 
        # It takes a dataloader as input, which provides batches of training data. 
        # The method iterates over the dataloader, passing each batch of data through the model to obtain predictions, calculating the loss using the specified criterion, and then performing backpropagation to update the model's parameters using the optimizer. 
        # The total loss for the epoch is accumulated and returned at the end. This method allows us to train our model on the provided training data and optimize its performance over multiple epochs.
        self.model.train()# train() is a method in PyTorch that sets the model to training mode. When a model is in training mode, certain layers such as dropout and batch normalization behave differently compared to evaluation mode. For example, dropout layers will randomly drop out units during training to prevent overfitting, and batch normalization layers will use the batch statistics instead of the running statistics. This is important because it allows the model to learn and generalize better during training. By calling self.model.train(), we can ensure that the model is in the appropriate mode for training and that it behaves correctly when processing the training data.
        total_loss = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device) # this line is used to move the input data (x) and the corresponding labels (y) to the specified device (e.g., GPU or CPU) for training. By using .to(self.device), we can ensure that the data is processed on the appropriate device, which can help speed up training and improve performance, especially when using a GPU. 
            # This is an important step in the training process, as it allows us to leverage the computational power of the device and efficiently train our model on the provided data.

            self.optimizer.zero_grad() # this line is used to reset the gradients of the model's parameters before performing backpropagation. By calling self.optimizer.zero_grad(), we can ensure that the gradients from the previous training step do not accumulate and interfere with the current training step. This is important because if we do not reset the gradients, they will be added to the existing gradients, which can lead to incorrect updates of the model's parameters and hinder the training process. By zeroing out the gradients at the beginning of each training step, we can ensure that the optimization process is based solely on the current batch of data and that the model's parameters are updated correctly.

            outputs = self.model(x)# this line is used to pass the input data (x) through the model to obtain the predicted outputs. The model takes the input data and processes it through its layers to generate predictions, which are stored in the variable outputs. These outputs can then be compared to the true labels (y) to calculate performance metrics such as accuracy, which can help us evaluate how well the model is performing on the given data.
            loss = self.criterion(outputs, y)# this line is used to calculate the loss between the model's predictions (outputs) and the true labels (y) using the specified criterion. The criterion is a loss function that measures the discrepancy between the predicted outputs and the true labels, and it is used to guide the optimization process during training. By calculating the loss, we can assess how well our model is performing on the given data and use it to update the model's parameters through backpropagation, ultimately improving its performance over time.
            # FOR GCN 
            # outputs = model(x, adj)
            # loss = criterion(outputs, y)
            loss.backward()# this line is used to perform backpropagation, which calculates the gradients of the loss with respect to the model's parameters. By calling loss.backward(), we can compute the gradients for all the parameters in the model, which will be used by the optimizer to update the parameters during the optimization step. This is a crucial step in the training process, as it allows us to optimize the model's performance by adjusting its parameters based on the computed gradients.

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)# this line is used to clip the gradients of the model's parameters to a maximum norm of 1.0. Gradient clipping is a technique used to prevent exploding gradients during training, which can occur when the gradients become too large and cause instability in the optimization process.
            #By using torch.nn.utils.clip_grad_norm_, we can ensure that the gradients are scaled down if their norm exceeds the specified threshold (in this case, 1.0), which helps maintain stable training and prevents issues such as divergence or NaN values in the model's parameters.

            self.optimizer.step()# this line is used to perform an optimization step, which updates the model's parameters based on the computed gradients. By calling self.optimizer.step(), we can apply the optimization algorithm (e.g., SGD, Adam, etc.) to adjust the model's parameters in a way that minimizes the loss function. This is a crucial step in the training process, as it allows us to improve the model's performance by iteratively updating its parameters based on the feedback provided by the loss function and the computed gradients.

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader, evaluator, epochs, ckpt_path):# this method is used to train the model for a specified number of epochs.
        # It takes the training and validation dataloaders, an evaluator for assessing the model's performance, the number of epochs to train for, and a checkpoint path for saving the model's state.
        # The method iterates over the specified number of epochs, calling the train_epoch method to train the model on the training data and then using the evaluator to assess its performance on the validation data. After each epoch, it saves a checkpoint of the model's state using the save_checkpoint function. This allows us to track the model's progress during training and save its state at regular intervals for later use or analysis.
        for epoch in range(epochs):

            for hook in self.hooks: # this loop is used to execute any hooks that are defined in the trainer before the start of each epoch. 
                #Hooks are a way to execute some code at specific points during the training process, such as at the beginning or end of an epoch. #
                # By iterating over the self.hooks list and calling the on_epoch_start method for each hook, we can allow users to define custom behavior that should be executed at the start of each epoch, such as logging, adjusting learning rates, or performing any other necessary actions. This provides flexibility and extensibility to the training process, allowing users to customize it according to their specific needs and requirements.
                hook.on_epoch_start(self)

            train_loss = self.train_epoch(train_loader)# this line is used to train the model for one epoch using the provided training dataloader (train_loader).
            # The train_epoch method is called, which iterates over the training data, performs forward and backward passes, and updates the model's parameters based on the computed loss. 
            # The average loss for the epoch is returned and stored in the variable train_loss, which can be used for logging or monitoring the training process.
            
            val_acc = evaluator.evaluate(val_loader)# this line is used to evaluate the model's performance on the validation data using the provided evaluator. The evaluate method of the evaluator is called, which takes the validation dataloader (val_loader) as input and returns the calculated accuracy (or other performance metric) for the model on the validation data. This allows us to assess how well our model is performing on unseen data and make informed decisions about its performance and potential areas for improvement.
            
            if self.scheduler: # this line is used to check if a LEARNING RATE scheduler is defined in the trainer.
                #If a scheduler is present, it will be called to update the learning rate after each epoch.
                self.scheduler.step()

            save_checkpoint(self.model, self.optimizer, epoch, ckpt_path)

            print(f"Epoch {epoch}: Loss={train_loss:.4f}, ValAcc={val_acc:.4f}")

            for hook in self.hooks: # this loop is used to execute any hooks that are defined in the trainer at the end of each epoch. 
                # By iterating over the self.hooks list and calling the on_epoch_end method for
                # each hook, we can allow users to define custom behavior that should be executed at the end of each epoch, such as logging, adjusting learning rates, or performing any other necessary actions. This provides flexibility and extensibility to the training process, allowing users to customize it according to their specific needs and requirements.
                hook.on_epoch_end(self)