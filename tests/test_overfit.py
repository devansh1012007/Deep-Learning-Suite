def test_overfit(trainer, dataloader):
    for _ in range(50):
        loss = trainer.train_epoch(dataloader)# this line is used to train the model for one epoch using the provided dataloader. The train_epoch method of the trainer is called, which iterates over the data in the dataloader, performs forward and backward passes, and updates the model's parameters based on the computed loss. The average loss for the epoch is returned and stored in the variable loss.
        #By training the model for multiple epochs using this line, we can observe how the loss changes over time and assess whether the model is overfitting to the training data or not.

    print("Final loss:", loss)