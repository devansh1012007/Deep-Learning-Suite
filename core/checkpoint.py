import torch
import os
# checkpointing is a technique used in machine learning to save the state of a model at a specific point during training. 
# This allows you to resume training from that point if needed, or to evaluate the model's performance on a validation set without having to retrain it from scratch.
#  Checkpointing can be particularly useful when training large models or when training takes a long time, as it allows you to save your progress and avoid losing valuable training time in case of interruptions or crashes.

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),# here we are saving optimizer setting 
        # optimizer is used to update the model's parameters based on the computed gradients during training.
        # By saving the optimizer's state, we can ensure that when we load the checkpoint later, we can resume training with the same optimization settings and continue updating the model's parameters effectively.
        # This is important for maintaining the continuity of the training process and achieving better performance when resuming from a checkpoint.
        "epoch": epoch # epoch is used to keep track of the current training epoch. 
        #By saving the epoch number in the checkpoint, we can easily resume training from the correct point when loading the checkpoint later. 
        # This allows us to continue training without having to start from the beginning, which can save time and computational resources, especially when training large models or when training takes a long time.
    }, path) # path is the location where the checkpoint will be saved. 
    #It can be a file path or a directory path, depending on how you want to organize your checkpoints.
    #  By specifying the path, you can easily manage and access your checkpoints for later use, such as resuming training or evaluating the model's performance on a validation set.

def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path)

    model.load_state_dict(ckpt["model"])

    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])# used to load the state of the optimizer from the checkpoint.
        #This allows you to resume training with the same optimization settings and continue updating the model's parameters effectively.
        # By loading the optimizer's state, you can ensure that the training process continues smoothly without any disruptions, and you can achieve better performance when resuming from a checkpoint.

    return ckpt["epoch"]