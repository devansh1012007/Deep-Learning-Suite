def plot_lr(optimizer, scheduler, steps=100):# plot_lr is a function that takes in an optimizer, a learning rate scheduler, and the number of steps to simulate.
    # It generates a plot of the learning rate schedule by recording the learning rate at each step and plotting it over time. This allows you to visualize how 
    # the learning rate changes throughout the training process, which can be useful for understanding the behavior of the model and identifying any issues with the learning rate schedule.
    lrs = []

    for _ in range(steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    import matplotlib.pyplot as plt
    plt.plot(lrs)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Step")
    plt.ylabel("LR")
    plt.savefig("plots/resnet_training.png")
    plt.show()


# post training, we can call this function to visualize the learning rate schedule used during training. By plotting the learning rates over the training steps, we can gain insights into how the learning rate changed throughout the training process, which can help us understand the behavior of the model and potentially identify any issues or areas for improvement in our training strategy.
'''
from visualization.plot_metrics import plot_training

result = run_experiment(...)

plot_training(result["history"])
'''