import matplotlib.pyplot as plt

def plot_training(history, title="Training Curve"): # plot_training is a function that takes in a training history dictionary, which contains the training loss
    # and validation accuracy for each epoch, and generates a plot of the training curve. The plot includes the training loss and validation accuracy over the 
    # epochs, allowing you to visualize the model's performance during training. The title parameter allows you to specify a title for the plot, and the generated 
    # plot is saved as "plots/resnet_training.png" and displayed on the screen.
    
    epochs = range(len(history["train_loss"]))

    plt.figure()

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")

    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.savefig("plots/resnet_training.png")
    plt.show()

def smooth(values, weight=0.9): # smooth is a function that takes in a list of values and a weight parameter, and applies an exponential moving average to the 
    # values to create a smoothed version of the data.
    smoothed = []
    last = values[0]

    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)

    return smoothed