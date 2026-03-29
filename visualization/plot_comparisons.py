import matplotlib.pyplot as plt

def compare_models(results): # compare_models is a function that takes in a list of results from different model experiments, and generates a bar plot comparing
    # the final validation accuracy of each model. The plot includes the model names on the x-axis and the final accuracy on the y-axis, allowing you to visually
    # compare the performance of different models. The generated plot is saved as "plots/resnet_training.png" and displayed on the screen.
    names = [r["model"] for r in results]
    accs = [r["history"]["val_acc"][-1] for r in results]

    plt.figure()
    plt.bar(names, accs)

    plt.xlabel("Model")
    plt.ylabel("Final Accuracy")
    plt.title("Model Comparison")
    plt.savefig("plots/resnet_training.png")
    plt.show()

def compare_speed(results):# compare_speed is a function that takes in a list of results from different model experiments, and generates a bar plot comparing 
    # the training time of each model.
    names = [r["model"] for r in results]
    times = [r["time"] for r in results]

    plt.figure()
    plt.bar(names, times)

    plt.xlabel("Model")
    plt.ylabel("Time (seconds)")
    plt.title("Training Time Comparison")
    plt.savefig("plots/resnet_training.png")
    plt.show()