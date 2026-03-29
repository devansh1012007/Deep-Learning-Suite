def lr_ablation(): # lr_ablation is a function that performs an ablation study on the learning rate hyperparameter. It runs multiple experiments with different 
    # learning rates and collects the results for comparison.
    lrs = [0.1, 0.01, 0.001]
    results = []

    for lr in lrs:
        config = {"lr": lr, ...}
        result = run_experiment(config, ...)
        results.append(result)

    return results


def model_ablation(): # model_ablation is a function that performs an ablation study on the model architecture. It runs multiple experiments with different 
    # models and collects the results for comparison.
    models = ["resnet18", "vit"]

    for m in models:
        config = {"model_name": m, ...}
        run_experiment(config, ...)


depths = [4, 6, 8]  # for ViT

import json

def save_results(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=4)