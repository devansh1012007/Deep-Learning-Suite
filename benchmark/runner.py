
import time
config = {
    "model_name": "resnet18",
    "epochs": 50,
    "lr": 0.1
}
def run_experiment(config, build_model, build_data, trainer_cls, evaluator_cls): # run_experiment is a function that takes in a configuration dictionary, 
    # functions to build the model and data, and classes for the trainer and evaluator. It then runs the training and evaluation process for the specified model
    # and returns the results, including the training history, final accuracy, and total time taken for the experiment.
    start_time = time.time()

    # Data
    train_loader, val_loader = build_data(config)

    # Model
    model = build_model(config)

    # Trainer
    trainer = trainer_cls(
        model=model,
        optimizer=config["optimizer"],
        criterion=config["criterion"],
        device=config["device"]
    )

    evaluator = evaluator_cls(model, config["device"])

    # Train
    history = trainer.fit(
        train_loader,
        val_loader,
        evaluator,
        epochs=config["epochs"],
        ckpt_path=config["ckpt_path"]
    )

    total_time = time.time() - start_time
    
    return {
        "model": config["model_name"],
        "history": history,
        "accuracy": evaluator.evaluate(val_loader),
        "time": total_time
    }


def build_resnet(config): # build_resnet is a function that takes in a configuration dictionary and builds a ResNet model based on the specified parameters.
    # It initializes the model, optimizer, and loss function, and returns them for use in the training process.
    from models.resnet.resnet18 import ResNet18
    import torch.optim as optim
    import torch.nn as nn

    model = ResNet18(num_classes=10).to(config["device"])

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion

# maybe v need to make this for other models too, but for now we can just use this one for resnet