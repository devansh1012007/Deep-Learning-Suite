import torch
import random
import numpy as np

def set_seed(seed: int):# seed is an integer value that is used to initialize the random number generator. By setting the seed, you can ensure that the same sequence of random numbers is generated each time you run the code, which can be useful for reproducibility and debugging purposes.
    # in training machine learning models, we often use random numbers for various purposes, such as initializing model weights, shuffling data, or applying random transformations. By setting a seed, we can ensure that the same sequence of random numbers is generated each time we run the code, which can help us reproduce our results and debug our code more effectively.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)# torch.manual_seed(seed) is a function in the PyTorch library that sets the seed for generating random numbers on the CPU. By calling this function with a specific seed value, you can ensure that the same sequence of random numbers is generated each time you run your code, which can be useful for reproducibility and debugging purposes when training machine learning models.
    torch.cuda.manual_seed_all(seed)# torch.cuda.manual_seed_all(seed) is a function in the PyTorch library that sets the seed for generating random numbers on all available CUDA devices (GPUs). By calling this function with a specific seed value, you can ensure that the same sequence of random numbers is generated each time you run your code on the GPU, which can be useful for reproducibility and debugging purposes when training machine learning models using CUDA.

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


import yaml

def load_config(path): # load_config is a function that takes in a file path to a YAML configuration file, reads the contents of the file, and returns the configuration as a Python dictionary. This allows you to easily manage and access your experiment configurations in a structured format.
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def merge_configs(default, specific): # merge_configs is a function that takes in a default configuration dictionary and a specific configuration dictionary, and 
    # merges the two dictionaries together. The specific configuration values will override the default values, allowing you to easily customize your experiment 
    # configurations while still maintaining a base set of default parameters.
    for key, value in specific.items():
        if isinstance(value, dict) and key in default:
            default[key] = merge_configs(default[key], value)
        else:
            default[key] = value
    return default

default = load_config("configs/default.yaml")
model_cfg = load_config("configs/resnet.yaml")

config = merge_configs(default, model_cfg)

run(config)