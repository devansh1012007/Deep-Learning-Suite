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


