from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    return DataLoader(# DataLoader is a PyTorch class that provides an efficient way to load and iterate over a dataset. It takes care of batching, shuffling, and parallel loading of data, making it easier to work with large datasets during training or evaluation. By using a DataLoader, you can easily manage the data loading process and improve the performance of your model by efficiently utilizing system resources.
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
# u can chyage the valuews of the parameters when creating the dataloader. For example, you can set shuffle to False if you don't want to shuffle the data during training, or you can adjust the num_workers parameter to control the number of subprocesses used for data loading based on your system's capabilities. The batch_size parameter can also be modified to suit your specific needs and memory constraints. By customizing these parameters, you can optimize the data loading process for your particular use case and improve the efficiency of your training or evaluation pipeline.
# to make it moduler ! 
# add a system that allows us to easily switch between different datasets and dataloaders without having to modify the existing code. This can be achieved by creating a base dataset class and a base dataloader function that can be reused for different datasets. 