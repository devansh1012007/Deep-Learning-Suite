# this is a hook class which is used to define the hooks that can be used in the trainer.
# hooks are a way to execute some code at specific points during the training process.
class Hook:
    def on_epoch_start(self, trainer): pass # you can specify custom behavior that should be executed at the beginning of each epoch, such as logging, adjusting learning rates, or performing any other necessary actions. This allows for greater flexibility and extensibility in the training process, enabling users to customize it according to their specific needs and requirements.
    def on_epoch_end(self, trainer): pass
    def on_batch_start(self, trainer): pass # you can specify custom behavior that should be executed before processing each batch of data, such as logging, adjusting learning rates, or performing any other necessary actions. This allows for greater flexibility and extensibility in the training process, enabling users to customize it according to their specific needs and requirements.
    ### v can do Custom Logging, Learning Rate Adjustments, Early Stopping, frezing layers, Adversarial Training, Debugging and Visualization, Gradient Accumulation Control, Dynamic Data Augmentation, Custom Metrics Calculation, etc. 
    def on_batch_end(self, trainer): pass