import torch
import torch.nn as nn

class DiffusionModel(nn.Module): # this is the main model that combines the UNet and the NoiseScheduler. 
    # It takes in an image, adds noise to it using the NoiseScheduler, and then passes the noisy image through the UNet to predict the noise. 
    # The forward method returns both the predicted noise and the true noise, which can be used to calculate the loss during training. 
    # The sample method is used to generate new images by reversing the diffusion process, starting from random noise and iteratively denoising it using the UNet and the NoiseScheduler.
    def __init__(self, unet, scheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler

    def forward(self, x):
        batch_size = x.size(0)

        t = torch.randint(0, self.scheduler.timesteps, (batch_size,), device=x.device)

        noisy_x, noise = self.scheduler.add_noise(x, t)

        pred_noise = self.unet(noisy_x)

        return pred_noise, noise
    
    # reversing the diffusion process to generate new samples from pure noise. \
    # The sample method starts with a random noise tensor and iteratively denoises it using the UNet model and the noise scheduler until it reaches a clean image. 
    # This allows us to generate new images that are similar to the training data by starting from random noise and applying the learned denoising process.
    #def backward(self, x):
    # maybe v need add backward method to reverse the diffusion process, but we can also do this in the sample method itself. 
    # We will see later when we implement the sampling process. For now, we will focus on the forward method and the training process.
    
    @torch.no_grad()
    def sample(self, shape): # this method is used to generate new images by reversing the diffusion process, starting from random noise and iteratively denoising
        # it using the UNet and the NoiseScheduler. The shape parameter specifies the shape of the generated images, and the method returns a tensor containing 
        # the generated images. By calling this method, we can create new samples that are similar to the training data, allowing us to explore the learned 
        # distribution of the data and generate novel images based on that distribution.
        device = next(self.parameters()).device
        x = torch.randn(shape).to(device)

        for t in reversed(range(self.scheduler.timesteps)):# this loop iterates through the timesteps in reverse order, starting from the last timestep and going back to the first. 
            #This is done to reverse the diffusion process and generate new images from pure noise. 
            # By iterating in reverse, we can apply the denoising process step by step, gradually transforming the random noise into a clean image that resembles the training data.
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

            pred_noise = self.unet(x)

            alpha = self.scheduler.alpha[t]
            alpha_hat = self.scheduler.alpha_hat[t]

            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred_noise
            )

        return x