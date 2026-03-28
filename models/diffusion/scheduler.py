import torch

class NoiseScheduler: # this class is responsible for adding noise to the input data during the training process. 
    #It defines a schedule for how much noise to add at each timestep, and it provides a method to add noise to the input data based on the current timestep. 
    # The NoiseScheduler is an essential component of the diffusion model, as it allows us to simulate the process of adding noise to the data and training 
    # the model to predict that noise, which is crucial for learning how to reverse the diffusion process and generate new samples.
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        self.beta = torch.linspace(beta_start, beta_end, timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)

        alpha_hat_t = self.alpha_hat[t].view(-1, 1, 1, 1)

        noisy = (
            torch.sqrt(alpha_hat_t) * x +
            torch.sqrt(1 - alpha_hat_t) * noise
        )

        return noisy, noise
    
# You’re not predicting image directly; You’re predicting: the noise that was added