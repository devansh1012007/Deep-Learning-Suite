import torch
import torch.nn as nn

class PatchEmbedding(nn.Module): # The PatchEmbedding class is responsible for dividing the input image into patches and projecting them into an embedding space.
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x): # The forward method defines how the input data flows through the patch embedding layer. 
        # It takes an input tensor x, which is expected to be of shape (B, 3, 32, 32) for a batch of images.
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x