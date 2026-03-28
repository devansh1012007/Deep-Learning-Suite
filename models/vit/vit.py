import torch
import torch.nn as nn
from models.vit.patch_embedding import PatchEmbedding
from models.vit.transformer_block import TransformerBlock
# improve ViT with dropout, stochastic depth, better positional encoding, better optimizers, data augmentation, etc. and see how it affects performance on CIFAR-10. 
# Also, experiment with different hyperparameters like embedding dimension, number of heads, depth of the model, etc. to find the best configuration for this task.
class ViT(nn.Module):
    def __init__(self,img_size=32,patch_size=4,embed_dim=128,depth=6,num_heads=4,num_classes=10):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim) # The PatchEmbedding class takes the image size, patch size, number of input channels (3 for RGB),
        # and embedding dimension as arguments. It divides the input image into patches and projects them into the embedding space.

        num_patches = self.patch_embed.num_patches # The number of patches is calculated based on the image size and patch size. 
        # For example, if the image size is 32 and the patch size is 4, there will be (32/4) * (32/4) = 64 patches. Each patch will be represented as a vector in the embedding space.

        # CLS token --> learnable parameter, during training, the model learns to use this token to capture relevant information from the patches for making predictions by attending to the patch embeddings and aggregating information from them. And then backpropagating the error signal to update the class token's parameters, allowing it to learn to capture relevant information from the patches for making predictions.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # The class token is a learnable parameter that is added to the sequence of patch embeddings. 
        #It serves as a summary representation of the entire input image and is used for classification tasks. 
        # During training, the model learns to use this token to capture relevant information from the patches for making predictions.

        # Positional embedding --> learnable parameter, during training, the model learns to use this positional information to understand the spatial relationships between patches, which is crucial for tasks like image classification where the arrangement of features matters.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) # The positional embedding is also a learnable parameter that is added to the patch embeddings to provide information about the position of each patch in the original image. 
        # This helps the model to understand the spatial relationships between patches, which is crucial for tasks

        # Transformer layers
        # here v r creating a list of Transformer blocks, where each block consists of multi-head self-attention and feed-forward layers.
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)# The TransformerBlock class implements a single block of the Transformer architecture, 
            # which consists of multi-head self-attention and feed-forward layers.
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim) # Layer normalization is applied to the output of the Transformer blocks to stabilize training and improve convergence.

        self.head = nn.Linear(embed_dim, num_classes) # The final classification head is a linear layer that takes the output of the class token (after processing through the Transformer blocks) and produces the final class predictions

    def forward(self, x): # The forward method defines how the input data flows through the model. 
        # It takes an input tensor x, which is expected to be of shape (B, 3, 32, 32) for a batch of images.
        B = x.size(0)

        x = self.patch_embed(x) # The input image is passed through the patch embedding layer, which divides it into patches and projects them into the embedding space.

        cls_tokens = self.cls_token.expand(B, -1, -1) # The class token is expanded to match the batch size. This allows the class token to be concatenated with the patch embeddings for each image in the batch.
        x = torch.cat((cls_tokens, x), dim=1) # The class token is concatenated with the patch embeddings along the sequence dimension (dim=1).

        x = x + self.pos_embed # The positional embedding is added to the combined class token and patch embeddings to provide positional information to the model.

        for block in self.blocks: # The combined embeddings are passed through each Transformer block in the sequence. Each block processes the input and produces an output that is fed into the next block.
            x = block(x)

        x = self.norm(x) # After processing through the Transformer blocks, layer normalization is applied to the output to stabilize training and improve convergence. 

        cls_output = x[:, 0] # The output corresponding to the class token (the first token in the sequence) is extracted. This token is expected to capture the relevant information from the entire input image for classification.
        return self.head(cls_output)