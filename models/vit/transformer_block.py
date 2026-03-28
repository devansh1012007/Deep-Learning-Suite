import torch.nn as nn

class TransformerBlock(nn.Module): # The TransformerBlock class implements a single block of the Transformer architecture, 
    # which consists of multi-head self-attention and feed-forward layers.
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        
        # we don't do masked attention here because in ViT, the input sequence is not autoregressive (like in language models), but rather consists of all the patch embeddings and the class token at once.
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) # The multi-head attention layer allows the model to attend to 
        # different parts of the input sequence simultaneously, which helps in capturing complex relationships between patches in the input image.

        self.norm2 = nn.LayerNorm(embed_dim)

        # The feed-forward network consists of two linear layers with a GELU activation function in between. The first linear layer expands the embedding dimension by a factor of mlp_ratio, and the second linear layer projects it back to the original embedding dimension.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), 
            nn.GELU(), # The GELU activation function is used in the feed-forward network to introduce non-linearity, which helps the model to learn more complex representations.
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x): # The forward method defines how the input data flows through the Transformer block. It takes an input tensor x, which is expected to be of shape (B, N, D), where B is the batch size, N is the sequence length (number of patches plus one for the class token), and D is the embedding dimension.
        # Attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # Feed Forward
        x = x + self.mlp(self.norm2(x))

        return x