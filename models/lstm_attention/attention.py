import torch
import torch.nn as nn

class Attention(nn.Module): # The Attention mechanism allows the decoder to focus on different parts of the encoder's outputs at each time step, which can improve performance on tasks like machine translation.
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size * 2, hidden_size) # The attention scores are computed using a feedforward network that takes the decoder's hidden state and the encoder's outputs as input.
        self.v = nn.Linear(hidden_size, 1) # The attention scores are computed using a feedforward network that takes the decoder's hidden state and the encoder's outputs as input. 
        # The output is a single score for each encoder output, which is then normalized to get the attention weights.

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)# The decoder's hidden state is repeated across the sequence length of the encoder's outputs so that we can compute attention scores for each encoder output in parallel.

        energy = torch.tanh(self.W(torch.cat([hidden, encoder_outputs], dim=2))) # The attention scores are computed by concatenating the decoder's hidden state and the encoder's outputs, passing them through a feedforward network, and applying a non-linearity.
        scores = self.v(energy).squeeze(-1)

        weights = torch.softmax(scores, dim=1) # The attention scores are normalized using softmax to get the attention weights, which indicate how much focus the decoder should put on each encoder output.

        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)# The context vector is computed as a weighted sum of the encoder's outputs, 
        # where the weights are the attention weights. This context vector is then used by the decoder to generate the next output.
        context = context.squeeze(1)

        return context, weights