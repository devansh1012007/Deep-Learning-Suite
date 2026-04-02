import torch
import torch.nn as nn

class LSTMCell(nn.Module): # LSTMCell is a single time step of an LSTM, while the Encoder and Decoder will use it to process sequences.
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(input_size + hidden_size, 4 * hidden_size) # We compute all gates in one go for efficiency. The output is split into 4 parts for the forget, input, candidate, and output gates.

    def forward(self, x, h_prev, c_prev):# x is the input at the current time step, h_prev is the hidden state from the previous time step, and c_prev is the cell state from the previous time step.
        combined = torch.cat([x, h_prev], dim=1) # We concatenate the input and the previous hidden state to compute the gates together.

        gates = self.W(combined)
        f, i, g, o = gates.chunk(4, dim=1)# We split the output of the linear layer into four parts, each corresponding to one of the gates.

        f = torch.sigmoid(f)# The forget gate determines how much of the previous cell state to keep.
        i = torch.sigmoid(i) # The input gate determines how much of the new candidate values to add to the cell state.
        g = torch.tanh(g) # The candidate values are the new information that could be added to the cell state.
        o = torch.sigmoid(o) # The output gate determines how much of the cell state to output as the new hidden state.

        c = f * c_prev + i * g # The new cell state is a combination of the previous cell state (modulated by the forget gate) and the new candidate values (modulated by the input gate).
        h = o * torch.tanh(c) # The new hidden state is the cell state modulated by the output gate.

        return h, c
    
    def forward_sequence(self, x): # This method processes an entire sequence of inputs. 
        # It initializes the hidden and cell states to zeros and iterates through the sequence, updating the states at each time step.
        h, c = init_states = (torch.zeros(x.size(0), self.hidden_size, device=x.device), # The hidden and cell states are initialized to zeros. The batch size is determined by the first dimension of the input x, and the hidden size is determined by the LSTMCell's configuration.
                                  torch.zeros(x.size(0), self.hidden_size, device=x.device))

        outputs = []
        seq_len = x.size(1) # The sequence length is determined by the second dimension of the input x. We loop through each time step in the sequence, updating the hidden and cell states using the forward method defined above.
        for t in range(seq_len):
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(h)

        return torch.stack(outputs, dim=1)# The outputs from each time step are collected and stacked together to form the final output sequence.