# import necessary libraries and modules later
import torch
# import ResNet18, Seq2Seq, ViT, GCN from their respective files later

def test_resnet_shape(): # This test checks if the ResNet18 model produces the correct output shape when given an input tensor of shape (1, 3, 32, 32). 
    # The expected output shape is (1, 10), which corresponds to the number of classes in the output layer.
    model = ResNet18(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)

    assert y.shape == (1, 10)
def test_seq2seq(): # This test checks if the Seq2Seq model produces the correct output shape when given input tensors of shape (2, 10, 32) for both the source and target sequences. 
    # The expected output shape is (2, 9, 32), which corresponds to the batch size, sequence length (target sequence length minus one for the start token), and output size.
    model = Seq2Seq(...)
    x = torch.randn(2, 10, 32)
    y = torch.randn(2, 10, 32)

    out = model(x, y)

    assert out.shape == (2, 9, 32)

def test_vit(): # This test checks if the ViT (Vision Transformer) model produces the correct output shape when given an input tensor of shape (2, 3, 32, 32). 
    # The expected output shape is (2, 10), which corresponds to the batch size and the number of classes in the output layer.
    model = ViT()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)

    assert y.shape == (2, 10)

def test_gnn(): # This test checks if the GCN (Graph Convolutional Network) model produces the correct output shape when given input tensors of shape (10, 16) for node features and (10, 10) for the adjacency matrix. 
    # The expected output shape is (10, 3), which corresponds to the number of nodes and the number of classes in the output layer.
    model = GCN(16, 32, 3)

    x = torch.randn(10, 16)
    adj = torch.randint(0, 2, (10, 10)).float()

    out = model(x, adj)

    assert out.shape == (10, 3)