def test_resnet_shape():
    model = ResNet18(num_classes=10)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)

    assert y.shape == (1, 10)