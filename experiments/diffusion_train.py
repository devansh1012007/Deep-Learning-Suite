model = DiffusionModel(unet, scheduler).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)