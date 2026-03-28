model = Seq2Seq(encoder, decoder).to(device)

criterion = nn.MSELoss()  # or CrossEntropy depending on task