def train_rqvae():
    data = torch.randn(1000, 10)  # 1000 samples, 10 features
    train_dataset = MyDataset(data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = RQVAE(input_dim=10, latent_dim=5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.float()  # Ensure the data is in float32 format
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            recon_batch, z = model(batch_data)

            # Compute the loss (e.g., MSE loss for reconstruction + KL divergence for VAE)
            # Replace with your actual loss calculation
            reconstruction_loss = F.mse_loss(recon_batch, batch_data, reduction='sum')
            # Example of a simple KL divergence term for VAE (replace with your method)
            kl_loss = -0.5 * torch.sum(1 + z - z.pow(2) - z.exp())  # Example KL loss

            # Total loss (can combine reconstruction and KL losses)
            loss = reconstruction_loss + kl_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print statistics at the end of the epoch
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Training finished!")