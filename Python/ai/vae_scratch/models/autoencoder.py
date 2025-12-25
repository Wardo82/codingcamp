import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Set the number of hidden units
        self.num_hidden = 8

        # Define the encoder part of the autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),  # input size: 784, output size: 256
            nn.ReLU(),  # apply the ReLU activation function
            nn.Linear(256, self.num_hidden),  # input size: 256, output size: num_hidden
            nn.ReLU(),  # apply the ReLU activation function
        )

        # Define the decoder part of the autoencoder
        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden, 256),  # input size: num_hidden, output size: 256
            nn.ReLU(),  # apply the ReLU activation function
            nn.Linear(256, 784),  # input size: 256, output size: 784
            nn.Sigmoid(),  # apply the sigmoid activation function to compress the output to a range of (0, 1)
        )

    def forward(self, x):
        # Pass the input through the encoder
        encoded = self.encoder(x)
        # Pass the encoded representation through the decoder
        decoded = self.decoder(encoded)
        # Return both the encoded representation and the reconstructed output
        return encoded, decoded

    def save(self, filepath: str = "autoencoder.pt"):
        """ Stores the model in .pt format for later use

        Reload as:

            model = autoencoder.AutoEncoder()
            model.load_state_dict(torch.load("autoencoder.pt", map_location=device))
            model.eval()
        """
        torch.save(self.state_dict(), filepath)

    def save_to_onnx(self, device, filepath: str = "autoencoder.onnx"):
        """
        Export the encoder part of the autoencoder to ONNX.
        """
        self.eval()

        dummy_input = torch.randn(1, 784, device=device)

        torch.onnx.export(
            self.encoder,
            dummy_input,
            filepath,
            input_names=["input"],
            output_names=["latent"],
            dynamic_axes={
                "input": {0: "batch"},
                "latent": {0: "batch"},
            },
            opset_version=11,
        )
