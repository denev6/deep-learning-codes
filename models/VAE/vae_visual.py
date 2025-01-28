import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        x_recon = torch.sigmoid(self.fc_out(h))
        return x_recon


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "vae-2d.pth"

input_dim = 28 * 28  # MNIST
hidden_dim = 500
latent_dim = 2

model = VAE(input_dim, hidden_dim, latent_dim)
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
model.eval()

grid_size = 20
z_range = 3  # 99.7% fall within [-3, 3]
z1 = np.linspace(-z_range, z_range, grid_size)
z2 = np.linspace(-z_range, z_range, grid_size)
grid_latents = np.array([[x, y] for y in reversed(z2) for x in z1])

z_tensor = torch.tensor(grid_latents, dtype=torch.float32)
with torch.no_grad():
    latent_images = model.decoder(z_tensor)

fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
latent_images = latent_images.view(-1, 28, 28).numpy()

for i, ax in enumerate(axes.flatten()):
    ax.imshow(latent_images[i], cmap="gray")
    ax.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis("off")
plt.savefig("latent-2d.png", bbox_inches="tight")
plt.show()
