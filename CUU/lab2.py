import os
import argparse
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from tqdm import trange


class CVAE(nn.Module):
    def __init__(self, img_size=28, img_channels=1, latent_dim=20, n_classes=10, hidden_dim=400):
        super().__init__()
        self.img_size = img_size
        self.input_dim = img_channels * img_size * img_size
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # We'll append one-hot label to flattened image for encoder
        enc_input = self.input_dim + n_classes
        self.fc1 = nn.Linear(enc_input, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: take z and label
        dec_input = latent_dim + n_classes
        self.fc_dec1 = nn.Linear(dec_input, hidden_dim)
        self.fc_dec_out = nn.Linear(hidden_dim, self.input_dim)

    def encode(self, x, y_onehot):
        # x: [B, C, H, W] -> flatten
        B = x.size(0)
        x_flat = x.view(B, -1)
        inp = torch.cat([x_flat, y_onehot], dim=1)
        h = F.relu(self.fc1(inp))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y_onehot):
        inp = torch.cat([z, y_onehot], dim=1)
        h = F.relu(self.fc_dec1(inp))
        out = torch.sigmoid(self.fc_dec_out(h))
        B = out.size(0)
        out_img = out.view(B, 1, self.img_size, self.img_size)
        return out_img

    def forward(self, x, y_onehot):
        mu, logvar = self.encode(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y_onehot)
        return recon, mu, logvar


# Loss: BCE reconstruction + KL
def loss_function(recon_x, x, mu, logvar):
    # flatten
    BCE = F.binary_cross_entropy(recon_x.view(-1, recon_x.numel() // recon_x.size(0)),
                                 x.view(-1, x.numel() // x.size(0)), reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD


def one_hot(labels, num_classes, device):
    B = labels.size(0)
    y = torch.zeros(B, num_classes, device=device)
    y.scatter_(1, labels.view(-1,1), 1)
    return y


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_kld = 0.0
    for x, labels in dataloader:
        x = x.to(device)
        labels = labels.to(device)
        yoh = one_hot(labels, model.n_classes, device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x, yoh)
        loss, bce, kld = loss_function(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_bce += bce.item()
        total_kld += kld.item()
    n = len(dataloader.dataset)
    return total_loss / n, total_bce / n, total_kld / n


def test_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device)
            labels = labels.to(device)
            yoh = one_hot(labels, model.n_classes, device)
            recon, mu, logvar = model(x, yoh)
            loss, _, _ = loss_function(recon, x, mu, logvar)
            total_loss += loss.item()
    return total_loss / len(dataloader.dataset)


def sample_and_save(model, device, samples_per_class=10, out_path='grid_samples.png'):
    model.eval()
    n = model.n_classes
    samples = []
    with torch.no_grad():
        for cls in range(n):
            # generate several samples per class
            y = torch.zeros(samples_per_class, n, device=device)
            y[:, cls] = 1.0
            z = torch.randn(samples_per_class, model.latent_dim, device=device)
            gen = model.decode(z, y)  # [B,1,28,28]
            samples.append(gen)
    samples = torch.cat(samples, dim=0)  # [n * s, 1, 28,28]
    grid = utils.make_grid(samples, nrow=samples_per_class, pad_value=1.0)
    plt.figure(figsize=(samples_per_class, ceil(n/2)))
    plt.axis('off')
    plt.imshow(grid.permute(1,2,0).cpu().numpy(), cmap=None)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved generated grid to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--out', type=str, default='grid_samples.png')
    args = parser.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device:', device)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CVAE(img_size=28, img_channels=1, latent_dim=args.latent_dim, n_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_bce, train_kld = train_epoch(model, train_loader, optimizer, device)
        test_loss = test_epoch(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} (BCE {train_bce:.4f}, KLD {train_kld:.4f}) | Test loss: {test_loss:.4f}")

    # Сохраняем модель
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/cvae_mnist.pth')
    print('Model saved to checkpoints/cvae_mnist.pth')

    # Сэмплы
    sample_and_save(model, device, samples_per_class=10, out_path=args.out)


if __name__ == '__main__':
    main()
