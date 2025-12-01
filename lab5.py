# pixelcnn_improved_fixed.py
"""
Исправленная и улучшенная версия PixelCNN для MNIST.
Основные правки:
- маска применяется только в forward (не мутируя self.weight.data)
- единообразное масштабирование входа [0,1] и таргетов 0..255
- инициализация весов, фиксирование random seed
- немного увеличенная архитектура, grad clipping, scheduler
- стабильная функция sample и show
"""

import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------------- reproducibility ----------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(42)

# ---------------- Masked Conv --------------------
class MaskedConv2d(nn.Conv2d):
    """
    Масочная свёртка: маска применяется в forward через weight * mask (не меняя weight.data)
    mask_type: 'A' (первая свёртка) или 'B' (последующие)
    """
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        assert mask_type in ("A", "B")
        # маска той же формы что и вес (out_channels, in_channels, kH, kW)
        self.register_buffer("mask", torch.ones_like(self.weight))

        kH, kW = self.kernel_size
        yc, xc = kH // 2, kW // 2

        # обнулить все будущие пиксели (ниже по y) и справа в той же строке
        self.mask[:, :, yc+1:, :] = 0
        # если mask_type == 'A', исключаем центр; если 'B' — центр разрешаем
        center_cut = 1 if mask_type == "A" else 0
        self.mask[:, :, yc, xc + center_cut: ] = 0

    def forward(self, x):
        # не меняем self.weight.data — маска применяется только на вычисляемом тензоре
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, self.stride, self.padding)

# ---------------- Residual Block --------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv2d(channels, channels, 1))
        self.conv2 = weight_norm(nn.Conv2d(channels, channels, 1))
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        return x + out

# ---------------- PixelCNN Model --------------------
class PixelCNN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=128, num_layers=12, num_classes=256):
        super().__init__()
        self.layers = nn.ModuleList()

        # первая масочная свёртка типа A
        self.layers.append(
            weight_norm(MaskedConv2d('A', input_channels, hidden_channels, kernel_size=7, padding=3))
        )
        self.layers.append(nn.ReLU(True))

        # блоки mask-B + residuals
        for _ in range(num_layers):
            self.layers.append(
                weight_norm(MaskedConv2d('B', hidden_channels, hidden_channels, kernel_size=3, padding=1))
            )
            self.layers.append(nn.ReLU(True))
            self.layers.append(ResidualBlock(hidden_channels))

        # выходная часть: 1x1 сверточные
        self.output = nn.Sequential(
            weight_norm(nn.Conv2d(hidden_channels, hidden_channels, 1)),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, num_classes, 1)
        )

        self._initialize_weights()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return self.output(out)

    def _initialize_weights(self):
        # init для стабильности
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# ---------------- Data --------------------
def get_loaders(batch=64, data_dir="./data"):
    transform = transforms.ToTensor()  # даёт тензор float32 в [0,1]
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

# ---------------- Training utils --------------------
def preprocess(batch):
    """
    Возвращает:
     - inputs: float32 [0,1] (B,1,28,28)
     - targets: long 0..255 (B, H, W)
    """
    x, _ = batch
    # targets: 0..255 ints
    t = (x * 255.0).long().squeeze(1)  # (B, H, W)
    return x, t

def train_epoch(model, loader, opt, device, criterion, clip_grad=1.0, log_interval=200):
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        x, targets = preprocess(batch)
        x = x.to(device)
        targets = targets.to(device)

        opt.zero_grad()
        logits = model(x)  # (B, C=256, H, W)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()

        total_loss += loss.item()
        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx+1}/{len(loader)}  avg_loss={avg:.4f}")

    return total_loss / len(loader)

def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x, targets = preprocess(batch)
            x = x.to(device)
            targets = targets.to(device)
            logits = model(x)
            loss = criterion(logits, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def sample(model, device, n=16, num_classes=256):
    """
    Генерация: samples хранит входы в том же виде, в каком модель ожидала (float [0,1]).
    Для каждого пикселя берём категорию и записываем pixel/255.0 -> float in [0,1].
    """
    model.eval()
    samples = torch.zeros((n, 1, 28, 28), device=device, dtype=torch.float32)

    for i in range(28):
        for j in range(28):
            logits = model(samples)  # (n, num_classes, H, W)
            pixel_logits = logits[:, :, i, j]  # (n, num_classes)
            probs = F.softmax(pixel_logits, dim=-1)
            # sample categorical (multinomial)
            pixel_sample = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (n,)
            samples[:, 0, i, j] = pixel_sample.float() / (num_classes - 1)  # scale to [0,1]

    return samples.cpu()

def show(imgs, nrow=4, figsize=(6,6), cmap="gray"):
    """
    imgs: torch.Tensor CPU (N,1,H,W) float [0,1]
    """
    imgs = imgs.detach().cpu().numpy()
    N = imgs.shape[0]
    ncol = nrow
    nrow = math.ceil(N / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    axs = np.array(axs).reshape(-1)
    for idx in range(len(axs)):
        ax = axs[idx]
        ax.axis("off")
        if idx < N:
            ax.imshow(imgs[idx,0], cmap=cmap, vmin=0, vmax=1)
    plt.tight_layout()
    plt.show()

# ---------------- MAIN --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    batch_size = 128
    epochs = 16
    lr = 2e-3

    train_loader, test_loader = get_loaders(batch=batch_size)

    model = PixelCNN(input_channels=1, hidden_channels=128, num_layers=12, num_classes=256).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()  # expects (B, C, H, W) logits and (B, H, W) targets

    os.makedirs("outputs_fixed", exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, clip_grad=1.0, log_interval=200)
        val_loss = eval_epoch(model, test_loader, device, criterion)
        scheduler.step()
        print(f" -> Epoch {epoch} done. TrainLoss {train_loss:.4f}  ValLoss {val_loss:.4f}")

        # сохраняем чекпоинт
        torch.save(model.state_dict(), f"outputs_fixed/pixelcnn_epoch_{epoch}.pt")

        # сэмплируем и показываем 16 изображений
        if epoch % 1 == 0:
            samples = sample(model, device, n=16, num_classes=256)
            show(samples, nrow=4)
            # сохранить картинку
            import torchvision.utils as vutils
            grid = vutils.make_grid(samples, nrow=4, normalize=False)
            plt.imsave(f"outputs_fixed/samples_epoch_{epoch}.png", grid.permute(1,2,0).numpy().squeeze(), cmap="gray")

if __name__ == "__main__":
    main()
