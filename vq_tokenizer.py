import torch
import numpy as np

# 2x downscale, no pooling
class Residual_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual_Block, self).__init__()
        self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(inplace=True)
        )
        self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                torch.nn.BatchNorm2d(out_channels)
        )
        self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),     
        )
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.shortcut(residual)

        return self.activation(x + residual)

# assuming grayscale images
class Image_Encoder(torch.nn.Module):
    def __init__(self, num_channels=[64, 128, 256]):
        super(Image_Encoder, self).__init__()
        num_channels = 1 + num_channels
        self.conv = torch.nn.Sequential(Residual_Block(num_channels[i], num_channels[i + 1]) for i in range(len(num_channels) - 1))

    def forward(self, x):
        x = self.conv(x)
        return torch.mean(x, dim=[2, 3])

# code_dim should be the same as the number of dims in input vectors to this block
class Quantization_Block(torch.nn.Module):
    def __init__(self, codebook_size, code_dim, distance="L2"):
        super(Quantization_Block, self).__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.codebook = torch.nn.Parameter(torch.randn(codebook_size, code_dim))
        assert distance in ["L2", "cosine"], "unknown distance metric!"
        self.distance = distance

    @torch.no_grad()
    def get_nearest_code_idx(self, x):
        batch_size = x.shape[0]
        if self.distance == "L2":
            # norm of each individual x vec
            x_sq = (x ** 2).sum(axis=1).unsqueeze(1)
            e_sq = (self.codebook ** 2).sum(axis=1).unsqueeze(0)
            product = x @ self.codebook.T
            dist = -2 * product + x_sq + e_sq
        else:
            x_norm = x / (x ** 2).sum(axis=1)
            e_norm = self.codebook  / (self.codebook ** 2).sum(axis=1)
            dist = x_norm.unsqueeze(1) @ e_norm.unsqueeze(0)

        return torch.argmax(dist, dim=1)
        
    def forward(self, x):
        best_idx = self.get_nearest_code_idx(x)
        return self.codebook[best_idx]