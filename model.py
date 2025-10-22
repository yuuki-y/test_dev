import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

# --- Helper Modules ---

class PositionalEncoding(nn.Module):
    """3Dスライスのz位置に対するPositional Encodingを生成"""
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, z_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_indices (torch.Tensor): スライスのインデックスのバッチ (B,)
        """
        return self.pe[z_indices]

class TimestepEmbedding(nn.Module):
    """拡散ステップtに対するEmbeddingを生成"""
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=self.half, dtype=torch.float32) / self.half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

class ConvBlock(nn.Module):
    """基本的なConvolutional Block (Conv -> GroupNorm -> SiLU)"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None, cond_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None
        self.cond_mlp = nn.Linear(cond_dim, out_channels) if cond_dim is not None else None

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x, time_emb=None, cond=None):
        h = self.act1(self.norm1(self.conv1(x)))

        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb.unsqueeze(-1).unsqueeze(-1)

        if self.cond_mlp is not None and cond is not None:
            cond_emb = self.cond_mlp(cond)
            h = h + cond_emb.unsqueeze(-1).unsqueeze(-1)

        h = self.act2(self.norm2(self.conv2(h)))
        return h

# --- Model Parallel U-Net Components ---

class UNetPart1(nn.Module):
    """U-Net Encoder Part on cuda:0"""
    def __init__(self, in_channels=1, base_dim=64, time_emb_dim=256, cond_dim=256):
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)

        self.down1 = ConvBlock(base_dim, base_dim, time_emb_dim, cond_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ConvBlock(base_dim, base_dim * 2, time_emb_dim, cond_dim)
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x, t_emb, c_emb):
        x = self.init_conv(x)

        s1 = checkpoint(self.down1, x, t_emb, c_emb, use_reentrant=False)
        x = self.pool1(s1)

        s2 = checkpoint(self.down2, x, t_emb, c_emb, use_reentrant=False)
        x = self.pool2(s2)

        return x, s1, s2

class UNetPart2(nn.Module):
    """U-Net Bottleneck Part on cuda:1"""
    def __init__(self, base_dim=64, time_emb_dim=256, cond_dim=256):
        super().__init__()
        self.down3 = ConvBlock(base_dim * 2, base_dim * 4, time_emb_dim, cond_dim)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_dim * 4, base_dim * 8, time_emb_dim, cond_dim)

        self.up1_conv = nn.ConvTranspose2d(base_dim * 8, base_dim * 4, kernel_size=2, stride=2)
        self.up1 = ConvBlock(base_dim * 8, base_dim * 4, time_emb_dim, cond_dim)

    def forward(self, x, t_emb, c_emb):
        s3 = checkpoint(self.down3, x, t_emb, c_emb, use_reentrant=False)
        x = self.pool3(s3)

        x = checkpoint(self.bottleneck, x, t_emb, c_emb, use_reentrant=False)

        x = self.up1_conv(x)
        x = torch.cat([x, s3], dim=1)
        x = checkpoint(self.up1, x, t_emb, c_emb, use_reentrant=False)
        return x

class UNetPart3(nn.Module):
    """U-Net Decoder Part on cuda:2"""
    def __init__(self, out_channels=1, base_dim=64, time_emb_dim=256, cond_dim=256):
        super().__init__()
        self.up2_conv = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=2, stride=2)
        self.up2 = ConvBlock(base_dim * 4, base_dim * 2, time_emb_dim, cond_dim)

        self.up3_conv = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.up3 = ConvBlock(base_dim * 2, base_dim, time_emb_dim, cond_dim)

        self.out_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, x, s1, s2, t_emb, c_emb):
        x = self.up2_conv(x)
        x = torch.cat([x, s2], dim=1)
        x = checkpoint(self.up2, x, t_emb, c_emb, use_reentrant=False)

        x = self.up3_conv(x)
        x = torch.cat([x, s1], dim=1)
        x = checkpoint(self.up3, x, t_emb, c_emb, use_reentrant=False)

        return self.out_conv(x)

# --- Main DX2CT Model ---

class DX2CT(nn.Module):
    def __init__(self, devices=['cuda:0', 'cuda:1', 'cuda:2'], max_depth=256, base_dim=64):
        super().__init__()
        self.devices = devices

        # --- Conditioning Modules (on device 0) ---
        self.xray_encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128)
        )
        self.pos_encoder = PositionalEncoding(d_model=128, max_len=max_depth)
        transformer_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.condition_transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.cond_out_proj = nn.Linear(256, 256)

        # --- Diffusion Time Embedding ---
        self.time_embedding = TimestepEmbedding(dim=128)
        self.time_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

        # --- Model Parallel U-Net ---
        self.unet_part1 = UNetPart1(in_channels=1, base_dim=base_dim, time_emb_dim=256, cond_dim=256)
        self.unet_part2 = UNetPart2(base_dim=base_dim, time_emb_dim=256, cond_dim=256)
        self.unet_part3 = UNetPart3(out_channels=1, base_dim=base_dim, time_emb_dim=256, cond_dim=256)

        self.to(devices[0])
        self.unet_part2.to(devices[1])
        self.unet_part3.to(devices[2])

    def forward(self, noisy_slice, time, z_index, xrays):
        d0, d1, d2 = self.devices

        xray_feat = self.xray_encoder(xrays)
        pos_feat = self.pos_encoder(z_index)
        combined_feat = torch.cat([xray_feat, pos_feat], dim=1)
        transformer_input = combined_feat.unsqueeze(1)
        cond_vec = self.condition_transformer(transformer_input).squeeze(1)
        cond_vec = self.cond_out_proj(cond_vec)

        time_emb_base = self.time_embedding(time)
        time_emb = self.time_mlp(time_emb_base)

        x, s1, s2 = self.unet_part1(noisy_slice, time_emb, cond_vec)

        x = x.to(d1)
        time_emb_d1 = time_emb.to(d1)
        cond_vec_d1 = cond_vec.to(d1)

        x = self.unet_part2(x, time_emb_d1, cond_vec_d1)

        x = x.to(d2)
        s1 = s1.to(d2)
        s2 = s2.to(d2)
        time_emb_d2 = time_emb.to(d2)
        cond_vec_d2 = cond_vec.to(d2)

        predicted_noise = self.unet_part3(x, s1, s2, time_emb_d2, cond_vec_d2)

        return predicted_noise

# --- Test code ---
if __name__ == '__main__':
    if torch.cuda.device_count() < 3:
        devices = ['cpu', 'cpu', 'cpu']
    else:
        devices = ['cuda:0', 'cuda:1', 'cuda:2']

    model = DX2CT(devices=devices, base_dim=16) # Test with small base_dim

    noisy_slice = torch.randn(2, 1, 64, 64).to(devices[0])
    time = torch.randint(0, 1000, (2,)).to(devices[0])
    z_index = torch.randint(0, 64, (2,)).to(devices[0])
    xrays = torch.randn(2, 2, 64, 64).to(devices[0])

    predicted_noise = model(noisy_slice, time, z_index, xrays)

    assert predicted_noise.shape == noisy_slice.shape
    assert str(predicted_noise.device) == devices[-1]

    target = torch.randn_like(predicted_noise).to(devices[-1])
    loss = F.mse_loss(predicted_noise, target)
    loss.backward()
    print("✅ Model test successful!")
