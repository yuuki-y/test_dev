import torch
import torch.nn as nn
import math

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (self.img_size[1] // self.patch_size[1]) * (self.img_size[0] // self.patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class Transformer3DDecoder(nn.Module):
    """Decoder to reconstruct 3D image from sequence"""
    def __init__(self, embed_dim=768, decoder_start_res=16):
        super().__init__()
        self.decoder_start_res = decoder_start_res
        self.embed_dim = embed_dim

        # Project the transformer output to a feature map for the 3D decoder
        self.proj = nn.Linear(embed_dim, (decoder_start_res**3) * embed_dim // 8)

        # Decoder blocks using Transposed Convolution
        self.decoder = nn.Sequential(
            # Start from a low resolution, e.g., 16x16x16
            nn.ConvTranspose3d(embed_dim // 8, 256, kernel_size=4, stride=2, padding=1), # -> 32x32x32
            nn.BatchNorm3d(256),
            nn.ReLU(True),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1), # -> 64x64x64
            nn.BatchNorm3d(128),
            nn.ReLU(True),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1), # -> 128x128x128
            nn.BatchNorm3d(64),
            nn.ReLU(True),

            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1), # -> 256x256x256
            nn.BatchNorm3d(32),
            nn.ReLU(True),

            # Final layer to get the 3 output channels (dx, dy, dz)
            # MODIFIED: Output channels changed from 1 to 3
            nn.Conv3d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh() # Use Tanh if deformation is normalized between -1 and 1
        )

    def forward(self, x):
        # x shape: (B, num_patches_total, embed_dim)
        # We take the mean of all patch tokens as the global representation
        x = x.mean(dim=1) # (B, embed_dim)

        # Project and reshape for 3D decoder
        x = self.proj(x) # (B, (start_res^3) * C)
        x = x.view(x.size(0), self.embed_dim // 8, self.decoder_start_res, self.decoder_start_res, self.decoder_start_res)

        # Apply decoder
        x = self.decoder(x) # (B, 3, 256, 256, 256)
        return x

class XrayFusionTransformer(nn.Module):
    """
    Transformer model that processes frontal and lateral X-ray pairs,
    fuses them, and decodes them into a 3D deformation map.
    """
    def __init__(self,
                 img_size=256,
                 patch_size=16,
                 in_chans=1,
                 embed_dim=768,
                 depth_individual=4,
                 depth_fusion=8,
                 num_heads=12,
                 mlp_ratio=4.,
                 decoder_start_res=16):
        super().__init__()

        # --- Shared Patch Embedding ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # --- Positional Embeddings ---
        # Positional embedding for individual view encoders (2 images concatenated)
        self.pos_embed_individual = nn.Parameter(torch.zeros(1, num_patches * 2, embed_dim))
        # Positional embedding for the final fusion encoder (4 images concatenated)
        self.pos_embed_fusion = nn.Parameter(torch.zeros(1, num_patches * 4, embed_dim))

        # --- Individual View Encoders ---
        encoder_layer_individual = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(mlp_ratio * embed_dim), batch_first=True
        )
        self.frontal_encoder = nn.TransformerEncoder(encoder_layer_individual, num_layers=depth_individual)
        self.lateral_encoder = nn.TransformerEncoder(encoder_layer_individual, num_layers=depth_individual)

        # --- Fusion Encoder ---
        encoder_layer_fusion = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(mlp_ratio * embed_dim), batch_first=True
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer_fusion, num_layers=depth_fusion)

        # --- 3D Decoder ---
        self.decoder = Transformer3DDecoder(embed_dim, decoder_start_res)

        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed_individual, std=.02)
        nn.init.trunc_normal_(self.pos_embed_fusion, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x shape: (B, 4, C, H, W). Order: F1, F2, L1, L2
        B = x.shape[0]

        # Unpack the 4 images
        x_f1, x_f2, x_l1, x_l2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        # --- Patch Embedding ---
        p_f1 = self.patch_embed(x_f1)
        p_f2 = self.patch_embed(x_f2)
        p_l1 = self.patch_embed(x_l1)
        p_l2 = self.patch_embed(x_l2)

        # --- Frontal Path ---
        x_frontal = torch.cat([p_f1, p_f2], dim=1)  # (B, 2 * num_patches, embed_dim)
        x_frontal = x_frontal + self.pos_embed_individual
        encoded_frontal = self.frontal_encoder(x_frontal)

        # --- Lateral Path ---
        x_lateral = torch.cat([p_l1, p_l2], dim=1)  # (B, 2 * num_patches, embed_dim)
        x_lateral = x_lateral + self.pos_embed_individual # Share positional embedding
        encoded_lateral = self.lateral_encoder(x_lateral)

        # --- Fusion Path ---
        x_fusion = torch.cat([encoded_frontal, encoded_lateral], dim=1) # (B, 4 * num_patches, embed_dim)
        x_fusion = x_fusion + self.pos_embed_fusion
        encoded_fusion = self.fusion_encoder(x_fusion)

        # --- Decoder ---
        output_3d = self.decoder(encoded_fusion) # (B, 3, 256, 256, 256)

        return output_3d

if __name__ == '__main__':
    # --- Test the full model ---
    print("Testing XrayFusionTransformer model...")

    # Use smaller, more memory-friendly parameters for a quick test
    # These parameters are chosen to be small for a CPU test.
    # For GPU training, these should be increased significantly.
    model = XrayFusionTransformer(
        img_size=256,
        patch_size=32,          # Larger patch size -> smaller sequence length
        embed_dim=256,          # Smaller embedding dim
        depth_individual=1,     # Shallow individual encoders
        depth_fusion=2,         # Shallow fusion encoder
        num_heads=4,            # Fewer heads
        mlp_ratio=2.,
        decoder_start_res=16
    ).to("cpu") # Test on CPU to avoid CUDA errors if no GPU

    # Create a dummy input tensor
    # Batch size of 1, 4 images, 1 channel, 256x256
    # Order: Frontal1, Frontal2, Lateral1, Lateral2
    dummy_input = torch.rand(1, 4, 1, 256, 256)
    print(f"Input shape: {dummy_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}") # Expected: (1, 3, 256, 256, 256)

    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params / 1e6:.2f}M")

    print("\nModel test successful!")
