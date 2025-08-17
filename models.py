import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --- 2D Blocks ---

class ConvBlock2D(nn.Module):
    """2D Convolutional Block: Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Encoder2D(nn.Module):
    """2D U-Net Encoder"""
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoders = nn.ModuleList()

        for feature in features:
            self.encoders.append(ConvBlock2D(in_channels, feature))
            in_channels = feature

        self.bottleneck = ConvBlock2D(features[-1], features[-1] * 2)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        return x, skip_connections[::-1] # Reverse for decoder

# --- 3D Blocks ---

class DecoderBlock3D(nn.Module):
    """3D Decoder Block: Upsample -> Concat -> ConvBlock"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsampling layer
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Convolutional block
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        x = self.up(x)

        # Adjust skip connection shape if necessary
        if x.shape != skip_connection.shape:
            # Pad skip_connection to match the spatial dimensions of x
            diff_d = x.size(2) - skip_connection.size(2)
            diff_h = x.size(3) - skip_connection.size(3)
            diff_w = x.size(4) - skip_connection.size(4)
            skip_connection = F.pad(skip_connection, [diff_w // 2, diff_w - diff_w // 2,
                                                       diff_h // 2, diff_h - diff_h // 2,
                                                       diff_d // 2, diff_d - diff_d // 2])

        x = torch.cat((skip_connection, x), dim=1)
        return self.conv(x)

# --- Main Model ---

class LateFusionUNet(nn.Module):
    def __init__(self, num_2d_inputs=4, in_channels_2d=1, out_channels_3d=3,
                 encoder_features=[16, 32, 64, 128], use_checkpointing=True, img_size=256):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.img_size = img_size
        self.encoder_features = encoder_features

        # 1. Four separate 2D encoders
        self.encoders = nn.ModuleList(
            [Encoder2D(in_channels=in_channels_2d, features=encoder_features) for _ in range(num_2d_inputs)]
        )

        # 2. Fusion layer
        bottleneck_out_features = encoder_features[-1] * 2
        fusion_in_features = bottleneck_out_features * num_2d_inputs
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_features, fusion_in_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_in_features),
            nn.ReLU(inplace=True)
        )

        # 3. 3D Decoder
        self.decoders = nn.ModuleList()

        decoder_features = encoder_features[::-1]
        in_channels = fusion_in_features

        for i, skip_features in enumerate(decoder_features):
            skip_channels_fused = skip_features * num_2d_inputs
            self.decoders.append(
                DecoderBlock3D(in_channels, skip_channels_fused, skip_features)
            )
            in_channels = skip_features

        # Final convolution to get the desired number of output channels
        self.final_conv = nn.Conv3d(decoder_features[-1], out_channels_3d, kernel_size=1)

    def _encoder_forward(self, encoder, x):
        """Helper for checkpointing"""
        return encoder(x)

    def _decoder_block_forward(self, decoder_block, x, skip):
        """Helper for checkpointing"""
        return decoder_block(x, skip)

    def forward(self, inputs):
        # inputs: a list/tuple of 4 tensors of shape (N, C_2d, H, W)
        encoder_outputs = []
        all_skip_connections = []

        # 1. Pass each input through its own encoder
        for i, x in enumerate(inputs):
            if self.use_checkpointing:
                bottleneck, skips = checkpoint(self._encoder_forward, self.encoders[i], x, use_reentrant=False)
            else:
                bottleneck, skips = self.encoders[i](x)
            encoder_outputs.append(bottleneck)
            all_skip_connections.append(skips)

        # 2. Late Fusion of bottleneck features
        fused_bottleneck = torch.cat(encoder_outputs, dim=1)
        fused_bottleneck = self.fusion_conv(fused_bottleneck)

        # 3. Reshape fused bottleneck to start the 3D decoder path
        num_downsamples = len(self.encoder_features)
        initial_3d_depth = self.img_size // (2 ** num_downsamples)
        x = fused_bottleneck.unsqueeze(2).repeat(1, 1, initial_3d_depth, 1, 1)

        # 4. Fuse skip connections from all encoders at each level
        fused_skips_2d = []
        num_skip_levels = len(all_skip_connections[0])
        for level in range(num_skip_levels):
            skips_at_level = [skips[level] for skips in all_skip_connections]
            fused = torch.cat(skips_at_level, dim=1)
            fused_skips_2d.append(fused)

        # 5. Run 3D Decoder, expanding skip connections just-in-time
        for i, decoder_block in enumerate(self.decoders):
            # Get the 2D fused skip connection for the current level
            skip_2d = fused_skips_2d[i]

            # The upsampling in the decoder block will double the depth of x
            target_depth = x.shape[2] * 2

            # Expand the 2D skip connection to the target 3D depth
            skip_3d = skip_2d.unsqueeze(2).repeat(1, 1, target_depth, 1, 1)

            if self.use_checkpointing:
                x = checkpoint(self._decoder_block_forward, decoder_block, x, skip_3d, use_reentrant=False)
            else:
                x = decoder_block(x, skip_3d)

        # 6. Final 1x1x1 convolution
        return self.final_conv(x)

if __name__ == '__main__':
    # --- Verification ---
    # NOTE: The parameters here are reduced for low-memory environments.
    # The model is designed for larger inputs as defined in the class defaults.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters (reduced for testing)
    BATCH_SIZE = 1
    INPUT_CHANNELS = 1
    OUTPUT_CHANNELS = 3
    IMG_SIZE = 64  # Reduced from 256
    NUM_INPUTS = 4
    ENCODER_FEATURES = [8, 16, 32] # Reduced feature map sizes

    # Create dummy input
    inputs = [torch.randn(BATCH_SIZE, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE).to(device) for _ in range(NUM_INPUTS)]

    # Initialize model with reduced features
    model = LateFusionUNet(
        num_2d_inputs=NUM_INPUTS,
        in_channels_2d=INPUT_CHANNELS,
        out_channels_3d=OUTPUT_CHANNELS,
        encoder_features=ENCODER_FEATURES,
        use_checkpointing=False, # Checkpointing is not useful without autograd
        img_size=IMG_SIZE
    ).to(device)

    # Test forward pass
    print("Testing forward pass with reduced parameters...")
    with torch.no_grad():
        output = model(inputs)

    print(f"Input shapes: {[x.shape for x in inputs]}")
    print(f"Output shape: {output.shape}")

    # Calculate expected output shape based on test parameters
    # Encoder downsamples 3 times (64 -> 32 -> 16 -> 8). Bottleneck is 8x8.
    # Initial depth is also 8.
    # Decoder upsamples 3 times (8 -> 16 -> 32 -> 64)
    expected_shape = (BATCH_SIZE, OUTPUT_CHANNELS, IMG_SIZE, IMG_SIZE, IMG_SIZE)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"

    print("\nModel Summary (with reduced test parameters):")
    # print(model) # Printing the whole model is too verbose

    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {num_params / 1e6:.2f} M")
    print("\nModel verification successful!")
