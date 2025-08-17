import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

from models import LateFusionUNet
from loss import CombinedLoss

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device, use_amp):
    """
    Performs one full training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, target) in enumerate(loader):
        inputs = [i.to(device) for i in inputs]
        target = target.to(device)

        # Forward pass with Automatic Mixed Precision
        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model(inputs)
            loss = loss_fn(predictions, target)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    return avg_loss

class Dummy3DDataset(Dataset):
    """
    A dummy dataset that generates random 2D inputs and a 3D target.
    This is for demonstration and testing purposes.
    Replace this with your actual dataset loader.
    """
    def __init__(self, num_samples=100, img_size=256, num_2d_inputs=4):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_2d_inputs = num_2d_inputs
        print(f"Initializing dummy dataset with {num_samples} samples of size {img_size}.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 4 separate 2D images (1 channel)
        inputs = [torch.randn(1, self.img_size, self.img_size) for _ in range(self.num_2d_inputs)]
        # 1 3D image (3 channels)
        target = torch.randn(3, self.img_size, self.img_size, self.img_size)
        return inputs, target

def main():
    parser = argparse.ArgumentParser(description="Train a 2D-to-3D UNet model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--img-size", type=int, default=64, help="Size of input images (and output volume). Using a smaller size for testing.")
    parser.add_argument("--features", type=int, nargs='+', default=[16, 32, 64], help="List of feature channels for encoder.")
    parser.add_argument('--use-checkpointing', action='store_true', help='Use gradient checkpointing to save memory.')
    parser.add_argument('--no-amp', action='store_true', help='Disable Automatic Mixed Precision.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda", "cpu").')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == 'cuda'

    print("--- Training Configuration ---")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.img_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Encoder Features: {args.features}")
    print(f"Gradient Checkpointing: {args.use_checkpointing}")
    print(f"Automatic Mixed Precision: {use_amp}")
    print("----------------------------\n")

    # --- Data ---
    # NOTE: Replace Dummy3DDataset with your actual dataset
    train_dataset = Dummy3DDataset(num_samples=50, img_size=args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # --- Model ---
    model = LateFusionUNet(
        num_2d_inputs=4,
        in_channels_2d=1,
        out_channels_3d=3,
        encoder_features=args.features,
        use_checkpointing=args.use_checkpointing,
        img_size=args.img_size
    ).to(device)

    # --- Loss & Optimizer ---
    loss_fn = CombinedLoss(alpha=0.85)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- Training Loop ---
    print("Starting training...\n")
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        avg_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, device, use_amp)
        print(f"Epoch {epoch+1} complete. Average Loss: {avg_loss:.4f}\n")

    print("--- Training Finished ---")
    print("NOTE: This was a test run with dummy data.")
    print("To train on your data, replace the Dummy3DDataset with your own torch.utils.data.Dataset.")

if __name__ == '__main__':
    main()
