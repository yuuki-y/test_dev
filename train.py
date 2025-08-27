import argparse
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from model import XrayFusionTransformer
from dataset import XrayDataset
from utils import save_nifti

logger = get_logger(__name__)

def main(args):
    # --- Accelerator and Logging Setup ---
    accelerator = Accelerator(log_with=args.log_with, project_dir=args.output_dir)
    logger.info(accelerator.state, main_process_only=False)
    set_seed(42)

    # --- Data Loading and Splitting ---
    # Get all filenames and shuffle for splitting
    all_files_dir = os.path.join(args.data_dir, 'frontal_1')
    all_filenames = sorted([f for f in os.listdir(all_files_dir) if f.endswith('.pt')])
    random.shuffle(all_filenames)

    # Split filenames into training and validation sets
    val_split = int(len(all_filenames) * args.validation_split)
    val_filenames = all_filenames[:val_split]
    train_filenames = all_filenames[val_split:]

    logger.info(f"Total samples: {len(all_filenames)}, Training samples: {len(train_filenames)}, Validation samples: {len(val_filenames)}")

    # Create datasets
    train_dataset = XrayDataset(root_dir=args.data_dir, filenames=train_filenames)
    val_dataset = XrayDataset(root_dir=args.data_dir, filenames=val_filenames)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Model, Optimizer, and Loss Function ---
    model = XrayFusionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.model_embed_dim,
        depth_individual=args.model_depth_individual,
        depth_fusion=args.model_depth_fusion,
        num_heads=args.model_num_heads,
        decoder_start_res=args.decoder_start_res
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss_fn = nn.L1Loss() # L1 Loss is often good for regression on physical quantities

    # --- Prepare for Distributed Training ---
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # --- Training Loop ---
    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss_total = 0.0
        for step, (inputs, labels, _) in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_dataloader)

        # --- Validation Phase ---
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for step, (inputs, labels, filenames) in enumerate(val_dataloader):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss_total += loss.item()

                # Save a few output examples from the first validation batch on the main process
                if step == 0 and accelerator.is_main_process:
                    for i in range(min(len(outputs), args.save_n_outputs)): # Save N outputs
                        output_tensor = outputs[i]
                        base_filename = filenames[i]
                        save_path = os.path.join(args.output_dir, "validation_outputs", f"epoch_{epoch+1}", f"pred_{base_filename.replace('.pt', '.nii.gz')}")
                        save_nifti(output_tensor, save_path)

        avg_val_loss = val_loss_total / len(val_dataloader)

        # --- Logging and Checkpointing ---
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            accelerator.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch)

            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                accelerator.save_state(os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}"))

    accelerator.end_training()
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D X-ray Deformation Transformer")

    # Paths
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the root data directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate.")
    parser.add_argument("--validation_split", type=float, default=0.15, help="Fraction of data to use for validation.")
    parser.add_argument("--log_with", type=str, default="tensorboard", help="Logger to use (tensorboard, wandb).")
    parser.add_argument("--save_every", type=int, default=10, help="Save a checkpoint every N epochs.")
    parser.add_argument("--save_n_outputs", type=int, default=4, help="Number of validation outputs to save as .nii.gz files.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for dataloader.")

    # Model Hyperparameters
    parser.add_argument("--image_size", type=int, default=256, help="Size of input images.")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the Vision Transformer.")
    parser.add_argument("--model_embed_dim", type=int, default=768, help="Embedding dimension for the transformer model.")
    parser.add_argument("--model_depth_individual", type=int, default=4, help="Number of layers in individual view encoders.")
    parser.add_argument("--model_depth_fusion", type=int, default=8, help="Number of layers in the fusion encoder.")
    parser.add_argument("--model_num_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--decoder_start_res", type=int, default=16, help="Starting resolution of the 3D decoder.")

    args = parser.parse_args()
    main(args)
