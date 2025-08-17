# 2D-to-3D Image Generation via Late Fusion U-Net

## Overview

This project provides a PyTorch implementation of a deep learning model that generates a single 3D image volume from four separate 2D image inputs. The model is specifically designed to handle inputs of shape `(1, 256, 256)` and produce a 3D image of shape `(3, 256, 256, 256)`.

The architecture is tailored for memory efficiency and performance on modern GPUs like the NVIDIA RTX 6000 Ada series.

## Model Architecture

The core of the project is a U-Net-like convolutional neural network with a novel structure for fusing information from multiple 2D sources to create a 3D output.

1.  **Four 2D Encoders**: Each of the four 2D input images is processed by a separate, independent 2D encoder. These encoders are based on a standard U-Net downsampling path, consisting of repeated blocks of `Conv2d -> BatchNorm2d -> ReLU` followed by `MaxPool2d`. Skip connections are stored at each resolution level.
2.  **Late Fusion**: The features from the bottleneck (the most compressed layer) of all four encoders are concatenated. This combined feature map is then processed by a fusion convolution layer. This "late fusion" approach allows the model to learn specialized features for each input before combining them.
3.  **3D Decoder**: The fused feature map is expanded from 2D to 3D to form the starting point of the decoder. The decoder then progressively upsamples the volume using `ConvTranspose3d`.
4.  **3D Skip Connections**: At each upsampling stage in the 3D decoder, the corresponding 2D skip connections from all four encoders are fused and expanded into a 3D volume. This 3D skip connection is then concatenated with the decoder's feature map, mimicking the U-Net design and helping to preserve fine-grained details.

## File Structure

-   `models.py`: Contains the complete PyTorch implementation of the `LateFusionUNet`, including all its sub-modules (`Encoder2D`, `DecoderBlock3D`, etc.).
-   `loss.py`: Defines a custom `CombinedLoss` function, which is a weighted sum of L1 Loss and a 3D Structural Similarity Index (SSIM) loss. This is designed to improve perceptual quality in the generated images.
-   `train.py`: A comprehensive training script. It includes a dummy data generator for testing, command-line arguments for hyperparameter tuning, and a full training loop with support for modern GPU features.
-   `requirements.txt`: A list of necessary Python packages to run the project.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `train.py` script is the main entry point for training the model.

### Running a Test Training

You can start a test run using the default parameters (which are set for low-memory environments):
```bash
python train.py
```

### Customizing Training

The script provides several command-line arguments to control the training process:

-   `--epochs`: Number of training epochs (default: 5).
-   `--batch-size`: Batch size (default: 1).
-   `--learning-rate`: Learning rate for the AdamW optimizer (default: 1e-4).
-   `--img-size`: The spatial dimension of the input images and output volume (default: 64). For full-scale training, set this to `256`.
-   `--features`: A list of integers defining the number of channels in the encoder blocks (e.g., `--features 16 32 64 128`).
-   `--use-checkpointing`: A flag to enable gradient checkpointing, which trades computation for a significant reduction in memory usage. Recommended for large image sizes.
-   `--no-amp`: A flag to disable Automatic Mixed Precision (AMP). AMP is enabled by default on CUDA devices.
-   `--device`: The device to train on (default: `cuda`).

**Example for full-scale training on an RTX 6000 Ada:**
```bash
python train.py \
    --epochs 100 \
    --batch-size 1 \
    --img-size 256 \
    --features 16 32 64 128 \
    --learning-rate 0.0002 \
    --use-checkpointing
```

### Using Your Own Data

The script uses a `Dummy3DDataset` by default. To train on your own data, you need to create your own `torch.utils.data.Dataset` class and replace it in `train.py`. Your dataset's `__getitem__` method should return a tuple containing:
1.  A list of 4 tensors, each of shape `(1, H, W)`.
2.  A single target tensor of shape `(3, D, H, W)`.

## Key Features

-   **Memory Efficient**: Includes gradient checkpointing (`torch.utils.checkpoint`) to reduce VRAM usage during training.
-   **High Performance**: Supports Automatic Mixed Precision (`torch.cuda.amp`) for faster training on NVIDIA GPUs with Tensor Cores.
-   **Flexible**: Model architecture and training parameters can be easily configured via command-line arguments.
-   **Advanced Loss Function**: Uses a combination of L1 and 3D SSIM loss to optimize for both pixel accuracy and perceptual quality.
