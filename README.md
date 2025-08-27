# 3D X-ray Deformation Transformer

This project implements a Transformer-based model to predict 3D deformation fields from a series of 2D X-ray images. It takes four 2D images as input (2 frontal, 2 lateral) and outputs a 3D volume representing the deformation (dx, dy, dz).

The model is designed for distributed training using PyTorch and Hugging Face Accelerate, with support for DeepSpeed and Fully Sharded Data Parallelism (FSDP) for memory optimization on multi-GPU setups.

## Project Structure

- `model.py`: Contains the PyTorch implementation of the `XrayFusionTransformer` model.
- `dataset.py`: Defines the custom `Dataset` for loading `.pt` images and `.nii.gz` labels.
- `train.py`: The main training script using `Accelerate` for distributed training.
- `utils.py`: Utility functions, such as saving predictions to `.nii.gz` format.
- `requirements.txt`: A list of Python dependencies.
- `config/`: Contains sample configuration files for `Accelerate` and `DeepSpeed`.
  - `default_config.yaml`: Sample `Accelerate` config for a 2-node, 6-GPU FSDP setup.
  - `ds_config.json`: Sample `DeepSpeed` config using ZeRO Stage 2.
- `data/`: (You need to create this directory) Root directory for your datasets.

## Setup

### 1. Create Data Directories

You need to organize your data as follows. The training script expects specific directory names.

```
/path/to/your/data/
├── frontal_1/      # Contains first frontal images (*.pt)
├── frontal_2/      # Contains second frontal images (*.pt)
├── lateral_1/      # Contains first lateral images (*.pt)
├── lateral_2/      # Contains second lateral images (*.pt)
└── labels/         # Contains 3D label volumes (*.nii.gz)
```

**Note:** The filenames in each directory must correspond to each other. For example, `frontal_1/scan_001.pt` should correspond to `labels/scan_001.nii.gz`.

### 2. Install Dependencies

It is recommended to use a virtual environment (e.g., venv or conda).

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install all required packages
pip install -r requirements.txt
```

### 3. Configure Distributed Environment

This project uses `Hugging Face Accelerate` to manage distributed training.

**First, run the configuration wizard:**
```bash
accelerate config
```

The wizard will ask you a series of questions about your setup (e.g., number of machines, number of GPUs, if you want to use DeepSpeed or FSDP). Answer them according to your hardware. A sample configuration for a 2-node, 6-GPU FSDP setup is provided in `config/default_config.yaml`. You will need to **edit the `main_process_ip`** in the generated file.

If you choose to use DeepSpeed, you can use the `config/ds_config.json` as a starting point by passing it during the `accelerate config` process.

## Training

To start training, use the `accelerate launch` command. You need to provide the path to your data directory.

```bash
accelerate launch train.py \
    --data_dir /path/to/your/data/ \
    --output_dir /path/to/save/checkpoints_and_logs/ \
    --epochs 50 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --image_size 256 \
    --patch_size 16
```

### Command Line Arguments for `train.py`

- `--data_dir`: (Required) Path to the root data directory.
- `--output_dir`: (Required) Directory to save model checkpoints and validation outputs.
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size per GPU.
- `--learning_rate`: Peak learning rate for the optimizer.
- `--image_size`: Size of the input images (e.g., 256).
- `--patch_size`: Patch size for the Vision Transformer.
- `--model_embed_dim`: Embedding dimension for the transformer model.
- `--model_depth`: Number of layers in the main transformer encoder.
- `--model_num_heads`: Number of attention heads in the transformer.
- `--log_with`: Logger to use (e.g., 'tensorboard', 'wandb'). Defaults to 'tensorboard'.
- `--save_every`: Save a checkpoint every N epochs.

## Inference

To run inference on new data, you can adapt the validation loop in `train.py`. You would load a trained checkpoint and process your data through the model's `eval()` mode. The script already saves validation outputs as `.nii.gz` files in the `output_dir`.
