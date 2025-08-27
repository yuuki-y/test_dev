import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class XrayDataset(Dataset):
    """
    Custom dataset for loading X-ray data.
    It loads 4 input images from .pt files and 1 label volume from a .nii.gz file.

    Directory structure is expected to be:
    - root_dir/
        - frontal_1/
        - frontal_2/
        - lateral_1/
        - lateral_2/
        - labels/
    """
    def __init__(self, root_dir, filenames=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.frontal1_dir = os.path.join(root_dir, 'frontal_1')
        self.frontal2_dir = os.path.join(root_dir, 'frontal_2')
        self.lateral1_dir = os.path.join(root_dir, 'lateral_1')
        self.lateral2_dir = os.path.join(root_dir, 'lateral_2')
        self.labels_dir = os.path.join(root_dir, 'labels')

        # If a list of filenames is provided, use it. Otherwise, scan the directory.
        if filenames:
            self.filenames = filenames
        else:
            self.filenames = sorted([f for f in os.listdir(self.frontal1_dir) if f.endswith('.pt')])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        base_filename = self.filenames[idx]

        # --- Load Input Images ---
        try:
            f1_path = os.path.join(self.frontal1_dir, base_filename)
            f2_path = os.path.join(self.frontal2_dir, base_filename)
            l1_path = os.path.join(self.lateral1_dir, base_filename)
            l2_path = os.path.join(self.lateral2_dir, base_filename)

            # Load tensors from .pt files
            img_f1 = torch.load(f1_path)
            img_f2 = torch.load(f2_path)
            img_l1 = torch.load(l1_path)
            img_l2 = torch.load(l2_path)
        except FileNotFoundError as e:
            print(f"Error loading image file for base name {base_filename}: {e}")
            raise e


        # --- Load Label ---
        label_filename = base_filename.replace('.pt', '.nii.gz')
        label_path = os.path.join(self.labels_dir, label_filename)
        try:
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata(dtype=np.float32)
            label_tensor = torch.from_numpy(label_data).float()
        except FileNotFoundError as e:
            print(f"Error loading label file {label_path}: {e}")
            raise e

        # --- Standardize Shapes ---
        images = [img_f1, img_f2, img_l1, img_l2]
        processed_images = []
        for img in images:
            if img.ndim == 2: # (H, W) -> (1, H, W)
                img = img.unsqueeze(0)
            processed_images.append(img)

        input_tensor = torch.stack(processed_images, dim=0)

        # Label: Nibabel loads as (W, H, D, C) or (W, H, D). We want (C, D, H, W).
        if label_tensor.ndim == 4 and label_tensor.shape[-1] == 3: # (W, H, D, 3)
            label_tensor = label_tensor.permute(3, 2, 1, 0) # (3, D, H, W)
        elif label_tensor.ndim == 3: # (W, H, D)
             # This case is ambiguous. Assuming it needs a channel dim and permutation.
             # This might need adjustment based on actual data format.
             label_tensor = label_tensor.permute(2, 1, 0).unsqueeze(0)

        if label_tensor.shape[0] != 3:
            raise ValueError(f"Label for {label_filename} has shape {label_tensor.shape}. Expected 3 channels first.")

        sample = {'image': input_tensor, 'label': label_tensor, 'filename': base_filename}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['label'], sample['filename']

if __name__ == '__main__':
    print("Running a dummy test for XrayDataset...")
    dummy_dir = './dummy_data_dataset_test'
    os.makedirs(os.path.join(dummy_dir, "frontal_1"), exist_ok=True)
    os.makedirs(os.path.join(dummy_dir, "frontal_2"), exist_ok=True)
    os.makedirs(os.path.join(dummy_dir, "lateral_1"), exist_ok=True)
    os.makedirs(os.path.join(dummy_dir, "lateral_2"), exist_ok=True)
    os.makedirs(os.path.join(dummy_dir, "labels"), exist_ok=True)

    try:
        dummy_img = torch.rand(1, 256, 256)
        torch.save(dummy_img, os.path.join(dummy_dir, "frontal_1/sample1.pt"))
        torch.save(dummy_img, os.path.join(dummy_dir, "frontal_2/sample1.pt"))
        torch.save(dummy_img, os.path.join(dummy_dir, "lateral_1/sample1.pt"))
        torch.save(dummy_img, os.path.join(dummy_dir, "lateral_2/sample1.pt"))

        dummy_label_np = np.random.rand(256, 256, 256, 3).astype(np.float32)
        affine = np.eye(4)
        nifti_img = nib.Nifti1Image(dummy_label_np, affine)
        nib.save(nifti_img, os.path.join(dummy_dir, "labels/sample1.nii.gz"))

        print("Dummy data created.")

        dataset = XrayDataset(root_dir=dummy_dir)
        print(f"Dataset size: {len(dataset)}")
        image, label, filename = dataset[0]
        print(f"Sample image shape: {image.shape}")
        print(f"Sample label shape: {label.shape}")
        print(f"Sample filename: {filename}")

    finally:
        import shutil
        if os.path.exists(dummy_dir):
            shutil.rmtree(dummy_dir)
        print("Dummy data cleaned up.")
        print("Dataset test successful!")
