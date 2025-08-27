import torch
import nibabel as nib
import numpy as np
import os

def save_nifti(tensor, filename, affine=None):
    """
    Saves a PyTorch tensor as a NIFTI file.

    Args:
        tensor (torch.Tensor): The tensor to save. Assumed to be in (C, D, H, W) or (D, H, W) format.
                               It will be detached from the graph and moved to the CPU.
        filename (str): The path to save the NIFTI file to.
        affine (np.ndarray, optional): The affine transformation matrix for the NIFTI header.
                                       If None, an identity matrix is used. Defaults to None.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Detach tensor from graph, move to CPU, and convert to NumPy array
    if tensor.is_cuda:
        tensor = tensor.cpu()
    numpy_array = tensor.detach().numpy()

    # If the tensor has a batch dimension of 1, remove it
    if numpy_array.shape[0] == 1:
        numpy_array = numpy_array.squeeze(0)

    # PyTorch convention is (C, D, H, W). Nibabel expects (W, H, D, C) or (W, H, D).
    # Let's permute from (C, D, H, W) to (W, H, D, C)
    if numpy_array.ndim == 4:
        numpy_array = np.transpose(numpy_array, (3, 2, 1, 0))
    # Or from (D, H, W) to (W, H, D)
    elif numpy_array.ndim == 3:
        numpy_array = np.transpose(numpy_array, (2, 1, 0))


    # If no affine is provided, create a default one (identity matrix)
    if affine is None:
        affine = np.eye(4)

    # Create a NIFTI image object
    try:
        nifti_img = nib.Nifti1Image(numpy_array, affine)
        # Save the NIFTI image
        nib.save(nifti_img, filename)
    except Exception as e:
        print(f"Error saving nifti file: {e}")
        print(f"Array shape: {numpy_array.shape}, dtype: {numpy_array.dtype}")

    # print(f"Saved NIFTI file to {filename}")
