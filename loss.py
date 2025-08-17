import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIM3D(nn.Module):
    """
    3D Structural Similarity Index Measure (SSIM)
    """
    def __init__(self, window_size=7, k1=0.01, k2=0.03, sigma=1.5):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.k1 = k1
        self.k2 = k2
        self.sigma = sigma

        # Create a 1D Gaussian kernel
        gauss = torch.arange(-(window_size // 2), window_size // 2 + 1, dtype=torch.float)
        gauss = torch.exp(-gauss.pow(2.0) / (2 * sigma ** 2))

        # Create 3D Gaussian window
        _1D_window = (gauss / gauss.sum()).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', _3D_window)

    def forward(self, img1, img2):
        C = img1.size(1)
        window = self.window.expand(C, 1, self.window_size, self.window_size, self.window_size).contiguous()

        # --- Calculate mu ---
        mu1 = F.conv3d(img1, window, padding=self.window_size // 2, groups=C)
        mu2 = F.conv3d(img2, window, padding=self.window_size // 2, groups=C)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # --- Calculate sigma ---
        sigma1_sq = F.conv3d(img1 * img1, window, padding=self.window_size // 2, groups=C) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=self.window_size // 2, groups=C) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=self.window_size // 2, groups=C) - mu1_mu2

        # --- Calculate SSIM ---
        # Get dynamic range of the images
        L = img1.max() - img1.min() # Assumes images are scaled similarly
        c1 = (self.k1 * L) ** 2
        c2 = (self.k2 * L) ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return ssim_map.mean()

class CombinedLoss(nn.Module):
    """
    Combines L1 loss and 1-SSIM loss
    """
    def __init__(self, alpha=0.85):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM3D()
        print(f"CombinedLoss initialized with alpha = {self.alpha} (L1) and 1-alpha = {1-self.alpha} (SSIM)")

    def forward(self, y_pred, y_true):
        l1 = self.l1_loss(y_pred, y_true)
        ssim = self.ssim_loss(y_pred, y_true)
        # SSIM is a similarity metric, so for loss we use 1 - ssim
        ssim_loss = 1.0 - ssim

        combined_loss = self.alpha * l1 + (1 - self.alpha) * ssim_loss
        return combined_loss

if __name__ == '__main__':
    # --- Verification ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy 3D data
    BATCH_SIZE = 1
    CHANNELS = 3
    DEPTH, HEIGHT, WIDTH = 32, 32, 32

    y_true = torch.rand(BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH).to(device)
    y_pred_identical = y_true.clone().to(device)
    y_pred_different = torch.rand(BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH).to(device)

    # Initialize loss function
    loss_fn = CombinedLoss(alpha=0.85).to(device)

    # --- Test Case 1: Identical images ---
    # L1 should be 0, SSIM should be 1, Combined Loss should be 0
    loss_identical = loss_fn(y_pred_identical, y_true)
    print(f"\nLoss for identical images: {loss_identical.item():.6f}")
    assert torch.isclose(loss_identical, torch.tensor(0.0), atol=1e-5), "Loss for identical images should be close to 0"

    # --- Test Case 2: Different images ---
    # Loss should be a positive value
    loss_different = loss_fn(y_pred_different, y_true)
    print(f"Loss for different images: {loss_different.item():.6f}")
    assert loss_different.item() > 0, "Loss for different images should be positive"

    print("\nLoss function verification successful!")
