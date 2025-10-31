import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# DiffDRRのモジュールをインポート
try:
    from diffdrr.drr import DRR
    from diffdrr.projectors.siddon import Siddon
    DIFFDRR_AVAILABLE = True
except ImportError:
    print("警告: diffdrrライブラリが見つかりません。X線一貫性損失は計算されません。")
    DIFFDRR_AVAILABLE = False

def euler_to_quaternion(euler_deg, device):
    """
    Euler角 (roll, pitch, yaw) をクォータニオン (w, x, y, z) に変換する。
    Args:
        euler_deg (tuple of float): (roll, pitch, yaw) in degrees.
        device: The torch device.
    """
    roll, pitch, yaw = [torch.deg2rad(torch.tensor(a, dtype=torch.float32)) for a in euler_deg]
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    # Shape: (1, 4)
    return torch.tensor([[w, x, y, z]], dtype=torch.float32, device=device)


class DX2CTLoss(nn.Module):
    """
    DX2CTモデルのための複合損失関数。
    1. 2D拡散損失 (MSE of predicted noise)
    2. X線一貫性損失 (L1 of DRR projections)
    """
    def __init__(self, device='cuda:0', xray_loss_weight=1.0, volume_shape=[256, 256, 256], pixel_size=1.0):
        super().__init__()
        self.device = device
        self.xray_loss_weight = xray_loss_weight
        self.volume_shape = volume_shape

        if not DIFFDRR_AVAILABLE:
            print("DiffDRRが利用不可のため、X線一貫性損失は無効化されます。")
            return

        # --- 新しいDiffDRRの射影ジオメトリを設定 ---
        sdd = 1000.0  # Source-to-Detector Distance (mm)
        delx = 2.0    # Detector pixel size (mm)

        detector_height = self.volume_shape[1]

        # DRRプロジェクタを初期化
        self.drr_projector = Siddon(
            volume_shape=self.volume_shape,
            voxel_spacing=[pixel_size, pixel_size, pixel_size],
            device=self.device
        )

        # 単一のDRRモジュールを初期化
        self.drr = DRR(
            self.drr_projector, sdd=sdd, height=detector_height, delx=delx
        ).to(self.device)

        # --- 回転と平行移動を定義し、バッファとして登録 ---
        # 1. AP (前方) と LAT (側面) の回転角 [deg]
        rotations_deg = {
            "AP": (0, 0, 180),
            "LAT": (0, 0, -90),
        }

        # 2. Euler角をクォータニオンに変換
        quat_ap = euler_to_quaternion(rotations_deg["AP"], self.device)
        quat_lat = euler_to_quaternion(rotations_deg["LAT"], self.device)
        self.register_buffer('quat_ap', quat_ap)
        self.register_buffer('quat_lat', quat_lat)

        # 3. 平行移動ベクトル (mm)
        translations = torch.tensor([[0.0, 500.0, 0.0]], device=self.device)
        self.register_buffer('translations', translations)


    def _predict_x0_from_noise(self, x_t, t, noise, alphas_cumprod):
        """
        DDPMの公式を用いて、予測されたノイズからクリーンな画像 x0 を推定する。
        """
        sqrt_alpha_bar_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1).to(x_t.device)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alphas_cumprod[t]).view(-1, 1, 1, 1).to(x_t.device)
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * noise) / sqrt_alpha_bar_t
        return pred_x0

    def forward(self,
                predicted_noise_slices,
                target_noise_slices,
                noisy_ct_slices,
                timesteps,
                ground_truth_xrays,
                diffusion_alphas_cumprod
               ):

        # 1. 2D拡散損失の計算
        diffusion_loss = F.mse_loss(predicted_noise_slices, target_noise_slices)

        # 2. X線一貫性損失の計算
        xray_consistency_loss = torch.tensor(0.0, device=self.device)

        if self.xray_loss_weight > 0 and DIFFDRR_AVAILABLE:
            batch_size = ground_truth_xrays.shape[0]

            pred_x0_slices = self._predict_x0_from_noise(
                noisy_ct_slices, timesteps, predicted_noise_slices, diffusion_alphas_cumprod
            )

            pred_x0_volumes = pred_x0_slices.view(
                batch_size, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2]
            )

            pred_x0_volumes_clipped = pred_x0_volumes.clamp(min=0)

            for i in range(batch_size):
                volume = pred_x0_volumes_clipped[i].unsqueeze(0)

                # クォータニオンを使ってDRRを生成
                drr_ap_pred = self.drr(volume, self.quat_ap, self.translations, parameterization="quaternion")
                drr_lat_pred = self.drr(volume, self.quat_lat, self.translations, parameterization="quaternion")

                gt_ap = ground_truth_xrays[i, 0].unsqueeze(0)
                gt_lat = ground_truth_xrays[i, 1].unsqueeze(0)

                xray_consistency_loss += F.l1_loss(drr_ap_pred, gt_ap)
                xray_consistency_loss += F.l1_loss(drr_lat_pred, gt_lat)

            xray_consistency_loss = xray_consistency_loss / batch_size

        # 3. 2つの損失を結合
        total_loss = diffusion_loss + self.xray_loss_weight * xray_consistency_loss

        return total_loss, diffusion_loss, xray_consistency_loss


# --- このファイル単体で実行した場合のテスト用コード ---
if __name__ == '__main__':
    if not DIFFDRR_AVAILABLE:
        print("DiffDRRがインストールされていないため、テストをスキップします。")
    else:
        print("DX2CTLossのテストを実行します。")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # 小さなボリュームでテスト
        D, H, W = 64, 64, 128 # Detector width != height
        loss_fn = DX2CTLoss(
            device=device, xray_loss_weight=0.5, volume_shape=[D, H, W], pixel_size=1.5
        )

        # ダミー入力データを作成
        B = 2
        predicted_noise = torch.randn(B * D, 1, H, W).to(device)
        target_noise = torch.randn(B * D, 1, H, W).to(device)
        noisy_slices = torch.randn(B * D, 1, H, W).to(device)
        timesteps = torch.randint(0, 1000, (B * D,)).to(device)
        # DRRの出力サイズは (B, H, W) になる
        xrays = torch.rand(B, 2, H, W).to(device)

        alphas_cumprod = torch.linspace(0.99, 0.01, 1000).to(device)

        total_loss, diff_loss, xray_loss = loss_fn(
            predicted_noise, target_noise, noisy_slices, timesteps, xrays, alphas_cumprod
        )

        print(f"Total Loss: {total_loss.item():.4f}")
        print(f"  - Diffusion Loss: {diff_loss.item():.4f}")
        print(f"  - X-ray Consistency Loss: {xray_loss.item():.4f}")

        try:
            total_loss.backward()
            print("✅ バックワードパスのテストに成功しました。")
        except Exception as e:
            print(f"❌ バックワードパスのテストでエラーが発生しました: {e}")
