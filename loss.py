import torch
import torch.nn as nn
import torch.nn.functional as F

# DiffDRRのモジュールをインポート
try:
    from diffdrr.drr import DRR
    from diffdrr.projectors.siddon import Siddon
    DIFFDRR_AVAILABLE = True
except ImportError:
    print("警告: diffdrrライブラリが見つかりません。X線一貫性損失は計算されません。")
    DIFFDRR_AVAILABLE = False

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

        # --- DiffDRRの射影ジオメトリを設定 ---
        sdd = 1200.0  # Source-to-Detector Distance (mm)
        detector_height = self.volume_shape[1]
        detector_width = self.volume_shape[2]

        # DRRを生成するためのモジュールを初期化
        # プロジェクタとしてSiddonを使用
        self.drr_projector = Siddon(volume_shape=self.volume_shape,
                                    voxel_spacing=[pixel_size, pixel_size, pixel_size],
                                    device=self.device)

        # AP (前方) と LAT (側面) の2つのビューを定義
        # APビュー: Z軸から撮影
        rotations_ap = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        translations_ap = torch.tensor([[0.0, 0.0, -sdd/2]], device=self.device)

        # LATビュー: Y軸周りに90度回転
        rotations_lat = torch.tensor([[0.0, 90.0 * (3.14159 / 180.0), 0.0]], device=self.device)
        translations_lat = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)

        # DRRモジュールをAPとLATビュー用にそれぞれ作成
        self.drr_ap = DRR(self.drr_projector, sdd, detector_height, detector_width, pixel_size,
                          rotations=rotations_ap, translations=translations_ap)
        self.drr_lat = DRR(self.drr_projector, sdd, detector_height, detector_width, pixel_size,
                           rotations=rotations_lat, translations=translations_lat)


    def _predict_x0_from_noise(self, x_t, t, noise, alphas_cumprod):
        """
        DDPMの公式を用いて、予測されたノイズからクリーンな画像 x0 を推定する。
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
        """
        # t に対応する alpha_cumprod を取得
        sqrt_alpha_bar_t = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alphas_cumprod[t]).view(-1, 1, 1, 1)

        # デバイスを合わせる
        sqrt_alpha_bar_t = sqrt_alpha_bar_t.to(x_t.device)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.to(x_t.device)

        # x0 を予測
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * noise) / sqrt_alpha_bar_t
        return pred_x0

    def forward(self,
                predicted_noise_slices,  # モデルが予測したノイズ (B*D, 1, H, W)
                target_noise_slices,     # 実際のノイズ (B*D, 1, H, W)
                noisy_ct_slices,         # ノイズが付与されたCTスライス (B*D, 1, H, W)
                timesteps,               # 拡散ステップ (B*D)
                ground_truth_xrays,      # 正解のX線画像 (B, 2, H, W)
                diffusion_alphas_cumprod # 拡散スケジューラのalpha_cumprod
               ):

        # 1. 2D拡散損失の計算
        diffusion_loss = F.mse_loss(predicted_noise_slices, target_noise_slices)

        # 2. X線一貫性損失の計算
        xray_consistency_loss = torch.tensor(0.0, device=self.device)

        if self.xray_loss_weight > 0 and DIFFDRR_AVAILABLE:
            batch_size = ground_truth_xrays.shape[0]
            depth = 256

            # 予測されたノイズから、予測されたクリーンなCTスライス(x0)を計算
            pred_x0_slices = self._predict_x0_from_noise(
                noisy_ct_slices, timesteps, predicted_noise_slices, diffusion_alphas_cumprod
            )

            # (B*D, 1, H, W) -> (B, D, H, W) に形状を戻す
            pred_x0_volumes = pred_x0_slices.view(batch_size, depth,
                                                  pred_x0_slices.shape[2],
                                                  pred_x0_slices.shape[3])

            # 生成された3DボリュームからDRRをレンダリング
            # ボリュームの値をDRRに適した範囲にクリップ (例: 0以上)
            pred_x0_volumes_clipped = pred_x0_volumes.clamp(min=0)

            # 各ボリュームに対してDRRを生成し、損失を計算
            for i in range(batch_size):
                volume = pred_x0_volumes_clipped[i].unsqueeze(0) # (1, D, H, W)

                # DRRを生成
                drr_ap_pred = self.drr_ap(volume)
                drr_lat_pred = self.drr_lat(volume)

                # 正解X線画像と比較 (AP: 0番目, LAT: 1番目)
                gt_ap = ground_truth_xrays[i, 0].unsqueeze(0)
                gt_lat = ground_truth_xrays[i, 1].unsqueeze(0)

                # L1損失を計算し、累積
                xray_consistency_loss += F.l1_loss(drr_ap_pred, gt_ap)
                xray_consistency_loss += F.l1_loss(drr_lat_pred, gt_lat)

            # バッチサイズで平均化
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

        loss_fn = DX2CTLoss(device=device, xray_loss_weight=0.5)

        # ダミー入力データを作成
        B, D, H, W = 2, 256, 256, 256

        predicted_noise = torch.randn(B * D, 1, H, W).to(device)
        target_noise = torch.randn(B * D, 1, H, W).to(device)
        noisy_slices = torch.randn(B * D, 1, H, W).to(device)
        timesteps = torch.randint(0, 1000, (B * D,)).to(device)
        xrays = torch.rand(B, 2, H, W).to(device) # [0, 1]の範囲

        # ダミーの拡散スケジュールを作成
        alphas_cumprod = torch.linspace(0.99, 0.01, 1000).to(device)

        # 損失を計算
        total_loss, diff_loss, xray_loss = loss_fn(
            predicted_noise, target_noise, noisy_slices, timesteps, xrays, alphas_cumprod
        )

        print(f"Total Loss: {total_loss.item():.4f}")
        print(f"  - Diffusion Loss: {diff_loss.item():.4f}")
        print(f"  - X-ray Consistency Loss: {xray_loss.item():.4f}")

        # バックワードパスをテスト
        try:
            total_loss.backward()
            print("✅ バックワードパスのテストに成功しました。")
        except Exception as e:
            print(f"❌ バックワードパスのテストでエラーが発生しました: {e}")
