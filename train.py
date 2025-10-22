import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DX2CT
from dataset import CTDataset, collate_fn
from loss import DX2CTLoss

import logging

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 拡散スケジューラ関連 ---
def get_linear_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    線形のノイズスケジュール（beta）と、それに基づくalpha、alpha_cumprodを生成する。
    これらは拡散プロセス（ノイズ付加）と逆拡散プロセス（ノイズ除去）の両方で使われる。
    """
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return alphas_cumprod

def add_noise_to_slices(slices, t, alphas_cumprod):
    """
    元のスライス（x0）に、指定されたタイムステップtのノイズを付加してx_tを生成する。
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    """
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod[t])

    # テンソルの形状を (B*D, 1, 1, 1) に変形してブロードキャスト可能にする
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1, 1).to(slices.device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1, 1).to(slices.device)

    noise = torch.randn_like(slices)
    noisy_slices = sqrt_alphas_cumprod * slices + sqrt_one_minus_alphas_cumprod * noise

    return noisy_slices, noise

# --- 学習パイプライン ---
def train(config):
    # --- 1. デバイスとモデルのセットアップ ---
    if torch.cuda.device_count() < 3:
        logging.warning("利用可能なGPUが3台未満です。CPUでモデル並列をシミュレートします。")
        devices = ['cpu', 'cpu', 'cpu']
    else:
        devices = [f'cuda:{i}' for i in range(3)]
    logging.info(f"使用デバイス: {devices}")

    D, H, W = config['volume_shape']
    model = DX2CT(devices=devices, max_depth=D, base_dim=config['base_dim'])

    # --- 2. データセットの準備 ---
    dataset = CTDataset(base_dir=config['data_path'], base_dir2=config['label_path'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    # --- 3. 損失関数とオプティマイザ ---
    loss_fn = DX2CTLoss(device=devices[-1],
                        xray_loss_weight=config['xray_loss_weight'],
                        volume_shape=config['volume_shape'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # --- 4. 拡散プロセスの準備 ---
    alphas_cumprod = get_linear_noise_schedule(timesteps=config['timesteps']).to(devices[0])

    # --- 5. 学習ループ ---
    logging.info("学習を開始します。")
    for epoch in range(config['epochs']):
        for i, (xrays, ct_volumes) in enumerate(dataloader):
            if xrays is None: continue # データロードに失敗したバッチはスキップ

            B, D, H, W = ct_volumes.shape

            # --- データを最初のデバイスに移動 ---
            xrays = xrays.to(devices[0])
            ct_volumes = ct_volumes.to(devices[0])

            optimizer.zero_grad()

            total_loss_accum = 0.0
            diff_loss_accum = 0.0
            xray_loss_accum = 0.0

            # メモリ節約のため、1ボリュームずつ、さらにスライスごとに処理
            for b in range(B):
                ct_volume = ct_volumes[b] # (D, H, W)
                xray_pair = xrays[b]     # (2, H, W)

                # --- 拡散プロセスの実行 ---
                t = torch.randint(0, config['timesteps'], (D,), device=devices[0]).long()
                ct_slices = ct_volume.unsqueeze(1) # (D, 1, H, W)
                noisy_slices, target_noise = add_noise_to_slices(ct_slices, t, alphas_cumprod)

                # --- スライスごとのフォワード・バックワード ---
                predicted_noise_slices = []
                for d in range(D):
                    # モデルへの入力 (バッチサイズ=1)
                    noisy_slice = noisy_slices[d].unsqueeze(0)
                    time_step = t[d].unsqueeze(0)
                    z_index = torch.tensor([d], device=devices[0])
                    xray_expanded = xray_pair.unsqueeze(0)

                    # フォワードパス
                    predicted_noise = model(noisy_slice, time_step, z_index, xray_expanded)
                    predicted_noise_slices.append(predicted_noise)

                # 全スライスの予測が完了してから損失を計算
                predicted_noise_all = torch.cat(predicted_noise_slices, dim=0)

                # 損失計算のためにテンソルを移動
                target_noise = target_noise.to(devices[-1])
                noisy_slices_loss = noisy_slices.to(devices[-1])
                t_loss = t.to(devices[-1])
                xrays_loss = xray_pair.unsqueeze(0).to(devices[-1]) # (1, 2, H, W)
                alphas_cumprod_loss = alphas_cumprod.to(devices[-1])

                # 損失計算 (バッチサイズ=1で)
                total_loss, diff_loss, xray_loss = loss_fn(
                    predicted_noise_all, target_noise, noisy_slices_loss, t_loss, xrays_loss, alphas_cumprod_loss
                )

                # 勾配を累積
                (total_loss / B).backward() # バッチ全体で平均化するために割る

                total_loss_accum += total_loss.item()
                diff_loss_accum += diff_loss.item()
                xray_loss_accum += xray_loss.item()

            # バッチ全体の勾配が累積された後で重みを更新
            optimizer.step()

            # ログ表示用に損失を平均化
            total_loss_avg = total_loss_accum / B
            diff_loss_avg = diff_loss_accum / B
            xray_loss_avg = xray_loss_accum / B

            if (i + 1) % config['log_interval'] == 0:
                logging.info(f"Epoch [{epoch+1}/{config['epochs']}], Step [{i+1}/{len(dataloader)}], "
                             f"Total Loss: {total_loss_avg:.4f}, "
                             f"Diffusion Loss: {diff_loss_avg:.4f}, "
                             f"X-ray Loss: {xray_loss_avg:.4f}")

    logging.info("学習が完了しました。")


# --- このファイル単体で実行した場合のテスト用コード ---
if __name__ == '__main__':
    import os
    import shutil
    from pathlib import Path

    # --- テスト開始前に古いデータをクリーンアップ ---
    if Path('./dummy_data_train').exists():
        shutil.rmtree('./dummy_data_train')

    # --- 1. テスト用のダミーデータを作成 ---
    logging.info("テスト用のダミーデータを作成します。")

    # メモリを節約するため、小さなボリュームでテスト
    D, H, W = 64, 64, 64

    data_path = Path('./dummy_data_train/xrays')
    label_path = Path('./dummy_data_train/ct_volumes')
    data_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)

    num_patients = 2 # 複数の患者データでテスト
    for i in range(num_patients):
        patient_id = f"patient_{i:03d}"
        patient_dir = data_path / patient_id
        patient_dir.mkdir(exist_ok=True)
        # X線画像: (H, W)
        torch.save(torch.randn(H, W), patient_dir / 'AP.pt')
        torch.save(torch.randn(H, W), patient_dir / 'LAT.pt')
        # CTボリューム: (D, H, W)
        torch.save(torch.randn(D, H, W), label_path / f"{patient_id}.pt")

    # --- 2. 学習設定 ---
    config = {
        'data_path': './dummy_data_train/xrays',
        'label_path': './dummy_data_train/ct_volumes',
        'batch_size': 2,
        'epochs': 1,
        'learning_rate': 1e-4,
        'timesteps': 1000,
        'xray_loss_weight': 0.1,
        'log_interval': 1,
        'volume_shape': [D, H, W],
        'base_dim': 16
    }

    # --- 3. 学習パイプラインを実行 ---
    try:
        # 1エポック、1ステップだけ実行して動作確認
        train(config)
        logging.info("✅ 学習パイプラインのテストが正常に完了しました。")
    except Exception as e:
        logging.error(f"❌ 学習パイプラインのテスト中にエラーが発生しました: {e}", exc_info=True)
    finally:
        # --- 4. ダミーデータをクリーンアップ ---
        logging.info("テスト用のダミーデータを削除します。")
        shutil.rmtree('./dummy_data_train')
