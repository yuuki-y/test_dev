import torch
from torch.utils.data import Dataset
import os
from pathlib import Path

class CTDataset(Dataset):
    """
    X線画像（AP/LAT）と3D CTボリュームのペアを読み込むためのカスタムDatasetクラス。
    """
    def __init__(self, base_dir, base_dir2, transform=None):
        """
        Args:
            base_dir (string): X線画像が格納されているベースディレクトリ。
                                  構造: base_dir/{patient_id}/AP.pt, LAT.pt
            base_dir2 (string): CTボリュームが格納されているベースディレクトリ。
                                   構造: base_dir2/{patient_id}.pt
            transform (callable, optional): サンプルに適用されるオプションの変換。
        """
        self.base_dir = Path(base_dir)
        self.base_dir2 = Path(base_dir2)
        self.transform = transform

        # base_dir内のサブディレクトリ名を患者IDとして取得
        self.patient_ids = sorted([d.name for d in self.base_dir.iterdir() if d.is_dir()])

        # base_dir2にも対応するCTファイルが存在するか検証 (任意)
        self.patient_ids = [
            pid for pid in self.patient_ids
            if (self.base_dir2 / f"{pid}.pt").exists()
        ]

    def __len__(self):
        """
        データセットの総サンプル数を返す。
        """
        return len(self.patient_ids)

    def __getitem__(self, idx):
        """
        指定されたインデックスのサンプル（X線画像とCTボリューム）を生成する。
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.patient_ids[idx]

        # ファイルパスを構築
        xray_ap_path = self.base_dir / patient_id / 'AP.pt'
        xray_lat_path = self.base_dir / patient_id / 'LAT.pt'
        ct_volume_path = self.base_dir2 / f"{patient_id}.pt"

        # PyTorchテンソルとしてデータをロード
        # .ptファイルは torch.save で保存されていることを想定
        try:
            xray_ap = torch.load(xray_ap_path)
            xray_lat = torch.load(xray_lat_path)
            ct_volume = torch.load(ct_volume_path)
        except FileNotFoundError as e:
            print(f"データ読み込みエラー (Patient ID: {patient_id}): {e}")
            # エラーが発生した場合、Noneを返してDataLoader側でスキップさせるなどの対応が可能
            return None, None

        # 2枚のX線画像をスタックして単一のテンソルにする
        # 入力Shape: (2, 256, 256)
        xrays = torch.stack([xray_ap, xray_lat], dim=0)

        # 出力Shape: (256, 256, 256)
        # 必要に応じてここで形状チェックや変換を行う
        # 例: ct_volume = ct_volume.permute(2, 0, 1) # (D, H, W) -> (D, H, W)

        sample = {'xrays': xrays, 'ct_volume': ct_volume}

        if self.transform:
            sample = self.transform(sample)

        return sample['xrays'], sample['ct_volume']

def collate_fn(batch):
    """
    __getitem__でNoneが返された場合にバッチから除外するためのカスタムcollate関数。
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

if __name__ == '__main__':
    # --- このファイル単体で実行した場合のテスト用コード ---
    # ダミーデータを作成して動作確認

    # ダミーデータディレクトリの作成
    data_path = Path('./dummy_data/xrays')
    label_path = Path('./dummy_data/ct_volumes')
    data_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)

    # ダミーの患者データを作成
    num_patients = 5
    for i in range(num_patients):
        patient_id = f"patient_{i:03d}"
        patient_dir = data_path / patient_id
        patient_dir.mkdir(exist_ok=True)

        # ダミーのX線画像を.ptファイルとして保存 (2, 256, 256)
        dummy_ap = torch.randn(256, 256)
        dummy_lat = torch.randn(256, 256)
        torch.save(dummy_ap, patient_dir / 'AP.pt')
        torch.save(dummy_lat, patient_dir / 'LAT.pt')

        # ダミーのCTボリュームを.ptファイルとして保存 (256, 256, 256)
        dummy_ct = torch.randn(256, 256, 256)
        torch.save(dummy_ct, label_path / f"{patient_id}.pt")

    print("ダミーデータを作成しました。")

    # DatasetとDataLoaderのインスタンスを作成
    dataset = CTDataset(base_dir='./dummy_data/xrays', base_dir2='./dummy_data/ct_volumes')

    # DataLoaderを作成（カスタムcollate_fnを使用）
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    print(f"データセットのサイズ: {len(dataset)}")

    # データを1バッチだけ取り出して形状を確認
    xrays_batch, ct_batch = next(iter(dataloader))

    if xrays_batch is not None:
        print(f"X線画像のバッチ形状: {xrays_batch.shape}") # 期待: (2, 2, 256, 256)
        print(f"CTボリュームのバッチ形状: {ct_batch.shape}") # 期待: (2, 256, 256, 256)
    else:
        print("データローダーからバッチを取得できませんでした。")

    # ダミーデータをクリーンアップ
    import shutil
    shutil.rmtree('./dummy_data')
    print("ダミーデータを削除しました。")
