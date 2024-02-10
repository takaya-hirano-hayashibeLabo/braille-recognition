import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ファイルパス
nn_file_path = '/mnt/ssd1/hiranotakaya/master/dev/braille-recognition/main/exps/2024_02_multi_seminar/exp4/result/nn/result.csv'
snn_file_path = '/mnt/ssd1/hiranotakaya/master/dev/braille-recognition/main/exps/2024_02_multi_seminar/exp4/result/snn/result.csv'

# CSVファイルを読み込む
nn_df = pd.read_csv(nn_file_path)
snn_df = pd.read_csv(snn_file_path)

# mask_rateをx軸に設定
x = nn_df['mask_size']

# 平均値を計算
nn_means = nn_df.values[:,1:].mean(axis=1)
snn_means = snn_df.values[:,1:].mean(axis=1)

# グラフをプロット
plt.figure(figsize=(10, 6))
plt.plot(x, nn_means, label='NN', marker='o')
plt.plot(x, snn_means, label='SNN', marker='x')

# グラフのタイトルと軸ラベルを設定
plt.title('Mask Size vs. Average Accuracy')
plt.xlabel('Mask size')
plt.ylabel('Average Accuracy')

# 凡例を表示
plt.legend()

# グラフを表示
plt.savefig(Path(__file__).parent/f"view_score.png")