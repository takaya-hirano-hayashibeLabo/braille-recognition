import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Read the first CSV file
df1 = pd.read_csv('/mnt/ssd1/hiranotakaya/master/dev/braille-recognition/main/exps/2024_02_multi_seminar/exp1/result/nn/result.csv')

# Read the second CSV file
df2 = pd.read_csv('/mnt/ssd1/hiranotakaya/master/dev/braille-recognition/main/exps/2024_02_multi_seminar/exp1/result/snn/result.csv')

# Calculate mean and std for columns starting from the 2nd column for both files
mean_values_df1 = df1.iloc[:, 1:].mean(axis=1)
std_values_df1 = df1.iloc[:, 1:].std(axis=1)

mean_values_df2 = df2.iloc[:, 1:].mean(axis=1)
std_values_df2 = df2.iloc[:, 1:].std(axis=1)

# Plot the graph for the first file
plt.figure(figsize=(10, 6))
plt.plot(df1['frac'], mean_values_df1, label='Mean (NN)')
plt.fill_between(df1['frac'],
                   mean_values_df1-std_values_df1,
                   mean_values_df1+std_values_df1,
                   alpha=0.3
              )

# Plot the graph for the second file
plt.plot(df2['frac'], mean_values_df2, label='Mean (SNN)')
plt.fill_between(df2['frac'],
                   mean_values_df2-std_values_df2,
                   mean_values_df2+std_values_df2,
                   alpha=0.3
                 )
plt.xlabel('pressure scale')
plt.ylabel('Values')
plt.legend()
plt.title('Mean and std for Records')
plt.savefig(Path(__file__).parent / "result/view_score.png")