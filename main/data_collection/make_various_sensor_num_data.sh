# 接触センサの数をいろんな数にしたときのデータ収集用shell


# >> いろんなセンサ数にしたときのシミュレーション >>
makedirs various_sensornum_data/original

python ./do_simulation.py --dot_shape default --episode_num 1 --touch_sensor_num 8  --save_dir various_sensornum_data/original/data_8x8_sensor_grid
python ./do_simulation.py --dot_shape default --episode_num 1 --touch_sensor_num 16 --save_dir various_sensornum_data/original/data_16x16_sensor_grid
python ./do_simulation.py --dot_shape default --episode_num 1 --touch_sensor_num 32 --save_dir various_sensornum_data/original/data_32x32_sensor_grid
python ./do_simulation.py --dot_shape default --episode_num 1 --touch_sensor_num 64 --save_dir various_sensornum_data/original/data_64x64_sensor_grid
# >> いろんなセンサ数にしたときのシミュレーション >>


# >> データを3次元データに変換
python ./to_3d.py --dot_shape default --start_time 1.0 --end_time 11.0 --save_dir various_sensornum_data/data3d/data_8x8_sensor_grid   --data_dir various_sensornum_data/original/data_8x8_sensor_grid
python ./to_3d.py --dot_shape default --start_time 1.0 --end_time 11.0 --save_dir various_sensornum_data/data3d/data_16x16_sensor_grid --data_dir various_sensornum_data/original/data_16x16_sensor_grid
python ./to_3d.py --dot_shape default --start_time 1.0 --end_time 11.0 --save_dir various_sensornum_data/data3d/data_32x32_sensor_grid --data_dir various_sensornum_data/original/data_32x32_sensor_grid
python ./to_3d.py --dot_shape default --start_time 1.0 --end_time 11.0 --save_dir various_sensornum_data/data3d/data_64x64_sensor_grid --data_dir various_sensornum_data/original/data_64x64_sensor_grid
# >> データを3次元データに変換


echo done