# たくさんのデータを集める際のシェル

DATA_ROOT_NAME="train_data"
RAW_DIR_NAME="original"
CONVERTED_DIR_NAME="3d_data"

python ./do_simulation.py --dot_shape default --episode_num 10 --touch_sensor_num 128  --save_dir "${DATA_ROOT_NAME}/${RAW_DIR_NAME}"  --is_handy_random
python ./to_3d.py --dot_shape default --start_time 8.4 --end_time 11.0 --save_dir "${DATA_ROOT_NAME}/${CONVERTED_DIR_NAME}"   --data_dir "${DATA_ROOT_NAME}/${RAW_DIR_NAME}"

echo done