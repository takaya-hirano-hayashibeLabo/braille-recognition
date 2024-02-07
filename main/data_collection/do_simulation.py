from pathlib import Path
PARENT=Path(__file__).parent
ROOT=PARENT.parent.parent

import sys
sys.path.append(str(ROOT))
import os

import pandas
import numpy as np
from tqdm import tqdm
import argparse
import json

from src import Simulator,BrailleMaker

def main():
    """
    csvに記録された点字リストを作成し,シミューレーションした後データを集めるソースコード
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--dot_shape",default="default",
                        type=str,help="dot_shape:{'default', 'sloped'}",
                        required=True
                        )
    parser.add_argument(
        "--episode_num",default=1,type=int,
        help="同じ文字を何エピソード取得するか"
    )
    parser.add_argument(
        "--is_view",type=bool,default=False
    )
    parser.add_argument(
        "--touch_sensor_num",type=int,default=128
    )
    parser.add_argument("--save_dir",default="data",type=str)
    parser.add_argument("--is_handy_random",action="store_true")
    args=parser.parse_args()
    
    dot_shape=args.dot_shape
    
    braille_list_csv=pandas.read_csv(f"{PARENT}/braille_list.csv")
    # print(braille_list_csv)
    
    braille_maker=BrailleMaker()
    simulator=Simulator()


    save_dir=f"{PARENT}/{args.save_dir}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #>> コマンドライン引数の保存 >>
    with open(f"{save_dir}/param.json","w",encoding="utf-8") as f:
        json.dump(args.__dict__,f,indent=4)
    #>> コマンドライン引数の保存 >>


    for episode in range(args.episode_num):
        print(f"[EPISODE {episode+1}/{args.episode_num}]"+"-"*50)
        #>> ランダムにhandのy位置を決める >>
        hand_y=np.random.uniform(low=-0.0025, high=0.0025) * args.is_handy_random
        #>> ランダムにhandのy位置を決める >>

        for braille in (braille_list_csv.values):
            name,dot_pos_list=braille
            dot_pos_list=dot_pos_list.split()
            # print(f"{name}, {dot_pos_list}")
            
            braille_maker.raise_dot_resolution(N=240,dot_shape=dot_shape)
            braille_maker.make_braille_body(name=name,dot_pos_list=dot_pos_list,dot_shape=dot_shape)
            
            simulator.laod_braille_body(braille_name=name,dot_shape=dot_shape)
            simulator.simulate(
                is_view=args.is_view,
                hand_y=hand_y,
                save_dir=save_dir,
                touch_sensor_num=args.touch_sensor_num
            )

    
if __name__=="__main__":
    main()