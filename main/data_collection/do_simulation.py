from pathlib import Path
PARENT=Path(__file__).parent
ROOT=PARENT.parent.parent

import sys
sys.path.append(str(ROOT))

import pandas
from tqdm import tqdm
import argparse

from src import Simulator,BrailleMaker

def main():
    """
    csvに記録された点字リストを作成し,シミューレーションした後データを集めるソースコード
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--dot_shape",default="default",
                        type=str,help="dot_shape:{'default', 'sloped'}"
                        )
    args=parser.parse_args()
    
    dot_shape=args.dot_shape
    
    braille_list_csv=pandas.read_csv(f"{PARENT}/braille_list.csv")
    # print(braille_list_csv)
    
    braille_maker=BrailleMaker()
    simulator=Simulator()
    if dot_shape=="default":
        save_dir=f"{PARENT}/data"
    elif dot_shape=="sloped":
        save_dir=f"{PARENT}/data-sloped"
    
    for braille in (braille_list_csv.values):
        name,dot_pos_list=braille
        dot_pos_list=dot_pos_list.split()
        # print(f"{name}, {dot_pos_list}")
        
        braille_maker.raise_dot_resolution(N=240,dot_shape=dot_shape)
        braille_maker.make_braille_body(name=name,dot_pos_list=dot_pos_list,dot_shape=dot_shape)
        
        simulator.laod_braille_body(braille_name=name,dot_shape=dot_shape)
        simulator.simulate(
            is_view=False,
            save_dir=save_dir
        )
    
if __name__=="__main__":
    main()