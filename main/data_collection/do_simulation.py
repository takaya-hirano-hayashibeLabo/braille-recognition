from pathlib import Path
PARENT=Path(__file__).parent
ROOT=PARENT.parent.parent

import sys
sys.path.append(str(ROOT))

import pandas
from tqdm import tqdm

from src import Simulator,BrailleMaker

def main():
    """
    csvに記録された点字リストを作成し,シミューレーションした後データを集めるソースコード
    """
    
    braille_list_csv=pandas.read_csv(f"{PARENT}/braille_list.csv")
    # print(braille_list_csv)
    
    braille_maker=BrailleMaker()
    simulator=Simulator()
    
    for braille in (braille_list_csv.values):
        name,dot_pos_list=braille
        dot_pos_list=dot_pos_list.split()
        # print(f"{name}, {dot_pos_list}")
        
        braille_maker.raise_dot_resolution(N=240)
        braille_maker.make_braille_body(name=name,dot_pos_list=dot_pos_list)
        
        simulator.laod_braille_body(braille_name=name)
        simulator.simulate(
            is_view=False,
            save_dir=f"{PARENT}/data"
        )
    
if __name__=="__main__":
    main()