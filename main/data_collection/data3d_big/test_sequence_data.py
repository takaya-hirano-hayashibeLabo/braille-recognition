"""
ちょうどいいSKIP_FRAMEを見つけるためのプログラム
"""
from pathlib import Path
PARENT=str(Path(__file__).parent)
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation as arani
import numpy as np


def main():

    SKIP_FRAME=20
    TIMESTEP=0.01
    TIME_SEQUENCE=16

    input_3d=np.load(f"{PARENT}/input_3d.npy")
    print(f"3d data size : {sys.getsizeof(input_3d)/1e9} G")
    
    fig=plt.figure(1)
    frames=[
        [plt.imshow(img)] for img
        in input_3d[10]
    ]
    ani=arani(fig,frames,interval=500)
    plt.show()
    # ani.save(f"{PARENT}/timestep{TIMESTEP}_timesequence{TIME_SEQUENCE}_skipframe{SKIP_FRAME}.mp4",writer="ffmpeg")


if __name__=="__main__":
    main()