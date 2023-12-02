import argparse
import os

from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--img_name',default='test',type=str)
    args=parser.parse_args()
    
    r=0.75 * 10**-3
    dot=patches.Circle(xy=(0,0),radius=r,color='white')
    
    plt.style.use('dark_background')
    fig,ax=plt.subplots(1,1,figsize=(1,1))
    ax.add_patch(dot)
        
    fig.subplots_adjust(0,0,1,1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-r,r])
    ax.set_ylim([-r,r])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
        
    # plt.show()
    plt.savefig(f"tmp.png")
    
    scale=15
    WIDTH,HEIGHT=r*1e4,r*1e4
    img_org=Image.open('tmp.png')
    img_resized=img_org.resize((round(WIDTH*scale),round(HEIGHT*scale)))
    img_resized.save(f"{args.img_name}.png",quality=90)
    os.remove("tmp.png")
    
    
if __name__=='__main__':
    main()