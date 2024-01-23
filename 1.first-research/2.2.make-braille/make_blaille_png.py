import argparse
import os

from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scale',default=1.0,type=float, help='scale of braile size.') # 点字を何倍のサイズにするか
    parser.add_argument('--dot_where',default='111111',type=str,
                        help='(default: --dot_where 111111)\n represent where dot exists using 6bit.\n 0(head bit):left-top, 1:left-center, 2:left-bottom, 3:right-top, 4:right-center, 5(tail-bit):right-bottom')
    parser.add_argument('--img_name',default='test',type=str)
    args=parser.parse_args()
    
    # 点字の間隔[m] 参考url(https://web.econ.keio.ac.jp/staff/nakanoy/article/braille/BR/chap3/3-2/3-2.html)
    r=args.scale * 0.75 * 10**-3
    l=args.scale * 2.37 * 10**-3
    b=args.scale * 1.065* 10**-3
    h=args.scale * 4.585* 10**-3
    z=args.scale * 0.5* 10**-3
    c=args.scale * 3.27/2 * 10**-3
    
    # この中のdot_existをTrueにするとポチができる。Falseにするとポチが消える。
    brailles_param:dict={
        'l-top'   :{'pos':[-b,l,z], 'size':[r,z], 'dot_exist':True},
        'l-center':{'pos':[-b,0,z], 'size':[r,z], 'dot_exist':True},
        'l-bottom':{'pos':[-b,-l,z],'size':[r,z], 'dot_exist':False},
        'r-top'   :{'pos':[b,l,z],  'size':[r,z], 'dot_exist':False},
        'r-center':{'pos':[b,0,z],  'size':[r,z], 'dot_exist':True},
        'r-bottom':{'pos':[b,-l,z], 'size':[r,z], 'dot_exist':False},
    }
    
    # >> dotの位置をコマンドライン引数で指定 >>
    dot_where=list(args.dot_where)
    for is_dot,key in zip(dot_where,brailles_param.keys()):
        brailles_param[key]['dot_exist']=True if is_dot=='1' else False
    # >> dotの位置をコマンドライン引数で指定 >>
    
    plt.style.use('dark_background')
    fig,ax=plt.subplots(1,1,figsize=(1,2.58))
    for key, val in brailles_param.items():
        if not val['dot_exist']:
            continue
        dot=patches.Circle(xy=(val['pos'][0],val['pos'][1]), radius=val['size'][0],color='white')
        ax.add_patch(dot)
        
    fig.subplots_adjust(0,0,1,1)
    ax.set_aspect('equal')
    ax.set_xlim([-(b+c),b+c])
    ax.set_ylim([-(l+h),l+h])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
        
    # plt.show()
    plt.savefig(f"tmp.png")
    
    scale=2
    WIDTH,HEIGHT=54,139
    img_org=Image.open('tmp.png')
    img_resized=img_org.resize((WIDTH*scale,HEIGHT*scale))
    img_resized.save(f"{args.img_name}.png",quality=90)
    os.remove("tmp.png")
    
    
if __name__=='__main__':
    main()