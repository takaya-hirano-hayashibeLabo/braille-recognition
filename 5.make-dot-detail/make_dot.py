from pathlib import Path
PARENT=str(Path(__file__).parent)

import argparse
import numpy as np
import re

MINI_DOT_GEOM="""<geom type='cylinder' size='{R} 0.0005' pos='{XY} 0'/>\n"""

DOT_BODY="""
<body pos='0 0 0'>
    {MINI_DOTS}
</body>
"""

DOT_RADIUS=0.00075 #[m] 1dotの半径

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--N',default=400,type=int,help='dot resolution')
    args=parser.parse_args()
    
    N=args.N
    
    r=DOT_RADIUS*np.sqrt(np.pi/N)/2
    
    xx,yy=np.meshgrid(
        np.arange(-DOT_RADIUS,DOT_RADIUS+2*r,2*r),
        np.arange(-DOT_RADIUS,DOT_RADIUS+2*r,2*r)
    )
    
    mini_dots=""
    idx_in_dot=np.where(xx**2+yy**2<=DOT_RADIUS**2)
    for x,y in zip(xx[idx_in_dot],yy[idx_in_dot]):
        mini_dot_geom_i=re.sub("{R}",f"{r}",MINI_DOT_GEOM)
        mini_dot_geom_i=re.sub("{XY}",f"{x} {y}",mini_dot_geom_i)
        mini_dots+=mini_dot_geom_i
    dot_body=re.sub("{MINI_DOTS}",f"{mini_dots}",DOT_BODY)
    
    with open(f"{PARENT}/detail_dot.xml","w") as f:
        f.write(dot_body)
        
    
    
if __name__=="__main__":
    main()
