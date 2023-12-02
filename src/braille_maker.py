from pathlib import Path
PARENT=str(Path(__file__).parent)

import numpy as np
import re


MINI_DOT_GEOM="""<geom type='cylinder' size='{R} 0.0005' pos='{XY} 0'/>\n"""

DOT_BODY="""<!--N={N}-->
<body pos='{XY} 0' name='{NAME}'>
    {MINI_DOTS}
</body>"""

DOT_RADIUS=0.00075 #[m] 1dotの半径


# 点字の間隔[m] 参考url(https://web.econ.keio.ac.jp/staff/nakanoy/article/braille/BR/chap3/3-2/3-2.html)
r=DOT_RADIUS
l=2.37 * 10**-3
b=1.065* 10**-3
h=4.585* 10**-3
z=0.5* 10**-3
c=3.27/2 * 10**-3

BRAILLE_BODY="""<!--{NAME}.xml-->
<body pos='0 0 0' name='{NAME}'>
    <geom type='box' size='{XYZ}' rgba='0 0.5 0 1'/>
    {DOT_BODIES}
</body>"""


class BrailleMaker():
    def __init__(self):
        pass
    
    def raise_dot_resolution(self,N=240):
        """
        1dotをN個のdotの集合にすることで,解像度を上げる.
        :param N: 1dotを何個の小さいdotで形成するか
        """
        
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
        
        dot_body=re.sub('{N}',f"{N}",dot_body)
        
        with open(f"{PARENT}/assets/detail_dot.xml","w") as f:
            f.write(dot_body)
            
    
    def make_braille_body(self,name='a',dot_pos='100000'):
        """
        名前とドット位置を指定して,点字bodyを作成する関数
        :param name: 点字の名前. ローマ字.  
        :param dot_pos: dotがある位置に1, ない位置は0.
                        0(head bit):left-top, 1:left-center, 2:left-bottom,  
                        3:right-top, 4:right-center, 5(tail-bit):right-bottom'
        """
    
        # この中のdot_existをTrueにするとポチができる。Falseにするとポチが消える。
        brailles_param:dict={
            'l-top'   :{'pos':[-b,l,z], 'size':[r,z], 'dot_exist':True},
            'l-center':{'pos':[-b,0,z], 'size':[r,z], 'dot_exist':True},
            'l-bottom':{'pos':[-b,-l,z],'size':[r,z], 'dot_exist':False},
            'r-top'   :{'pos':[b,l,z],  'size':[r,z], 'dot_exist':False},
            'r-center':{'pos':[b,0,z],  'size':[r,z], 'dot_exist':True},
            'r-bottom':{'pos':[b,-l,z], 'size':[r,z], 'dot_exist':False},
        }
        
        # >> dotの位置を引数で指定 >>
        for is_dot,key in zip(list(dot_pos),brailles_param.keys()):
            brailles_param[key]['dot_exist']=True if is_dot=='1' else False
        # >> dotの位置を引数で指定 >>
        
        
        with open(f"{PARENT}/assets/detail_dot.xml","r") as f:
            dot_body="".join(f.readlines())
        dot_bodies=""
        for key,val in brailles_param.items():    
            if not val['dot_exist']:
                continue
            dot_body_i=re.sub('{NAME}',f"{key}",dot_body)
            dot_body_i=re.sub('{XY}',f"{val['pos'][0]} {val['pos'][1]}",dot_body_i)
            dot_bodies+=dot_body_i+'\n'
            
            
        body_xml=re.sub("{NAME}",f"{name}",BRAILLE_BODY)
        body_x,body_y,body_z=2*(b+c), 2*(l+h), z
        body_xml=re.sub(
            "{XYZ}",f"{body_x} {body_y} {body_z}",
            body_xml
        )
        body_xml=re.sub('{DOT_BODIES}',dot_bodies,body_xml)
        
        
        with open(f"{PARENT}/assets/{name}.xml","w") as f:
            f.write(body_xml)
        
if __name__=="__main__":
    """
    nameとdot_posさえ入れれば,その形の点字ができる
    """
    braille_maker=BrailleMaker()
    braille_maker.raise_dot_resolution(N=240)
    braille_maker.make_braille_body(name='e',dot_pos='110100')