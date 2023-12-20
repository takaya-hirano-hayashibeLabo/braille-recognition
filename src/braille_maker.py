from pathlib import Path
PARENT=str(Path(__file__).parent)

import numpy as np
import re

import os


MINI_DOT_GEOM="""<geom type='cylinder' size='{R} {HEIGHT}' pos='{XY} 0'/>\n"""

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
z=0.5* 10**-3 #点字の高さ
c=3.27/2 * 10**-3

BRAILLE_BODY="""<!--{NAME}.xml-->
<body pos='{X} 0 0'>
    <geom type='box' size='{XYZ}' rgba='0 0.5 0 1'/>
    {DOT_BODIES}
</body>"""


class BrailleMaker():
    def __init__(self):
        pass
    
    def raise_dot_resolution(self,N=240,dot_shape="default"):
        """
        1dotをN個のdotの集合にすることで,解像度を上げる.
        :param N: 1dotを何個の小さいdotで形成するか
        """
        
        if dot_shape=="default":
            self.raise_dot_resolution_default(N=N)
        elif dot_shape=="sloped":
            self.raise_dot_resolution_sloped(N)
        else:
            print(f"dot_shape error. can not find {dot_shape} braille...")
            exit(1)
            
        
    def raise_dot_resolution_default(self,N=240):
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
            mini_dot_geom_i=re.sub("{HEIGHT}",f"{z}",mini_dot_geom_i) #ここで点字の高さを調整できる.
            mini_dots+=mini_dot_geom_i
        dot_body=re.sub("{MINI_DOTS}",f"{mini_dots}",DOT_BODY)
        
        dot_body=re.sub('{N}',f"{N}",dot_body)
                
        with open(f"{PARENT}/assets/braille-xml/detail_dot.xml","w") as f:
            f.write(dot_body)
            
        
    def raise_dot_resolution_sloped(self,N=240):
        """
        1dotをN個のdotの集合にすることで,解像度を上げる.
        また, スロープ状にして左側が削れたような形状にする.
        :param N: 1dotを何個の小さいdotで形成するか
        """
        
        r=DOT_RADIUS*np.sqrt(np.pi/N)/2

        z_max,z_min=0.5*1e-3,0.1*1e-3
        a_z=(z_max-z_min)/DOT_RADIUS #スロープの傾き
    
        xx,yy=np.meshgrid(
            np.arange(-DOT_RADIUS,DOT_RADIUS+2*r,2*r),
            np.arange(-DOT_RADIUS,DOT_RADIUS+2*r,2*r)
        )
        
        mini_dots=""
        idx_in_dot=np.where(xx**2+yy**2<=DOT_RADIUS**2)
        for x,y in zip(xx[idx_in_dot],yy[idx_in_dot]):
            mini_dot_geom_i=re.sub("{R}",f"{r}",MINI_DOT_GEOM)
            mini_dot_geom_i=re.sub("{XY}",f"{x} {y}",mini_dot_geom_i)
            
            #>> 左側をスロープ上にする >>
            if x<0:
                z=a_z*x+z_max
            else:
                z=z_max
            mini_dot_geom_i=re.sub("{HEIGHT}",f"{z}",mini_dot_geom_i) #ここで点字の高さを調整できる.
            #>> 左側をスロープ上にする >>
            
            mini_dots+=mini_dot_geom_i
        dot_body=re.sub("{MINI_DOTS}",f"{mini_dots}",DOT_BODY)
        
        dot_body=re.sub('{N}',f"{N}",dot_body)
                
        with open(f"{PARENT}/assets/braille-xml-sloped/detail_dot.xml","w") as f:
            f.write(dot_body)        
    
    def make_braille_body(self,name='a',dot_pos_list=['100000'],dot_shape="default"):
        """
        名前とドット位置を指定して,点字bodyを作成する関数
        :param name: 点字の名前. ローマ字.  
        :param dot_shape: ドットの形状. {default:通常の高さが均一なドット. sloped:左側がスロープ上になったドット}
        :param dot_pos: dotがある位置に1, ない位置は0.
                        0(head bit):left-top, 1:left-center, 2:left-bottom,  
                        3:right-top, 4:right-center, 5(tail-bit):right-bottom'
        """
        
        # print(dot_pos_list)
        
        braille_num=len(dot_pos_list)
        braille_x=0
        braille_bodies=[]
        
        # この中のdot_existをTrueにするとポチができる。Falseにするとポチが消える。
        brailles_param:dict={
            'l-top'   :{'pos':[-b,l,z], 'size':[r,z], 'dot_exist':True},
            'l-center':{'pos':[-b,0,z], 'size':[r,z], 'dot_exist':True},
            'l-bottom':{'pos':[-b,-l,z],'size':[r,z], 'dot_exist':False},
            'r-top'   :{'pos':[b,l,z],  'size':[r,z], 'dot_exist':False},
            'r-center':{'pos':[b,0,z],  'size':[r,z], 'dot_exist':True},
            'r-bottom':{'pos':[b,-l,z], 'size':[r,z], 'dot_exist':False},
        }
    
        for i in range(braille_num):            
            
            # >> dotの位置を引数で指定 >>
            for is_dot,key in zip(list(dot_pos_list[i]),brailles_param.keys()):
                brailles_param[key]['dot_exist']=True if is_dot=='1' else False
            # >> dotの位置を引数で指定 >>
            
            
            if dot_shape=="default":
                with open(f"{PARENT}/assets/braille-xml/detail_dot.xml","r") as f:
                    dot_body="".join(f.readlines())
            elif dot_shape=="sloped":
                with open(f"{PARENT}/assets/braille-xml-sloped/detail_dot.xml","r") as f:
                    dot_body="".join(f.readlines())
            else:
                print(f"can not find {dot_shape} braille...")
                exit(1)
                    
            dot_bodies=""
            for key,val in brailles_param.items():    
                if not val['dot_exist']:
                    continue
                dot_body_j=re.sub('{NAME}',f"{key}.{i}",dot_body)
                dot_body_j=re.sub('{XY}',f"{val['pos'][0]} {val['pos'][1]}",dot_body_j)
                dot_bodies+=dot_body_j+'\n'
                
                
            # body_xml_i=re.sub("{NAME}",f"{name}",BRAILLE_BODY)
            body_xml_i=re.sub("{X}",f"{braille_x}",BRAILLE_BODY)
            body_x,body_y,body_z=2*(b+c), 2*(l+h), z*1e-3
            body_xml_i=re.sub(
                "{XYZ}",f"{body_x} {body_y} {body_z}",
                body_xml_i
            )
            body_xml_i=re.sub('{DOT_BODIES}',dot_bodies,body_xml_i)
            braille_bodies.append(body_xml_i)
            
            braille_x+=(c+b)*2 #横にずらす
            
            
        body_xml="\n".join(braille_bodies) #xmlを全結合
        body_xml=f"""<body pos='0 0 0'>
        {body_xml}
        </body>"""
        
        if dot_shape=="default":
            with open(f"{PARENT}/assets/braille-xml/{name}.xml","w") as f:
                f.write(body_xml)
        elif dot_shape=="sloped":
            with open(f"{PARENT}/assets/braille-xml-sloped/{name}.xml","w") as f:
                f.write(body_xml)
        
if __name__=="__main__":
    """
    nameとdot_posさえ入れれば,その形の点字ができる
    """
    dot_shape="sloped"
    braille_maker=BrailleMaker()
    braille_maker.raise_dot_resolution(N=240,dot_shape=dot_shape)
    braille_maker.make_braille_body(name='0',dot_pos_list=['001111','010110'],dot_shape=dot_shape)