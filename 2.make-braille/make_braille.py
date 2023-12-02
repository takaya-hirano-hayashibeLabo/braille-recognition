from pathlib import Path
PARENT=str(Path(__file__).parent)

import argparse
import re

def braille_xml(name,pos,size):
    """
    点字の1ポチ分のxmlを作る関数
    :param name: ポチの場所
    :param pos: ポチの場所 [x, y, z]
    :param size: ポチのサイズ [radius, height]
    """
    x,y,z=pos
    r,z=size
    xml=f"""<body pos='{x} {y} {z}' name='{name}'>
                <geom type='cylinder' size='{r} {z}'/>
            </body>"""
            
    return xml


def main():
    """
    点字のXMLファイルを作るソースコード.
    コマンドラインから'--dot_where'を与えると, 作りたい位置にdotを作成することができる.
    作るxmlファイルはbodyのみ含む（<mujoco>とか<world>は配置しない）
    """
    parser=argparse.ArgumentParser()
    parser.add_argument('--scale',default=1.0,type=float, help='scale of braile size.') # 点字を何倍のサイズにするか
    parser.add_argument('--dot_where',default='111111',type=str,
                        help='(default: --dot_where 111111)\n represent where dot exists using 6bit.\n 0(head bit):rigiht-top, 1:rigiht-center, 2:right-bottom, 3:left-top, 4:left-center, 5(tail-bit):left-bottom')
    parser.add_argument('--export_xml_name',default='test',type=str)
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
        'r-top'   :{'pos':[b,l,z],  'size':[r,z], 'dot_exist':False},
        'r-center':{'pos':[b,0,z],  'size':[r,z], 'dot_exist':True},
        'r-bottom':{'pos':[b,-l,z], 'size':[r,z], 'dot_exist':False},
        'l-top'   :{'pos':[-b,l,z], 'size':[r,z], 'dot_exist':True},
        'l-center':{'pos':[-b,0,z], 'size':[r,z], 'dot_exist':True},
        'l-bottom':{'pos':[-b,-l,z],'size':[r,z], 'dot_exist':False},
    }
    
    # >> dotの位置をコマンドライン引数で指定 >>
    dot_where=list(args.dot_where)
    for is_dot,key in zip(dot_where,brailles_param.keys()):
        brailles_param[key]['dot_exist']=True if is_dot=='1' else False
    # >> dotの位置をコマンドライン引数で指定 >>
    
    # >> XMLの作成 >>
    body_x,body_y,body_z=2*(b+c), 2*(l+h), z
    body_xml=f"""<body pos='0 0 0'>
                    <geom type='box' size='{body_x/2} {body_y/2} {body_z}' rgba='0 0.5 0 1'/>
            """
    for key,val in brailles_param.items():    
        if val['dot_exist']:
            body_xml+='\n'+braille_xml(
                key,val['pos'],val['size']
            )
    body_xml+='\n'+"""</body>"""
    # >> XMLの作成 >>
    
    # >> XMLの書き込み >>
    xml_name=re.sub('.xml','',args.export_xml_name)
    with open(f'{PARENT}/{xml_name}.xml','w',encoding='utf-8') as f:
        f.write(body_xml)
    # >> XMLの書き込み >>
    
    
if __name__=='__main__':
    main()