from pathlib import Path
PARENT=str(Path(__file__).parent)

import mujoco
import mujoco.viewer
import time
import numpy as np
from copy import deepcopy

def main():
    """
    今後, 点字xmlとfingerXMLを合体するスクリプトを書く必要がある.
    """
    
    model=mujoco.MjModel.from_xml_path(f'{PARENT}/xml-resource/finger_with_touchsensor.xml') #xmlの読み込み
    data=mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model,data) as viewer:
        
        while data.time<100:
            
            step_start = time.time()
            
            #>> 人差指の先を平行にする >>
            new_qpos=deepcopy(data.qpos)
            index_finger_params=[
                {'idx':4,'qpos':0},
                {'idx':5,'qpos':-1.22e-2},
                {'idx':6,'qpos':-6.89e-4},
                {'idx':7,'qpos':0.131},
            ]
            for param in index_finger_params:
                new_qpos[param['idx']]=param['qpos']
            data.qpos=new_qpos
            #>> 人差指の先を平行にする >>
                
            mujoco.mj_step(model,data)
            
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        
if __name__=="__main__":
    main()