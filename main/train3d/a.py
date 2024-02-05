from pathlib import Path
PARENT=Path(__file__).parent
ROOT=PARENT.parent.parent
import sys
sys.path.append(str(ROOT))
import os

from datetime import datetime

with open(f"{PARENT}/train_param_template.txt", "r") as f:
    train_param_txt="".join(f.readlines())


with open(f"{PARENT}/train_param_{datetime.now().strftime('%Y%m%d_%H.%M.%S')}.txt", "w") as f:
    f.write(train_param_txt)