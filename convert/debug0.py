

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

# ignore numba warning
import warnings
warnings.filterwarnings('ignore')
import random
import datetime
import time
import numpy as np
import torch
import pickle
from numpy import unravel_index


import shapely
from shapely.geometry import Polygon


if __name__ == "__main__":
    pt1 =  [247.20982 , 733.4407,   356.6556 ,  729.7164 ,  363.7201 ,  937.3208,
  254.27432 , 941.0451  ]
    pt1 = np.array(pt1).reshape(-1  ,2)
    ploly1 = Polygon(pt1)
    
    #shapely
    print('ploly1', ploly1)

