import os
import numpy as np 
import random


def seedBasic(self,seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)




