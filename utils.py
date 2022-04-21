import os
import numpy as np 
import random


def seedBasic(self,random_state):
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)



if __name__ == "__main__":
    seedBasic(100)

    print(random.gauss(0,1))


