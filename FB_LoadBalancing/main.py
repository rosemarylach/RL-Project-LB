import ipdb
from env.LBCellularEnv import LBCellularEnv
from Policies.UserAssociationPolicies import *
import pandas as pd
import torch

if __name__ == '__main__':

    blah = np.arange(24).reshape((4, 3, 2))
    idx = np.random.randint(low=0, high=2, size=2)
    print(idx)

    ipdb.set_trace()