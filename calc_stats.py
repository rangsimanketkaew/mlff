#!/usr/bin/env python3

import sys
import numpy as np

f = sys.argv[1]

dist = np.loadtxt(f, delimiter='\n', dtype=np.float)

print("Num  : {0}".format(dist.shape[0]))
print("Max  : {0}".format(np.max(dist)))
print("Min  : {0}".format(np.min(dist)))
print("Mean : {0}".format(np.mean(dist)))
print("Var  : {0}".format(np.var(dist)))
print("Std  : {0}".format(np.std(dist)))
