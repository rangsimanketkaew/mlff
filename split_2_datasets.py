#!/usr/bin/env python3

"""
Rangsiman Ketkaew
April 26th, 2020

Split a dataset into training set and test set based on defined split ratio (%)

Normal usage:
$ python split_2_dataset2.py FILE.xyz 80 20
"""

import os
import sys
import glob
import numpy as np

try:
    input = str(sys.argv[1])
except IndexError:
    exit("Error: Please specify a dataset file name")

try:
    ptrain = float(sys.argv[2])
except IndexError:
    exit("Error: Please specify ratio for training set (%), i.e. 80")
    
try:
    ptest = float(sys.argv[3])
except IndexError:
    exit("Error: Please specify ratio for test set (%), i.e. 20")

if (ptrain + ptest) != 100.0:
    exit("Error: sum of the split ratio of training and test sets must be equal to 100%")

strcts = np.loadtxt(input, delimiter='\n', dtype=str)
nlines = strcts.size
natoms = strcts[0].astype(np.int)
mols = np.array(np.split(strcts, nlines / (natoms + 2)))

ndata = mols.shape[0]
ntrain = np.int(np.ceil(ndata * ptrain / 100.0))
ntest = np.int(ndata - ntrain)

trainset = np.array(mols[:ntrain])
trainset = trainset.flatten()
testset = np.array(mols[ntrain:])
testset = testset.flatten()

# define output xyz files
filename = os.path.basename(input)
output = os.path.splitext(filename)[0]
trainout = output + '_train_' + str(ptrain) + '%_' + str(ntrain) + '.xyz'
testout = output + '_test_' + str(ptest) + '%_' + str(ntest) + '.xyz'

out = open(trainout, 'w+')
for i in range(trainset.shape[0]):
    out.write(trainset[i]+'\n')
out.close()

out = open(testout, 'w+')
for i in range(testset.shape[0]):
    out.write(testset[i]+'\n')
out.close()

print("Successfully written \'{0}\'".format(trainout))
print("Successfully written \'{0}\'".format(testout))
print("----------- Done -----------")
