#!/usr/bin/env python3

"""
Rangsiman Ketkaew
April 26th, 2020

Split dataset into actual training set, validation set and test set based on defined split ratio (%)
A given dataset will be splitted into training and test sets first.
Then the training set will splitted again into actual training set and validation set

Normal usage:
$ python split_2_dataset2.py FILE.xyz 80 20 20
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
    pactrain = float(sys.argv[2])
except IndexError:
    exit("Error: Please specify ratio for training set (%), i.e. 80")

try:
    pvalid = float(sys.argv[3])
except IndexError:
    exit("Error: Please specify ratio for validation set (%), i.e. 20")

try:
    ptest = float(sys.argv[4])
except IndexError:
    exit("Error: Please specify ratio for test set (%), i.e. 20")

if (pactrain + pvalid) != 100.0:
    exit("Error: sum of the split ratio of actual training and valid sets must be equal to 100%")

strcts = np.loadtxt(input, delimiter='\n', dtype=str)
nlines = strcts.size
natoms = strcts[0].astype(np.int)
mols = np.array(np.split(strcts, nlines / (natoms + 2)))

# 1st splitting
ndata = mols.shape[0]
ntrain = np.int(np.ceil(ndata * (100.0 - ptest) / 100.0))
ntest = np.int(ndata - ntrain)

# 2nd splitting
nactrain = np.int(np.ceil(ntrain * pactrain / 100.0))
nvalid = np.int(ntrain - nactrain)

actrainset = np.array(mols[:nactrain])
actrainset = actrainset.flatten()
validset = np.array(mols[nactrain:(nactrain + nvalid)])
validset = validset.flatten()
testset = np.array(mols[:-ntest])
testset = testset.flatten()

# define output xyz files
filename = os.path.basename(input)
output = os.path.splitext(filename)[0]
trainout = output + '_actualtrain_' + str(pactrain) + '%_' + str(nactrain) + '.xyz'
validout = output + '_valid_' + str(pvalid) + '%_' + str(nvalid) + '.xyz'
testout = output + '_test_' + str(ptest) + '%_' + str(ntest) + '.xyz'

out = open(trainout, 'w+')
for i in range(actrainset.shape[0]):
    out.write(actrainset[i]+'\n')
out.close()

out = open(validout, 'w+')
for i in range(validset.shape[0]):
    out.write(validset[i]+'\n')
out.close()

out = open(testout, 'w+')
for i in range(testset.shape[0]):
    out.write(testset[i]+'\n')
out.close()

print("Successfully written \'{0}\'".format(trainout))
print("Successfully written \'{0}\'".format(validout))
print("Successfully written \'{0}\'".format(testout))
print("----------- Done -----------")
